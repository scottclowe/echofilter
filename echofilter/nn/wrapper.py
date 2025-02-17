"""
Model wrapper.
"""

# This file is part of Echofilter.
#
# Copyright (C) 2020-2022  Scott C. Lowe and Offshore Energy Research Association (OERA)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from .utils import TensorDict, logavgexp


class Echofilter(nn.Module):
    """
    Echofilter logit mapping wrapper.

    Parameters
    ----------
    model : :class:`torch.nn.Module`
        The model backbone, which converts inputs to logits.
    top : str, optional
        Type of output for top line and surface line. If ``"mask"``, the top
        output corresponds to logits, which are converted into probabilities
        with sigmoid. If ``"boundary"`` (default), the output corresponds to
        logits for the location of the line, which is converted into a
        probability mask using softmax and cumsum.
    bottom : str, optional
        As for ``top``, but for the bottom line. Default is ``"boundary"``.
    mapping : dict or None, optional
        Mapping from logit names to output channels provided by ``model``.
        If ``None``, a default mapping is used. The mapping is stored as
        ``self.mapping``.
    reduction_ispassive : str, default="logavgexp"
        Method used to reduce the depths dimension for the ``"logit_is_passive"``
        output.
    reduction_isremoved : str , default="logavgexp"
        Method used to reduce the depths dimension for the ``"logit_is_removed"``
        output.
    conditional : bool, optional
        Whether to build a conditional model as well as an unconditional model.
        If ``True``, there are additional logits in the call output named
        ``"x|downfacing"`` and ``"x|upfacing"``, in addition to
        ``"x"``. For instance, ``"p_is_above_turbulence|downfacing"``. Default is ``False``.
    """

    aliases = [("top", "turbulence")]

    def __init__(
        self,
        model,
        top="boundary",
        bottom="boundary",
        mapping=None,
        reduction_ispassive="logavgexp",
        reduction_isremoved="logavgexp",
        conditional=False,
    ):
        super(Echofilter, self).__init__()
        self.model = model
        self.params = {
            "top": top,
            "bottom": bottom,
            "reduction_ispassive": reduction_ispassive,
            "reduction_isremoved": reduction_isremoved,
            "conditional": conditional,
        }
        if mapping is None:
            mapping = {
                "logit_is_above_turbulence": 0,
                "logit_is_boundary_turbulence": 0,
                "logit_is_below_bottom": 1,
                "logit_is_boundary_bottom": 1,
                "logit_is_removed": 2,
                "logit_is_passive": 3,
                "logit_is_patch": 4,
                "logit_is_above_surface": 5,
                "logit_is_boundary_surface": 5,
                "logit_is_above_turbulence-original": 6,
                "logit_is_boundary_turbulence-original": 6,
                "logit_is_below_bottom-original": 7,
                "logit_is_boundary_bottom-original": 7,
                "logit_is_patch-original": 8,
                "logit_is_patch-ntob": 9,
            }
            if top == "boundary":
                mapping.pop("logit_is_above_turbulence")
                mapping.pop("logit_is_above_turbulence-original")
                mapping.pop("logit_is_above_surface")
            else:
                mapping.pop("logit_is_boundary_turbulence")
                mapping.pop("logit_is_boundary_turbulence-original")
                mapping.pop("logit_is_boundary_surface")
            if bottom == "boundary":
                mapping.pop("logit_is_below_bottom")
                mapping.pop("logit_is_below_bottom-original")
            else:
                mapping.pop("logit_is_boundary_bottom")
                mapping.pop("logit_is_boundary_bottom-original")
        self.mapping = mapping
        # Ensure all references for aliases are set
        mapping_extra = {}
        for key in mapping:
            for alias_map in self.aliases:
                for alias_a, alias_b in [alias_map, alias_map[::-1]]:
                    if "_" + alias_a not in key:
                        continue
                    alt_key = key.replace("_" + alias_a, "_" + alias_b)
                    if alt_key not in mapping:
                        mapping_extra[alt_key] = mapping[key]
        mapping.update(mapping_extra)

        self.conditions = [""]
        if conditional:
            self.conditions += ["downfacing", "upfacing"]
        self.n_outputs_per_condition = max(self.mapping.values())

    def forward(self, x, output_device=None):
        logits = self.model(x)
        outputs = TensorDict()

        for i_condition, condition in enumerate(self.conditions):
            # Define the condition string
            cs = condition
            if cs != "":
                cs = "|" + cs
            # Define the logit index offset
            ofs = i_condition * self.n_outputs_per_condition

            # Include raw logits in output
            for key, index in self.mapping.items():
                outputs[key + cs] = logits[:, index + ofs]

            # Flatten some outputs which are vectors not arrays
            if self.params["reduction_isremoved"] == "mean":
                outputs["logit_is_removed" + cs] = torch.mean(
                    outputs["logit_is_removed" + cs], dim=-1
                ).to(device=output_device)
            elif self.params["reduction_isremoved"] in {"logavgexp", "lae"}:
                outputs["logit_is_removed" + cs] = logavgexp(
                    outputs["logit_is_removed" + cs], dim=-1
                ).to(device=output_device)
            else:
                raise ValueError(
                    "Unsupported reduction_isremoved value: {}".format(
                        self.params["reduction_isremoved"]
                    )
                )

            if self.params["reduction_ispassive"] == "mean":
                outputs["logit_is_passive" + cs] = torch.mean(
                    outputs["logit_is_passive" + cs], dim=-1
                ).to(device=output_device)
            elif self.params["reduction_ispassive"] in {"logavgexp", "lae"}:
                outputs["logit_is_passive" + cs] = logavgexp(
                    outputs["logit_is_passive" + cs], dim=-1
                ).to(device=output_device)
            else:
                raise ValueError(
                    "Unsupported reduction_ispassive value: {}".format(
                        self.params["reduction_ispassive"]
                    )
                )

            # Convert logits to probabilities
            outputs["p_is_removed" + cs] = torch.sigmoid(
                outputs["logit_is_removed" + cs]
            ).to(device=output_device)
            outputs["p_is_passive" + cs] = torch.sigmoid(
                outputs["logit_is_passive" + cs]
            ).to(device=output_device)

            for sfx in ("turbulence", "turbulence-original", "surface"):
                if self.params["top"] == "mask":
                    outputs["p_is_above_" + sfx + cs] = torch.sigmoid(
                        outputs["logit_is_above_" + sfx + cs]
                    ).to(device=output_device)
                    outputs["p_is_below_" + sfx + cs] = (
                        1 - outputs["p_is_above_" + sfx + cs]
                    ).to(device=output_device)
                elif self.params["top"] == "boundary":
                    outputs["p_is_boundary_" + sfx + cs] = F.softmax(
                        outputs["logit_is_boundary_" + sfx + cs], dim=-1
                    ).to(device=output_device)
                    outputs["p_is_above_" + sfx + cs] = torch.flip(
                        torch.cumsum(
                            torch.flip(
                                outputs["p_is_boundary_" + sfx + cs], dims=(-1,)
                            ),
                            dim=-1,
                        ),
                        dims=(-1,),
                    ).to(device=output_device)
                    outputs["p_is_below_" + sfx + cs] = torch.cumsum(
                        outputs["p_is_boundary_" + sfx + cs], dim=-1
                    ).to(device=output_device)
                    # Due to floating point precision, max value can exceed 1.
                    # Fix this by clipping the values to the appropriate range.
                    outputs["p_is_above_" + sfx + cs].clamp_(0, 1)
                    outputs["p_is_below_" + sfx + cs].clamp_(0, 1)
                else:
                    raise ValueError(
                        'Unsupported "top" parameter: {}'.format(self.params["top"])
                    )

            for sfx in ("bottom", "bottom-original"):
                if self.params["bottom"] == "mask":
                    outputs["p_is_below_" + sfx + cs] = torch.sigmoid(
                        outputs["logit_is_below_" + sfx + cs]
                    ).to(device=output_device)
                    outputs["p_is_above_" + sfx + cs] = (
                        1 - outputs["p_is_below_" + sfx + cs]
                    ).to(device=output_device)
                elif self.params["bottom"] == "boundary":
                    outputs["p_is_boundary_" + sfx + cs] = F.softmax(
                        outputs["logit_is_boundary_" + sfx + cs], dim=-1
                    ).to(device=output_device)
                    outputs["p_is_below_" + sfx + cs] = torch.cumsum(
                        outputs["p_is_boundary_" + sfx + cs], dim=-1
                    ).to(device=output_device)
                    outputs["p_is_above_" + sfx + cs] = torch.flip(
                        torch.cumsum(
                            torch.flip(
                                outputs["p_is_boundary_" + sfx + cs], dims=(-1,)
                            ),
                            dim=-1,
                        ),
                        dims=(-1,),
                    ).to(device=output_device)
                    # Due to floating point precision, max value can exceed 1.
                    # Fix this by clipping the values to the appropriate range.
                    outputs["p_is_below_" + sfx + cs].clamp_(0, 1)
                    outputs["p_is_above_" + sfx + cs].clamp_(0, 1)
                else:
                    raise ValueError(
                        'Unsupported "bottom" parameter: {}'.format(
                            self.params["bottom"]
                        )
                    )

            for sfx in ("", "-original", "-ntob"):
                outputs["p_is_patch" + sfx + cs] = torch.sigmoid(
                    outputs["logit_is_patch" + sfx + cs]
                ).to(device=output_device)

            outputs["p_keep_pixel" + cs] = (
                1.0
                * 0.5
                * (
                    (1 - outputs["p_is_above_turbulence" + cs])
                    + outputs["p_is_below_turbulence" + cs]
                )
                * 0.5
                * (
                    (1 - outputs["p_is_below_bottom" + cs])
                    + outputs["p_is_above_bottom" + cs]
                )
                * (1 - outputs["p_is_removed" + cs].unsqueeze(-1))
                * (1 - outputs["p_is_passive" + cs].unsqueeze(-1))
                * (1 - outputs["p_is_patch" + cs])
            ).clamp_(0, 1)
            outputs["mask_keep_pixel" + cs] = (
                1.0
                * (outputs["p_is_above_turbulence" + cs] < 0.5)
                * (outputs["p_is_below_bottom" + cs] < 0.5)
                * (outputs["p_is_removed" + cs].unsqueeze(-1) < 0.5)
                * (outputs["p_is_passive" + cs].unsqueeze(-1) < 0.5)
                * (outputs["p_is_patch" + cs] < 0.5)
            )
        return outputs


class EchofilterLoss(_Loss):
    """
    Evaluate loss for an Echofilter model.

    Parameters
    ----------
    reduction : ``"mean"`` or ``"sum"``, optional
        The reduction method, which is used to collapse batch and timestamp
        dimensions. Default is ``"mean"``.
    turbulence_mask : float, optional
        Weighting for turbulence line/mask loss term. Default is ``1.0``.
    bottom_mask : float, optional
        Weighting for bottom line/mask loss term. Default is ``1.0``.
    removed_segment : float, optional
        Weighting for ``is_removed`` loss term. Default is ``1.0``.
    passive : float, optional
        Weighting for ``is_passive`` loss term. Default is ``1.0``.
    patch : float, optional
        Weighting for ``mask_patch`` loss term. Default is ``1.0``.
    overall : float, optional
        Weighting for overall mask loss term. Default is ``0.0``.
    surface : float, optional
        Weighting for surface line/mask loss term. Default is ``1.0``.
    auxiliary : float, optional
        Weighting for auxiliary loss terms ``"turbulence-original"``,
        ``"bottom-original"``, ``"mask_patches-original"``, and
        ``"mask_patches-ntob"``. Default is ``1.0``.
    ignore_lines_during_passive : bool, optional
        Whether targets for turbulence and bottom lines should be excluded from
        the loss during passive data collection. Default is ``True``.
    ignore_lines_during_removed : bool, optional
        Whether targets for turbulence and bottom lines should be excluded from
        the loss during entirely removed sections. Default is ``True``.
    ignore_surface_during_passive : bool, optional
        Whether target for the surface line should be excluded from the loss
        during passive data collection. Default is ``False``.
    ignore_surface_during_removed : bool, optional
        Whether target for the surface line should be excluded from the loss
        during entirely removed sections. Default is ``True``.
    """

    __constants__ = ["reduction"]

    def __init__(
        self,
        reduction="mean",
        conditional=False,
        turbulence_mask=1.0,
        bottom_mask=1.0,
        removed_segment=1.0,
        passive=1.0,
        patch=1.0,
        overall=0.0,
        surface=1.0,
        auxiliary=1.0,
        ignore_lines_during_passive=False,
        ignore_lines_during_removed=True,
        ignore_surface_during_passive=False,
        ignore_surface_during_removed=True,
    ):
        super(EchofilterLoss, self).__init__(None, None, reduction)
        self.conditional = conditional
        self.turbulence_mask = turbulence_mask
        self.bottom_mask = bottom_mask
        self.removed_segment = removed_segment
        self.passive = passive
        self.patch = patch
        self.overall = overall
        self.surface = surface
        self.auxiliary = auxiliary
        self.ignore_lines_during_passive = ignore_lines_during_passive
        self.ignore_lines_during_removed = ignore_lines_during_removed
        self.ignore_surface_during_passive = ignore_surface_during_passive
        self.ignore_surface_during_removed = ignore_surface_during_removed

        self.conditions = [""]
        if conditional:
            self.conditions += ["downfacing", "upfacing"]

    def forward(self, input, target):
        """
        Construct loss term.

        Parameters
        ----------
        input : dict
            Output from :class:`echofilter.wrapper.Echofilter` layer.
        target : dict
            A transect, as provided by :class:`echofilter.data.dataset.TransectDataset`.
        """
        loss = 0

        target["is_passive"] = target["is_passive"].to(
            input["logit_is_passive"].device,
            input["logit_is_passive"].dtype,
        )
        target["is_removed"] = target["is_removed"].to(
            input["logit_is_removed"].device,
            input["logit_is_removed"].dtype,
        )

        n_conditions_in_loss = 0
        for condition in self.conditions:
            closs = 0
            with torch.no_grad():
                if condition == "":
                    cs = condition
                    cmask = torch.ones_like(target["is_upward_facing"])
                else:
                    cs = "|" + condition
                    if condition == "upfacing":
                        cmask = target["is_upward_facing"] > 0.5
                    elif condition == "downfacing":
                        cmask = target["is_upward_facing"] < 0.5
                    else:
                        raise ValueError("Unsupported condition: {}".format(condition))
                    n_samples_in_condition = torch.sum(cmask)
                    if n_samples_in_condition.cpu().item() == 0:
                        # No samples in this batch match this condition
                        continue
                cmask = cmask.to(torch.float32)

            n_conditions_in_loss += 1

            for sfx in ("turbulence", "turbulence-original", "surface"):
                with torch.no_grad():
                    loss_inclusion_mask = (target["is_bad_labels"] < 1e-7).to(
                        torch.float32
                    )
                    if sfx == "surface":
                        target_key = "mask_surface"
                        target_i_key = "index_surface"
                        weight = self.surface
                        # Don't include surrogate surface datapoints in loss
                        loss_inclusion_mask *= (
                            target["is_surrogate_surface"] < 1e-7
                        ).to(torch.float32)
                        # Check whether surface line is masked out
                        if self.ignore_surface_during_passive:
                            loss_inclusion_mask *= 1 - target["is_passive"]
                        if self.ignore_surface_during_removed:
                            loss_inclusion_mask *= 1 - target["is_removed"]
                    else:
                        target_key = "mask_" + sfx
                        target_i_key = "index_" + sfx
                        weight = self.turbulence_mask
                        if sfx != "turbulence":
                            weight *= self.auxiliary
                        # Check whether line is masked out
                        if self.ignore_lines_during_passive:
                            loss_inclusion_mask *= 1 - target["is_passive"]
                        if self.ignore_lines_during_removed:
                            loss_inclusion_mask *= 1 - target["is_removed"]

                if not weight:
                    continue
                elif "logit_is_boundary_" + sfx in input:
                    # Load cross-entropy class target
                    C = target[target_i_key].to(
                        device=input["logit_is_boundary_" + sfx + cs].device,
                        dtype=torch.long,
                    )
                    loss_term = F.cross_entropy(
                        input["logit_is_boundary_" + sfx + cs].transpose(-2, -1),
                        C,
                        reduction="none",
                    )
                    loss_term *= cmask.unsqueeze(-1)
                    loss_term *= loss_inclusion_mask
                    if self.reduction == "mean":
                        loss_term = torch.mean(loss_term)
                    elif self.reduction == "sum":
                        loss_term = torch.sum(loss_term)
                    elif self.reduction != "none":
                        raise ValueError(
                            "Unsupported reduction: {}".format(self.reduction)
                        )
                elif "logit_is_above_" + sfx in input:
                    warnings.warn(
                        'Using loss corresponding to "mask" logits.'
                        ' The "boundary" is recommended instead.'
                        " The loss component for this line will be"
                        " F.binary_cross_entropy_with_logits(input[{}], target[{}])"
                        "".format("logit_is_above_" + sfx + cs, target_key)
                    )
                    loss_term = F.binary_cross_entropy_with_logits(
                        input["logit_is_above_" + sfx + cs],
                        target[target_key].to(
                            input["logit_is_above_" + sfx + cs].device,
                            input["logit_is_above_" + sfx + cs].dtype,
                        ),
                        reduction="none",
                    )
                    loss_term *= cmask.unsqueeze(-1).unsqueeze(-1)
                    loss_term *= loss_inclusion_mask.unsqueeze(-1)
                    if self.reduction == "mean":
                        loss_term = torch.mean(loss_term)
                    elif self.reduction == "sum":
                        loss_term = torch.sum(loss_term)
                    elif self.reduction != "none":
                        raise ValueError(
                            "Unsupported reduction: {}".format(self.reduction)
                        )
                else:
                    raise ValueError(
                        "The input does not contain either {} or {} fields."
                        " At least one of these is required if the loss term weighting"
                        " is non-zero.".format(
                            "logit_is_boundary_" + sfx,
                            "logit_is_above_" + sfx,
                        )
                    )
                if torch.isnan(loss_term).any():
                    print("Loss term {} is NaN".format(target_key))
                else:
                    closs += weight * loss_term

            for sfx in ("", "-original"):
                weight = self.bottom_mask
                if sfx != "":
                    weight *= self.auxiliary

                with torch.no_grad():
                    loss_inclusion_mask = (target["is_bad_labels"] < 1e-7).to(
                        torch.float32
                    )
                    # Check whether line is masked out
                    if self.ignore_lines_during_passive:
                        loss_inclusion_mask *= 1 - target["is_passive"]
                    if self.ignore_lines_during_removed:
                        loss_inclusion_mask *= 1 - target["is_removed"]

                if not weight:
                    continue
                elif "logit_is_boundary_bottom" + sfx in input:
                    # Load cross-entropy class target
                    C = target["index_bottom" + sfx].to(
                        device=input["logit_is_boundary_bottom" + sfx + cs].device,
                        dtype=torch.long,
                    )
                    loss_term = F.cross_entropy(
                        input["logit_is_boundary_bottom" + sfx + cs].transpose(-2, -1),
                        C,
                        reduction="none",
                    )
                    loss_term *= cmask.unsqueeze(-1)
                    loss_term *= loss_inclusion_mask
                    if self.reduction == "mean":
                        loss_term = torch.mean(loss_term)
                    elif self.reduction == "sum":
                        loss_term = torch.sum(loss_term)
                    elif self.reduction != "none":
                        raise ValueError(
                            "Unsupported reduction: {}".format(self.reduction)
                        )
                elif "logit_is_below_bottom" + sfx in input:
                    warnings.warn(
                        'Using loss corresponding to "mask" logits.'
                        ' The "boundary" is recommended instead.'
                        " The loss component for this line will be"
                        " F.binary_cross_entropy_with_logits(input[{}], target[{}])"
                        "".format("logit_is_below_bottom" + sfx + cs, target_key)
                    )
                    loss_term = F.binary_cross_entropy_with_logits(
                        input["logit_is_below_bottom" + sfx + cs],
                        target["mask_bottom" + sfx].to(
                            input["logit_is_below_bottom" + sfx + cs].device,
                            input["logit_is_below_bottom" + sfx + cs].dtype,
                        ),
                        reduction="none",
                    )
                    loss_term *= cmask.unsqueeze(-1).unsqueeze(-1)
                    loss_term *= loss_inclusion_mask.unsqueeze(-1)
                    if self.reduction == "mean":
                        loss_term = torch.mean(loss_term)
                    elif self.reduction == "sum":
                        loss_term = torch.sum(loss_term)
                    elif self.reduction != "none":
                        raise ValueError(
                            "Unsupported reduction: {}".format(self.reduction)
                        )
                else:
                    raise ValueError(
                        "The input does not contain either {} or {} fields."
                        " At least one of these is required if the loss term weighting"
                        " is non-zero.".format(
                            "logit_is_boundary_bottom" + sfx,
                            "logit_is_below_bottom" + sfx,
                        )
                    )
                if torch.isnan(loss_term).any():
                    print("Loss term mask_bottom{} is NaN".format(sfx))
                else:
                    closs += weight * loss_term

            if self.removed_segment:
                loss_term = F.binary_cross_entropy_with_logits(
                    input["logit_is_removed" + cs],
                    target["is_removed"].to(
                        input["logit_is_removed" + cs].device,
                        input["logit_is_removed" + cs].dtype,
                    ),
                    reduction="none",
                )
                loss_term *= cmask.unsqueeze(-1)
                if self.reduction == "mean":
                    loss_term = torch.mean(loss_term)
                elif self.reduction == "sum":
                    loss_term = torch.sum(loss_term)
                elif self.reduction != "none":
                    raise ValueError("Unsupported reduction: {}".format(self.reduction))
                if torch.isnan(loss_term).any():
                    print("Loss term is_removed is NaN")
                else:
                    closs += self.removed_segment * loss_term

            if self.passive:
                loss_term = self.passive * F.binary_cross_entropy_with_logits(
                    input["logit_is_passive" + cs],
                    target["is_passive"].to(
                        input["logit_is_passive" + cs].device,
                        input["logit_is_passive" + cs].dtype,
                    ),
                    reduction="none",
                )
                loss_term *= cmask.unsqueeze(-1)
                if self.reduction == "mean":
                    loss_term = torch.mean(loss_term)
                elif self.reduction == "sum":
                    loss_term = torch.sum(loss_term)
                elif self.reduction != "none":
                    raise ValueError("Unsupported reduction: {}".format(self.reduction))
                if torch.isnan(loss_term).any():
                    print("Loss term is_passive is NaN")
                else:
                    closs += self.passive * loss_term

            for sfx in ("", "-original", "-ntob"):
                weight = self.patch
                if sfx != "":
                    weight *= self.auxiliary
                if not weight:
                    continue
                loss_term = F.binary_cross_entropy_with_logits(
                    input["logit_is_patch" + sfx + cs],
                    target["mask_patches" + sfx].to(
                        input["logit_is_patch" + sfx + cs].device,
                        input["logit_is_patch" + sfx + cs].dtype,
                    ),
                    reduction="none",
                )
                loss_term *= cmask.unsqueeze(-1).unsqueeze(-1)
                if self.reduction == "mean":
                    loss_term = torch.mean(loss_term)
                elif self.reduction == "sum":
                    loss_term = torch.sum(loss_term)
                elif self.reduction != "none":
                    raise ValueError("Unsupported reduction: {}".format(self.reduction))
                if torch.isnan(loss_term).any():
                    print("Loss term mask_patches{} is NaN".format(sfx))
                else:
                    closs += weight * loss_term

            if self.overall:
                loss_term = self.overall * F.binary_cross_entropy(
                    input["p_keep_pixel" + cs],
                    target["mask"].to(
                        input["p_keep_pixel" + cs].device,
                        input["p_keep_pixel" + cs].dtype,
                    ),
                    reduction="none",
                )
                loss_term *= cmask.unsqueeze(-1).unsqueeze(-1)
                if self.reduction == "mean":
                    loss_term = torch.mean(loss_term)
                elif self.reduction == "sum":
                    loss_term = torch.sum(loss_term)
                elif self.reduction != "none":
                    raise ValueError("Unsupported reduction: {}".format(self.reduction))
                if torch.isnan(loss_term).any():
                    print("Loss term overall is NaN")
                else:
                    closs += self.overall * loss_term

            loss += closs

        # Avoid double-counting the loss
        if n_conditions_in_loss > 1:
            loss /= 2

        return loss
