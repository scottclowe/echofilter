#!/usr/bin/env python

"""
Inference routine.
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

import datetime
import os
import pprint
import sys
import tempfile
import textwrap
import time
import warnings

import numpy as np
import torch
import torch.nn
import torch.utils.data
import torchvision.transforms
from matplotlib import colors as mcolors
from tqdm.auto import tqdm

import echofilter.data.transforms
import echofilter.nn
import echofilter.path
import echofilter.raw
import echofilter.ui
import echofilter.ui.checkpoints
import echofilter.utils
import echofilter.win
from echofilter.nn.unet import UNet
from echofilter.nn.utils import count_parameters
from echofilter.nn.wrapper import Echofilter
from echofilter.raw.manipulate import join_transect, pad_transect, split_transect
from echofilter.raw.utils import fillholes2d
from echofilter.ui.inference_cli import DEFAULT_VARNAME, cli, main

EV_UNDEFINED_DEPTH = -10000.99


def run_inference(
    paths,
    source_dir=".",
    recursive_dir_search=True,
    extensions="csv",
    skip_existing=False,
    skip_incompatible=False,
    output_dir="",
    dry_run=False,
    continue_on_error=False,
    overwrite_existing=False,
    overwrite_ev_lines=False,
    import_into_evfile=True,
    generate_turbulence_line=True,
    generate_bottom_line=True,
    generate_surface_line=True,
    add_nearfield_line=True,
    suffix_file="",
    suffix_var=None,
    color_turbulence="orangered",
    color_turbulence_offset=None,
    color_bottom="orangered",
    color_bottom_offset=None,
    color_surface="green",
    color_surface_offset=None,
    color_nearfield="mediumseagreen",
    thickness_turbulence=2,
    thickness_turbulence_offset=None,
    thickness_bottom=2,
    thickness_bottom_offset=None,
    thickness_surface=1,
    thickness_surface_offset=None,
    thickness_nearfield=1,
    cache_dir=None,
    cache_csv=None,
    suffix_csv="",
    keep_ext=False,
    line_status=3,
    offset_turbulence=1.0,
    offset_bottom=1.0,
    offset_surface=1.0,
    nearfield=1.7,
    cutoff_at_nearfield=None,
    lines_during_passive="interpolate-time",
    collate_passive_length=10,
    collate_removed_length=10,
    minimum_passive_length=10,
    minimum_removed_length=-1,
    minimum_patch_area=-1,
    patch_mode=None,
    variable_name=DEFAULT_VARNAME,
    export_raw_csv=True,
    row_len_selector="mode",
    facing="auto",
    use_training_standardization=False,
    prenorm_nan_value=None,
    postnorm_nan_value=None,
    crop_min_depth=None,
    crop_max_depth=None,
    autocrop_threshold=0.35,
    image_height=None,
    checkpoint=None,
    force_unconditioned=False,
    logit_smoothing_sigma=0,
    device=None,
    hide_echoview="never",
    minimize_echoview=True,
    verbose=2,
):
    """
    Perform inference on input files, and generate output files.

    Outputs are written as lines in EVL and regions in EVR file formats.

    Parameters
    ----------
    paths : iterable or str
        Files and folders to be processed. These may be full paths or paths
        relative to ``source_dir``. For each folder specified, any files with
        extension ``"csv"`` within the folder and all its tree of subdirectories
        will be processed.
    source_dir : str, default="."
        Path to directory where files are found.
    recursive_dir_search : bool, default=True
        How to handle directory inputs in ``paths``. If ``False``, only files
        (with the correct extension) in the directory will be included.
        If ``True``, subdirectories will also be walked through to find input
        files.
    extensions : iterable or str, default="csv"
        File extensions to detect when running on a directory.
    skip_existing : bool, default=False
        Skip processing files which already have all outputs present.
    skip_incompatible : bool, default=False
        Skip processing CSV files which do not seem to contain an exported
        Echoview transect. If ``False``, an error is raised.
    output_dir : str, default=""
        Directory where output files will be written. If this is an empty
        string, outputs are written to the same directory as
        each input file. Otherwise, they are written to ``output_dir``,
        preserving their path relative to ``source_dir`` if relative paths were
        used.
    dry_run : bool, default=False
        If ``True``, perform a trial run with no changes made.
    continue_on_error : bool, default=False
        Continue running on remaining files if one file hits an error.

        .. versionadded:: 1.1.0

    overwrite_existing : bool, default=False
        Overwrite existing outputs without producing a warning message. If
        ``False``, an error is generated if files would be overwritten.
    overwrite_ev_lines : bool, default=False
        Overwrite existing lines within the Echoview file without warning.
        If ``False`` (default), the current datetime will be appended to line
        variable names in the event of a collision.
    import_into_evfile : bool, default=True
        Whether to import the output lines and regions into the EV file,
        whenever the file being processed in an EV file.
    generate_turbulence_line : bool, default=True
        Whether to output an evl file for the turbulence line. If this is
        ``False``, the turbulence line is also never imported into Echoview.
    generate_bottom_line : bool, default=True
        Whether to output an evl file for the bottom line. If this is ``False``,
        the bottom line is also never imported into Echoview.
    generate_surface_line : bool, default=True
        Whether to output an evl file for the surface line. If this is ``False``,
        the surface line is also never imported into Echoview.
    add_nearfield_line : bool, default=True
        Whether to add a nearfield line to the EV file in Echoview.
    suffix_file : str, default=""
        Suffix to append to output artifacts (evl and evr files), between
        the name of the file and the extension. If ``suffix_file`` begins with
        an alphanumeric character, ``"-"`` is prepended.
    suffix_var : str, optional
        Suffix to append to line and region names when imported back into
        EV file. If ``suffix_var`` begins with an alphanumeric character, ``"-"``
        is prepended. By default, ``suffix_var`` will match ``suffix_file``
        if it is set, and will be "_echofilter" otherwise.
    color_turbulence : str, default="orangered"
        Color to use for the turbulence line when it is imported into Echoview.
        This can either be the name of a supported color from
        matplotlib.colors, or a hexadecimal color, or a string representation
        of an RGB color to supply directly to Echoview (such as "(0,255,0)").
    color_turbulence_offset : str, optional
        Color to use for the offset turbulence line when it is imported into
        Echoview. By default, ``color_turbulence`` is used.
    color_bottom : str, default="orangered"
        Color to use for the bottom line when it is imported into Echoview.
        This can either be the name of a supported color from
        matplotlib.colors, or a hexadecimal color, or a string representation
        of an RGB color to supply directly to Echoview (such as "(0,255,0)").
    color_bottom_offset : str, optional
        Color to use for the offset bottom line when it is imported into
        Echoview. By default, ``color_bottom`` is used.
    color_surface : str, default="green"
        Color to use for the surface line when it is imported into Echoview.
        This can either be the name of a supported color from
        matplotlib.colors, or a hexadecimal color, or a string representation
        of an RGB color to supply directly to Echoview (such as "(0,255,0)").
    color_surface_offset : str, optional
        Color to use for the offset surface line when it is imported into
        Echoview. By default, ``color_surface`` is used.
    color_nearfield : str, default="mediumseagreen"
        Color to use for the nearfield line when it is created in Echoview.
        This can either be the name of a supported color from
        matplotlib.colors, or a hexadecimal color, or a string representation
        of an RGB color to supply directly to Echoview (such as "(0,255,0)").
    thickness_turbulence : int, default=2
        Thickness with which the turbulence line will be displayed in Echoview.
    thickness_turbulence_offset : str, optional
        Thickness with which the offset turbulence line will be displayed in
        Echoview. By default, ``thickness_turbulence`` is used.
    thickness_bottom : int, default=2
        Thickness with which the bottom line will be displayed in Echoview.
    thickness_bottom_offset : str, optional
        Thickness with which the offset bottom line will be displayed in
        Echoview. By default, ``thickness_bottom`` is used.
    thickness_surface : int, default=1
        Thickness with which the surface line will be displayed in Echoview.
    thickness_surface_offset : str, optional
        Thickness with which the offset surface line will be displayed in
        Echoview. By default, ``thickness_surface`` is used.
    thickness_nearfield : int, default=1
        Thickness with which the nearfield line will be displayed in Echoview.
    cache_dir : str, optional
        Path to directory where downloaded checkpoint files should be cached.
        By default, an OS-appropriate application-specific default
        cache directory is used.
    cache_csv : str, optional
        Path to directory where CSV files generated from EV inputs should be
        cached. By default, EV files which are exported to CSV files
        are temporary files, deleted after this program has completed. If
        ``cache_csv=""``, the CSV files are cached in the same directory as the
        input EV files.
    suffix_csv : str, default=""
        Suffix used for cached CSV files which are exported from EV files.
        If ``suffix_file`` begins with an alphanumeric character, a delimiter
        is prepended. The delimiter is ``"."`` if ``keep_ext=True`` or ``"-"`` if
        ``keep_ext=False``.
    keep_ext : bool, default=False
        Whether to preserve the file extension in the input file name when
        generating output file name. Default is ``False``, removing the
        extension.
    line_status : int, default=3
        Status to use for the lines.
        Must be one of:

        - ``0`` : none
        - ``1`` : unverified
        - ``2`` : bad
        - ``3`` : good

    offset_turbulence : float, default=1.0
        Offset for turbulence line, which moves the turbulence line deeper.
    offset_bottom : float, default=1.0
        Offset for bottom line, which moves the line to become more shallow.
    offset_surface : float, default=1.0
        Offset for surface line, which moves the surface line deeper.
    nearfield : float, default=1.7
        Nearfield approach distance, in metres.
        If the echogram is downward facing, the nearfield cutoff depth
        will be at a depth equal to the nearfield distance.
        If the echogram is upward facing, the nearfield cutoff will be
        ``nearfield`` meters above the deepest depth recorded in the input
        data.
        When processing an EV file, by default a nearfield line will be
        added at the nearfield cutoff depth. To prevent this behaviour,
        use the --no-nearfield-line argument.
    cutoff_at_nearfield : bool, optional
        Whether to cut-off the turbulence line (for downfacing data) or bottom
        line (for upfacing) when it is closer to the echosounder than the
        ``nearfield`` distance.
        By default, the bottom line is clipped (for upfacing data),
        but the turbulence line is not clipped (even with downfacing data).
    lines_during_passive : str, default="interpolate-time"
        Method used to handle line depths during collection
        periods determined to be passive recording instead of
        active recording.
        Options are:

        ``"interpolate-time"``
            depths are linearly interpolated from active
            recording periods, using the time at which
            recordings where made.
        ``"interpolate-index"``
            depths are linearly interpolated from active
            recording periods, using the index of the
            recording.
        ``"predict"``
            the model's prediction for the lines during
            passive data collection will be kept; the nature
            of the prediction depends on how the model was
            trained.
        ``"redact"``
            no depths are provided during periods determined
            to be passive data collection.
        ``"undefined"``
            depths are replaced with the placeholder value
            used by Echoview to denote undefined values,
            which is ``-10000.99``.

    collate_passive_length : int, default=10
        Maximum interval, in ping indices, between detected passive regions
        which will removed to merge consecutive passive regions together
        into a single, collated, region.
    collate_passive_length : int, default=10
        Maximum interval, in ping indices, between detected blocks
        (vertical rectangles) marked for removal which will also be removed
        to merge consecutive removed blocks together into a single,
        collated, region.
    minimum_passive_length : int, default=10
        Minimum length, in ping indices, which a detected passive region
        must have to be included in the output. Set to ``-1`` to omit all
        detected passive regions from the output.
    minimum_removed_length : int, default=-1
        Minimum length, in ping indices, which a detected removal block
        (vertical rectangle) must have to be included in the output.
        Set to -1 to omit all detected removal blocks from the output
        (default). Recommended minimum length is 10.
    minimum_patch_area : int, default=-1
        Minimum area, in pixels, which a detected removal patch
        (contour/polygon) region must have to be included in the output.
        Set to -1 to omit all detected patches from the output (default).
        Recommended minimum length 25.
    patch_mode : str, optional
        Type of mask patches to use. Must be supported by the
        model checkpoint used. Should be one of:

        ``"merged"``
            Target patches for training were determined
            after merging as much as possible into the
            turbulence and bottom lines.
        ``"original"``
            Target patches for training were determined
            using original lines, before expanding the
            turbulence and bottom lines.
        ``"ntob"``
            Target patches for training were determined
            using the original bottom line and the merged
            turbulence line.

        By default, ``"merged"`` is used if downfacing and ``"ntob"`` is
        used if upfacing.
    variable_name : str, default="Fileset1: Sv pings T1"
        Name of the Echoview acoustic variable to load from EV files.
    export_raw_csv : bool, default=True
        If ``True`` (default), exclusion and threshold settings in the EV file
        are temporarily disabled before exporting the CSV, in order to ensure
        all raw data is exported. If ``False``, thresholds and exclusions are
        used as per the EV file.

        .. versionadded:: 1.1.0

    row_len_selector : str, default="mode"
        Method used to handle input csv files with different number of Sv
        values across time (i.e. a non-rectangular input).
        See :meth:`echofilter.raw.loader.transect_loader` for options.
    facing : {"downward", "upward", "auto"}, default="auto"
        Orientation in which the echosounder is facing. Default is ``"auto"``,
        in which case the orientation is determined from the ordering of the
        depth values in the data (increasing = ``"upward"``,
        decreasing = ``"downward"``).
    use_training_standardization : bool, default=False
        Whether to use the exact normalization center and deviation values as
        used during training. If ``False`` (default), the center and deviation
        are determined per sample, using the same method methodology as used
        to determine the center and deviation values for training.
    prenorm_nan_value : float, optional
        If this is set, replace NaN values with a given Sv value before
        the data normalisation (Gaussian standardisation) step.
        By default, NaNs are left as they are until after standardising the
        data.

        .. versionadded:: 1.1.0

    postnorm_nan_value : float, optional
        Placeholder value to replace NaNs with. Does nothing if
        ``prenorm_nan_value`` is set. By default this is set to the
        value used to train the model.

        .. versionadded:: 1.1.0

    crop_min_depth : float, optional
        Minimum depth to include in input. By default, there is no
        minimum depth.
    crop_max_depth : float, optional
        Maxmimum depth to include in input. By default, there is no
        maximum depth.
    autocrop_threshold : float, default=0.35
        Minimum fraction of input height which must be found to be removable
        for the model to be re-run with an automatically cropped input.
    image_height : int, optional
        Height in pixels of input to model. The data loaded from the csv will
        be resized to this height (the width of the image is unchanged).
        By default, the height matches that used when the model was
        trained.
    checkpoint : str, optional
        A path to a checkpoint file, or name of a checkpoint known to this
        package (listed in ``echofilter/checkpoints.yaml``). By default,
        the first checkpoint in ``checkpoints.yaml`` is used.
    force_unconditioned : bool, default=False
        Whether to always use unconditioned logit outputs. If ``False``
        (default) conditional logits will be used if the checkpoint loaded is
        for a conditional model.
    logit_smoothing_sigma : float, optional
        Standard deviation over which logits will be smoothed before being
        converted into output. Disabled by default.

        .. versionchanged:: 1.1.0
            The default logit smoothing value changed from ``1`` (enabled) to
            ``0`` (disabled).

    device : str or torch.device, optional
        Name of device on which the model will be run. By default, the first
        available CUDA GPU is used if any are found, and otherwise the CPU is
        used. Set to ``"cpu"`` to use the CPU even if a CUDA GPU is available.
    hide_echoview : {"never", "new", "always"}, default="never"
        Whether to hide the Echoview window entirely while the code runs.
        If ``hide_echoview="new"``, the application is only hidden if it
        was created by this function, and not if it was already running.
        If ``hide_echoview="always"``, the application is hidden even if it was
        already running. In the latter case, the window will be revealed again
        when this function is completed.

        .. warning::
            Hiding Echoview has been found to cause problems when using
            Echoview 13 and above.
            For more details, see `#337 <https://github.com/DeepSenseCA/echofilter/pull/337>`__.

        .. deprecated:: 1.2.1
           Support for hiding echoview during inference will be dropped in a
           future release.

    minimize_echoview : bool, default=True
        If ``True`` (default), the Echoview window being used will be minimized
        while this function is running.
    verbose : int, default=2
        Verbosity level.
        Set to ``0`` to disable print statements, or elevate to a higher number
        to increase verbosity.
    """
    t_start_prog = time.time()

    if isinstance(paths, str):
        paths = [paths]

    progress_fmt = (
        echofilter.ui.style.dryrun_fmt if dry_run else echofilter.ui.style.progress_fmt
    )

    existing_file_msg = (
        "Run with overwrite_existing=True (with the command line"
        " interface, use the --force flag) to overwrite existing"
        " outputs."
    )

    if hide_echoview != "never":
        warnings.warn(
            "Hiding Echoview during inference is deprecated.", DeprecationWarning
        )
        if verbose >= 0:
            s = (
                "Warning: Hiding Echoview may result in unwanted side-effects."
                "\nFor details, see https://github.com/DeepSenseCA/echofilter/issues/337"
                "\nWarning: Hiding Echoview during inference is deprecated."
            )
            s = echofilter.ui.style.warning_fmt(s)
            print(s)

    if verbose >= 1:
        print(
            progress_fmt(
                "{}Starting inference {}.{} {}".format(
                    echofilter.ui.style.HighlightStyle.start,
                    "dry-run" if dry_run else "routine",
                    echofilter.ui.style.HighlightStyle.reset,
                    datetime.datetime.now().strftime("%A, %B %d, %Y at %H:%M:%S"),
                )
            )
        )

    if device is None:
        if verbose >= 2:
            print("Checking for available GPU device with CUDA capability...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if verbose >= 2:
            print(
                "Will run on {}{}{}".format(
                    echofilter.ui.style.HighlightStyle.start,
                    device,
                    echofilter.ui.style.HighlightStyle.reset,
                )
            )
    device = torch.device(device)

    if facing is None:
        facing = "auto"
    if facing.startswith("down"):
        facing = "downward"
    if facing.startswith("up"):
        facing = "upward"

    if suffix_file and suffix_file[0].isalpha():
        suffix_file = "-" + suffix_file

    if suffix_var and suffix_var[0].isalpha():
        suffix_var = "-" + suffix_var
    elif suffix_var is not None:
        pass
    elif suffix_file:
        suffix_var = suffix_file
    else:
        suffix_var = "_echofilter"

    if suffix_csv and suffix_csv[0].isalpha():
        if keep_ext:
            suffix_csv = "." + suffix_csv
        else:
            suffix_csv = "-" + suffix_csv

    line_colors = dict(
        turbulence=color_turbulence,
        turbulence_offset=color_turbulence_offset,
        bottom=color_bottom,
        bottom_offset=color_bottom_offset,
        surface=color_surface,
        surface_offset=color_surface_offset,
        nearfield=color_nearfield,
    )
    line_thicknesses = dict(
        turbulence=thickness_turbulence,
        turbulence_offset=thickness_turbulence_offset,
        bottom=thickness_bottom,
        bottom_offset=thickness_bottom_offset,
        surface=thickness_surface,
        surface_offset=thickness_surface_offset,
        nearfield=thickness_nearfield,
    )
    # Carry over default line colours and thicknesses
    for lname in ["turbulence", "bottom", "surface"]:
        key_source = lname
        key_dest = lname + "_offset"
        if line_colors[key_dest] is None:
            line_colors[key_dest] = line_colors[key_source]
        if line_thicknesses[key_dest] is None:
            line_thicknesses[key_dest] = line_thicknesses[key_source]

    if verbose >= 2:
        print("Fetching checkpoint...")

    # Load checkpoint
    checkpoint, ckpt_name = echofilter.ui.checkpoints.load_checkpoint(
        checkpoint,
        cache_dir=cache_dir,
        device=device,
        return_name=True,
        verbose=verbose,
    )

    if verbose >= 5:
        print("Loaded checkpoint {}:".format(ckpt_name))
        pprint.pprint(
            {
                k: checkpoint[k]
                for k in (
                    "sample_shape",
                    "epoch",
                    "best_loss",
                    "data_center",
                    "data_deviation",
                    "center_method",
                    "deviation_method",
                    "nan_value",
                    "training_routine",
                )
            }
        )

    if image_height is None:
        image_height = checkpoint.get("sample_shape", (128, 512))[1]

    if use_training_standardization:
        center_param = checkpoint.get("data_center", -80.0)
        deviation_param = checkpoint.get("data_deviation", 20.0)
    else:
        center_param = checkpoint.get("center_method", "mean")
        deviation_param = checkpoint.get("deviation_method", "stdev")
    if postnorm_nan_value is None:
        postnorm_nan_value = checkpoint.get("nan_value", -3)

    if verbose >= 4:
        print("Constructing U-Net model, with arguments:")
        pprint.pprint(checkpoint["model_parameters"])
    elif verbose >= 2:
        print("Constructing U-Net model...")
    unet = UNet(**checkpoint["model_parameters"])
    if hasattr(logit_smoothing_sigma, "__len__"):
        max_logit_smoothing_sigma = max(logit_smoothing_sigma)
    else:
        max_logit_smoothing_sigma = logit_smoothing_sigma
    if max_logit_smoothing_sigma > 0:
        ks = max(11, int(np.round(max_logit_smoothing_sigma * 6)))
        ks += (ks + 1) % 2  # Increment to odd number if even
        model = torch.nn.Sequential(
            unet,
            echofilter.nn.modules.GaussianSmoothing(
                channels=checkpoint["model_parameters"]["out_channels"],
                kernel_size=ks,
                sigma=logit_smoothing_sigma,
            ),
        )
    else:
        model = unet
    model = Echofilter(
        model,
        mapping=checkpoint.get("wrapper_mapping", None),
        **checkpoint.get("wrapper_params", {}),
    )
    is_conditional_model = model.params.get("conditional", False)
    if verbose >= 3:
        print(
            "Built {}model with {} trainable parameters".format(
                "conditional " if is_conditional_model else "",
                count_parameters(model, only_trainable=True),
            )
        )
    try:
        unet.load_state_dict(checkpoint["state_dict"])
        if verbose >= 3:
            print("Loaded U-Net state from the checkpoint")
    except RuntimeError as err:
        if verbose >= 5:
            s = (
                "Warning: Checkpoint doesn't seem to be for the UNet."
                "Trying to load it as the whole model instead."
            )
            s = echofilter.ui.style.warning_fmt(s)
            print(s)
        try:
            model.load_state_dict(checkpoint["state_dict"])
            if verbose >= 3:
                print("Loaded model state from the checkpoint")
        except RuntimeError:
            msg = (
                "Could not load the checkpoint state as either the whole model"
                "or the unet component."
            )
            with echofilter.ui.style.error_message(msg) as msg:
                print(msg)
                raise err

    # Ensure model is on correct device
    model.to(device)
    # Put model in evaluation mode
    model.eval()

    files_input = paths
    files = list(
        echofilter.path.parse_files_in_folders(
            paths, source_dir, extensions, recursive=recursive_dir_search
        )
    )
    if verbose >= 1:
        print(
            progress_fmt(
                "Processing {}{} file{}{}...".format(
                    echofilter.ui.style.HighlightStyle.start,
                    len(files),
                    "" if len(files) == 1 else "s",
                    echofilter.ui.style.HighlightStyle.reset,
                )
            )
        )

    if len(extensions) == 1 and "ev" in extensions:
        do_open = True
    else:
        do_open = False
        for file in files:
            if os.path.splitext(file)[1].lower() == ".ev":
                do_open = True
                break

    if dry_run:
        if verbose >= 3:
            print(
                "Echoview application would{} be opened {}.".format(
                    "" if do_open else " not",
                    (
                        "to convert EV files to CSV"
                        if do_open
                        else "(no EV files to process)"
                    ),
                )
            )
        do_open = False

    common_notes = textwrap.dedent(
        """
        Classified by echofilter {ver} at {dt}
        Model checkpoint: {ckpt_name}
        """.format(
            ver=echofilter.__version__,
            dt=datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            ckpt_name=os.path.split(ckpt_name)[1],
        )
    )

    disable_tqdm = len(files) == 1 or verbose <= 0
    skip_count = 0
    incompatible_count = 0
    error_msgs = []

    # Open Echoview connection
    with echofilter.win.maybe_open_echoview(
        do_open=do_open,
        minimize=minimize_echoview,
        hide=hide_echoview,
    ) as ev_app:
        for fname in tqdm(files, desc="Files", ascii=True, disable=disable_tqdm):
            if verbose >= 2:
                print(
                    "\n"
                    + progress_fmt(
                        "Processing {}".format(
                            echofilter.ui.style.highlight_fmt(fname),
                        )
                    )
                )

            # Check what the full path should be
            fname_full = echofilter.path.determine_file_path(fname, source_dir)

            # Determine where destination should be placed
            destination = echofilter.path.determine_destination(
                fname, fname_full, source_dir, output_dir
            )
            if not keep_ext:
                destination = os.path.splitext(destination)[0]

            # Make a list of all the outputs we will produce
            dest_files = {}
            for name in ("turbulence", "bottom", "surface"):
                dest_files[name] = "{}.{}{}.evl".format(destination, name, suffix_file)
            if not generate_turbulence_line:
                dest_files.pop("turbulence")
            if not generate_bottom_line:
                dest_files.pop("bottom")
            if not generate_surface_line:
                dest_files.pop("surface")
            dest_files["regions"] = "{}.{}{}.evr".format(
                destination, "regions", suffix_file
            )

            # Check if any of them exists and if there is any missing
            any_missing = False
            clobbers = []
            for _, dest_file in dest_files.items():
                if os.path.isfile(dest_file):
                    clobbers.append(dest_file)
                else:
                    any_missing = True

            # Check whether to skip processing this file
            if skip_existing and not any_missing:
                if verbose >= 2:
                    print(echofilter.ui.style.skip_fmt("  Skipping {}".format(fname)))
                skip_count += 1
                continue
            # Check whether we would clobber a file we can't overwrite
            if clobbers and not overwrite_existing:
                msg = "Output " + clobbers[0] + "\n"
                if len(clobbers) == 2:
                    msg += "and 1 other "
                elif len(clobbers) > 1:
                    msg += "  and {} others ".format(len(clobbers) - 1)
                msg += "already exist{} for file {}".format(
                    "s" if len(clobbers) == 1 else "",
                    fname,
                )
                with echofilter.ui.style.error_message(msg) as msg:
                    if dry_run:
                        error_msgs.append("Error: " + msg + "\n  " + existing_file_msg)
                        print(error_msgs[-1])
                        continue
                    raise EnvironmentError(msg + "\n  " + existing_file_msg)

            # Determine whether we need to run ev2csv on this file
            ext = os.path.splitext(fname)[1]
            if len(ext) > 0:
                ext = ext[1:].lower()
            if ext == "csv":
                process_as_ev = False
            elif ext == "ev":
                process_as_ev = True
            elif len(extensions) == 1 and "csv" in extensions:
                process_as_ev = False
            elif len(extensions) == 1 and "ev" in extensions:
                process_as_ev = True
            else:
                msg = "Unsure how to process file {} with unrecognised extension {}".format(
                    fname, ext
                )
                if not skip_incompatible:
                    with echofilter.ui.style.error_message(msg) as msg:
                        raise EnvironmentError(msg)
                if verbose >= 2:
                    print(
                        echofilter.ui.style.skip_fmt(
                            "  Skipping incompatible file {}".format(fname)
                        )
                    )
                incompatible_count += 1
                continue

            # If we are processing an ev file, we need to export it as a raw
            # csv file. Unless it has already been exported (which we will
            # check for below).
            export_to_csv = process_as_ev

            # Make a temporary directory in case we are not caching generated csvs
            # Directory and all its contents are deleted when we leave this context
            with tempfile.TemporaryDirectory() as tmpdirname:

                # Convert ev file to csv, if necessary
                ev2csv_dir = cache_csv
                if ev2csv_dir is None:
                    ev2csv_dir = tmpdirname

                if not export_to_csv:
                    csv_fname = fname_full
                else:
                    # Determine where exported CSV file should be placed
                    csv_fname = echofilter.path.determine_destination(
                        fname, fname_full, source_dir, ev2csv_dir
                    )
                    if not keep_ext:
                        csv_fname = os.path.splitext(csv_fname)[0]
                    csv_fname += suffix_csv + ".csv"

                if os.path.isfile(csv_fname):
                    # If CSV file is already cached, no need to re-export it
                    export_to_csv = False

                if not export_to_csv:
                    pass
                elif dry_run:
                    if verbose >= 1:
                        print(
                            "  Would export {} as CSV file {}".format(
                                fname_full, csv_fname
                            )
                        )
                else:
                    # Import ev2csv now. We delay this import so Linux users
                    # without pywin32 can run on CSV files.
                    from echofilter.ev2csv import ev2csv

                    # Export the CSV file
                    fname_full = os.path.abspath(fname_full)
                    csv_fname = os.path.abspath(csv_fname)
                    ev2csv(
                        fname_full,
                        csv_fname,
                        variable_name=variable_name,
                        export_raw=export_raw_csv,
                        ev_app=ev_app,
                        verbose=verbose - 1,
                    )

                if (dry_run and verbose >= 2) or verbose >= 3:
                    ww = "Would" if dry_run else "Will"
                    print("  {} write files:".format(ww))
                    for key, fname in dest_files.items():
                        if os.path.isfile(fname) and overwrite_existing:
                            over_txt = " " + echofilter.ui.style.overwrite_fmt(
                                "(overwriting existing file)"
                            )
                        else:
                            over_txt = ""
                        tp = "line" if os.path.splitext(fname)[1] == ".evl" else "file"
                        print(
                            "    {} export {} {} to: {}{}".format(
                                ww, key, tp, fname, over_txt
                            )
                        )
                if dry_run:
                    continue

                # Load the data
                try:
                    timestamps, depths, signals = echofilter.raw.loader.transect_loader(
                        csv_fname,
                        warn_row_overflow=0,
                        row_len_selector=row_len_selector,
                    )
                except KeyError:
                    if skip_incompatible and fname not in files_input:
                        if verbose >= 2:
                            print(
                                echofilter.ui.style.skip_fmt(
                                    "  Skipping incompatible file {}".format(fname)
                                )
                            )
                        incompatible_count += 1
                        continue
                    msg = "CSV file {} could not be loaded.".format(fname)
                    with echofilter.ui.style.error_message(msg) as msg:
                        print(msg)
                        raise

            try:
                output = inference_transect(
                    model,
                    timestamps,
                    depths,
                    signals,
                    device,
                    image_height,
                    facing=facing,
                    crop_min_depth=crop_min_depth,
                    crop_max_depth=crop_max_depth,
                    autocrop_threshold=autocrop_threshold,
                    force_unconditioned=force_unconditioned,
                    data_center=center_param,
                    data_deviation=deviation_param,
                    prenorm_nan_value=prenorm_nan_value,
                    postnorm_nan_value=postnorm_nan_value,
                    verbose=verbose - 1,
                )
            except BaseException as err:
                if not continue_on_error:
                    raise
                print(echofilter.ui.style.error_fmt("An error has occurred:"))
                print(echofilter.ui.style.error_fmt(str(err)))
                error_msgs.append(err)
                continue
            if verbose >= 5:
                s = "\n    ".join([""] + [str(k) for k in output.keys()])
                print("  Generated model output with fields:" + s)

            if is_conditional_model and not force_unconditioned:
                if output["is_upward_facing"]:
                    cs = "|upfacing"
                else:
                    cs = "|downfacing"
                if verbose >= 4:
                    print(
                        echofilter.ui.style.aside_fmt(
                            "  Using conditional probability outputs from model:"
                            " p(state{})".format(cs)
                        )
                    )
            else:
                cs = ""
                if is_conditional_model and verbose >= 4:
                    print(
                        echofilter.ui.style.aside_fmt(
                            "Using unconditioned output from conditional model"
                        )
                    )

            # Convert output into lines
            surface_depths = output["depths"][
                echofilter.utils.last_nonzero(
                    output["p_is_above_surface" + cs] > 0.5, -1
                )
            ]
            turbulence_depths = output["depths"][
                echofilter.utils.last_nonzero(
                    output["p_is_above_turbulence" + cs] > 0.5, -1
                )
            ]
            bottom_depths = output["depths"][
                echofilter.utils.first_nonzero(
                    output["p_is_below_bottom" + cs] > 0.5, -1
                )
            ]

            line_timestamps = output["timestamps"].copy()
            is_passive = output["p_is_passive" + cs] > 0.5
            if lines_during_passive == "predict":
                pass
            elif lines_during_passive == "redact":
                surface_depths = surface_depths[~is_passive]
                turbulence_depths = turbulence_depths[~is_passive]
                bottom_depths = bottom_depths[~is_passive]
                line_timestamps = line_timestamps[~is_passive]
            elif lines_during_passive == "undefined":
                surface_depths[is_passive] = EV_UNDEFINED_DEPTH
                turbulence_depths[is_passive] = EV_UNDEFINED_DEPTH
                bottom_depths[is_passive] = EV_UNDEFINED_DEPTH
            elif lines_during_passive.startswith("interp"):
                if lines_during_passive == "interpolate-time":
                    x = line_timestamps
                elif lines_during_passive == "interpolate-index":
                    x = np.arange(len(line_timestamps))
                else:
                    msg = "Unsupported passive line interpolation method: {}".format(
                        lines_during_passive
                    )
                    with echofilter.ui.style.error_message(msg) as msg:
                        raise ValueError(msg)

                if len(x[~is_passive]) == 0:
                    if verbose >= 0:
                        s = (
                            "Could not interpolate depths for passive data for"
                            " {}, as all data appears to be from passive"
                            " collection. The original model predictions will"
                            " be kept instead.".format(fname)
                        )
                        s = echofilter.ui.style.warning_fmt(s)
                        print(s)
                else:
                    surface_depths[is_passive] = np.interp(
                        x[is_passive], x[~is_passive], surface_depths[~is_passive]
                    )
                    turbulence_depths[is_passive] = np.interp(
                        x[is_passive], x[~is_passive], turbulence_depths[~is_passive]
                    )
                    bottom_depths[is_passive] = np.interp(
                        x[is_passive], x[~is_passive], bottom_depths[~is_passive]
                    )
            else:
                msg = "Unsupported passive line method: {}".format(lines_during_passive)
                with echofilter.ui.style.error_message(msg) as msg:
                    raise ValueError(msg)

            if output["is_upward_facing"]:
                nearfield_depth = np.max(depths) - nearfield
            else:
                nearfield_depth = np.min(depths) + nearfield

            # Export evl files
            destination_dir = os.path.dirname(destination)
            if destination_dir != "":
                os.makedirs(destination_dir, exist_ok=True)
            for line_name, line_depths in (
                ("turbulence", turbulence_depths),
                ("bottom", bottom_depths),
                ("surface", surface_depths),
            ):
                if line_name not in dest_files:
                    continue
                dest_file = dest_files[line_name]
                if verbose >= 3:
                    s = "  Writing output"
                    if not os.path.exists(dest_file):
                        pass
                    elif not overwrite_existing:
                        s = echofilter.ui.style.error_fmt(s)
                    else:
                        s = echofilter.ui.style.overwrite_fmt(s)
                    s += " {}".format(dest_file)
                    print(s)
                if os.path.exists(dest_file) and not overwrite_existing:
                    msg = "Output {} already exists.".format(dest_file)
                    with echofilter.ui.style.error_message(msg) as msg:
                        raise EnvironmentError(msg + "\n  " + existing_file_msg)

                echofilter.raw.loader.evl_writer(
                    dest_file,
                    line_timestamps,
                    line_depths,
                    pad=True,
                    status=line_status,
                )
            # Export evr file
            dest_file = dest_files["regions"]
            if verbose >= 3:
                s = "  Writing output"
                if not os.path.exists(dest_file):
                    pass
                elif not overwrite_existing:
                    s = echofilter.ui.style.error_fmt(s)
                else:
                    s = echofilter.ui.style.overwrite_fmt(s)
                s += " {}".format(dest_file)
                print(s)
            if os.path.exists(dest_file) and not overwrite_existing:
                msg = "Output {} already exists.".format(dest_file)
                with echofilter.ui.style.error_message(msg) as msg:
                    raise EnvironmentError(msg + "\n  " + existing_file_msg)

            patches_key = "p_is_patch"
            if patch_mode is None:
                if output["is_upward_facing"]:
                    patches_key += "-ntob"
            elif patch_mode != "merged":
                patches_key += "-" + patch_mode

            echofilter.raw.loader.write_transect_regions(
                dest_file,
                output,
                depth_range=depths,
                passive_key="p_is_passive" + cs,
                removed_key="p_is_removed" + cs,
                patches_key=patches_key + cs,
                collate_passive_length=collate_passive_length,
                collate_removed_length=collate_removed_length,
                minimum_passive_length=minimum_passive_length,
                minimum_removed_length=minimum_removed_length,
                minimum_patch_area=minimum_patch_area,
                name_suffix=suffix_var,
                common_notes=common_notes,
                verbose=verbose - 2,
                verbose_indent=2,
            )

            if not process_as_ev or not import_into_evfile:
                # Done with processing this file
                continue

            target_names = {key: key + suffix_var for key in dest_files}
            target_names["nearfield"] = "nearfield" + suffix_var

            offsets = dict(
                turbulence=offset_turbulence,
                bottom=offset_bottom,
                surface=offset_surface,
            )
            lines_cutoff_at_nearfield = []
            if output["is_upward_facing"]:
                if cutoff_at_nearfield or cutoff_at_nearfield is None:
                    lines_cutoff_at_nearfield = ["bottom"]
            elif cutoff_at_nearfield:
                lines_cutoff_at_nearfield = ["turbulence"]

            import_lines_regions_to_ev(
                fname_full,
                dest_files,
                target_names=target_names,
                nearfield_depth=nearfield_depth,
                add_nearfield_line=add_nearfield_line,
                lines_cutoff_at_nearfield=lines_cutoff_at_nearfield,
                offsets=offsets,
                line_colors=line_colors,
                line_thicknesses=line_thicknesses,
                ev_app=ev_app,
                overwrite=overwrite_ev_lines,
                common_notes=common_notes,
                verbose=verbose,
            )

    if verbose >= 1:
        print(
            progress_fmt(
                "{}Finished {}processing {} file{}{}.".format(
                    echofilter.ui.style.HighlightStyle.start,
                    "simulating " if dry_run else "",
                    len(files),
                    "" if len(files) == 1 else "s",
                    echofilter.ui.style.HighlightStyle.reset,
                )
            )
        )
        skip_total = skip_count + incompatible_count
        if skip_total > 0:
            s = ""
            s += "Of these, {}{}{} file{} skipped{}: {} already processed".format(
                "all " if skip_total == len(files) else "",
                echofilter.ui.style.HighlightStyle.start,
                skip_total,
                " was" if skip_total == 1 else "s were",
                echofilter.ui.style.HighlightStyle.reset,
                skip_count,
            )
            if not dry_run:
                s += ", {} incompatible".format(incompatible_count)
            s += "."
            s = echofilter.ui.style.skip_fmt(s)
            print(s)
        if error_msgs:
            print(
                echofilter.ui.style.error_fmt(
                    "There {} {} error{}:".format(
                        "was" if len(error_msgs) == 1 else "were",
                        len(error_msgs),
                        "" if len(error_msgs) == 1 else "s",
                    )
                )
            )
            for error_msg in error_msgs:
                print(echofilter.ui.style.error_fmt(str(error_msg)))
        print(
            "Total runtime: {}".format(
                datetime.timedelta(seconds=time.time() - t_start_prog)
            )
        )


def inference_transect(
    model,
    timestamps,
    depths,
    signals,
    device,
    image_height,
    facing="auto",
    crop_min_depth=None,
    crop_max_depth=None,
    autocrop_threshold=0.35,
    force_unconditioned=False,
    data_center="mean",
    data_deviation="stdev",
    prenorm_nan_value=None,
    postnorm_nan_value=-3,
    dtype=torch.float,
    verbose=0,
):
    """
    Run inference on a single transect.

    Parameters
    ----------
    model : echofilter.wrapper.Echofilter
        A pytorch Module wrapped in an Echofilter UI layer.
    timestamps : array_like
        Sample recording timestamps (in seconds since Unix epoch). Must be a
        vector.
    depths : array_like
        Recording depths from the surface (in metres). Must be a vector.
    signals : array_like
        Echogram Sv data. Must be a matrix shaped
        `(len(timestamps), len(depths))`.
    image_height : int
        Height to resize echogram before passing through model.
    facing : {"downward", "upward", "auto"}, default="auto"
        Orientation in which the echosounder is facing. Default is ``"auto"``,
        in which case the orientation is determined from the ordering of the
        depth values in the data (increasing = ``"upward"``,
        decreasing = ``"downward"``).
    crop_min_depth : float, optional
        Minimum depth to include in input. By default, there is no
        minimum depth.
    crop_max_depth : float, optional
        Maxmimum depth to include in input. By default, there is no
        maximum depth.
    autocrop_threshold : float, default=0.35
        Minimum fraction of input height which must be found to be removable
        for the model to be re-run with an automatically cropped input.
    force_unconditioned : bool, optional
        Whether to always use unconditioned logit outputs when deteriming the
        new depth range for automatic cropping.
    data_center : float or str, default="mean"
        Center point to use, which will be subtracted from the Sv signals
        (i.e. the overall sample mean).
        If ``data_center`` is a string, it specifies the method to use to
        determine the center value from the distribution of intensities seen
        in this sample transect.
    data_deviation : float or str, default="stdev"
        Deviation to use to normalise the Sv signals in divisive manner
        (i.e. the overall sample standard deviation).
        If ``data_deviation`` is a string, it specifies the method to use to
        determine the center value from the distribution of intensities seen
        in this sample transect.
    prenorm_nan_value : float, optional
        If this is set, replace NaN values with a given Sv value before
        the data normalisation (Gaussian standardisation) step.
        By default, NaNs are left as they are until after standardising the
        data.
    postnorm_nan_value : float, default=-3
        Placeholder value to replace NaNs with. Does nothing if
        ``prenorm_nan_value`` is set.
    dtype : torch.dtype, default=torch.float
        Datatype to use for model input.
    verbose : int, default=0
        Level of verbosity.

    Returns
    -------
    dict
        Dictionary with fields as output by :class:`echofilter.wrapper.Echofilter`,
        plus ``timestamps`` and ``depths``.
    """
    facing = facing.lower()
    timestamps = np.asarray(timestamps)
    depths = np.asarray(depths)
    signals = np.asarray(signals)
    transect = {
        "timestamps": timestamps,
        "depths": depths,
        "signals": signals,
    }
    if crop_min_depth is not None:
        # Apply minimum depth crop
        depth_crop_mask = transect["depths"] >= crop_min_depth
        transect["depths"] = transect["depths"][depth_crop_mask]
        transect["signals"] = transect["signals"][:, depth_crop_mask]
    if crop_max_depth is not None:
        # Apply maximum depth crop
        depth_crop_mask = transect["depths"] <= crop_max_depth
        transect["depths"] = transect["depths"][depth_crop_mask]
        transect["signals"] = transect["signals"][:, depth_crop_mask]

    if prenorm_nan_value is not None:
        # Replace NaNs before transforming the data
        transect = echofilter.data.transforms.ReplaceNan(prenorm_nan_value)(transect)
    # Standardize data distribution
    transect = echofilter.data.transforms.Normalize(data_center, data_deviation)(
        transect
    )

    # Configure data to match what the model expects to see
    # Determine whether depths are ascending or descending
    is_upward_facing = transect["depths"][-1] < transect["depths"][0]
    # Ensure depth is always increasing (which corresponds to descending from
    # the air down the water column)
    if facing[:2] == "up" or (facing == "auto" and is_upward_facing):
        transect["depths"] = transect["depths"][::-1].copy()
        transect["signals"] = transect["signals"][:, ::-1].copy()
        if facing == "auto" and verbose >= 2:
            print(
                echofilter.ui.style.aside_fmt(
                    "  Echogram was autodetected as upward facing, and was flipped"
                    " vertically before being input into the model."
                )
            )
        if not is_upward_facing:
            s = (
                '  Warning: facing = "{}" was provided, but data appears to be'
                " downward facing".format(facing)
            )
            s = echofilter.ui.style.warning_fmt(s)
            print(s)
        is_upward_facing = True
    elif facing[:4] != "down" and facing != "auto":
        msg = 'facing should be one of "downward", "upward", and "auto"'
        with echofilter.ui.style.error_message(msg) as msg:
            raise ValueError(msg)
    elif facing[:4] == "down" and is_upward_facing:
        s = (
            '  Warning: facing = "{}" was provided, but data appears to be'
            " upward facing".format(facing)
        )
        s = echofilter.ui.style.warning_fmt(s)
        print(s)
        is_upward_facing = False
    elif facing == "auto" and verbose >= 2:
        print(
            echofilter.ui.style.aside_fmt(
                "  Echogram was autodetected as downward facing."
            )
        )

    # To reduce memory consumption, split into segments whenever the recording
    # interval is longer than normal
    segments = split_transect(max_length=1280, **transect)
    disable_tqdm = verbose < 1
    outputs = []
    for segment in tqdm(
        list(segments), desc="  Segments", position=0, ascii=True, disable=disable_tqdm
    ):
        # Try to remove any NaNs in the raw input with using 2d interpolation
        n_nans = np.isnan(segment["signals"]).sum()
        if n_nans > 0:
            if verbose >= 1:
                print("  Interpolating to fill in {} missing Sv values".format(n_nans))
            segment["signals"] = fillholes2d(segment["signals"])

        # Preprocessing transform
        transform = torchvision.transforms.Compose(
            [
                echofilter.data.transforms.ReplaceNan(postnorm_nan_value),
                echofilter.data.transforms.Rescale(
                    (segment["signals"].shape[0], image_height),
                    order=1,
                ),
            ]
        )
        segment = transform(segment)
        # Pad the edges of the segment with extra data. Any padding we add
        # now will be stripped away from the output by join_transect later.
        segment = pad_transect(segment)
        input = torch.tensor(segment["signals"]).unsqueeze(0).unsqueeze(0)
        input = input.to(device, dtype).contiguous()
        # Put data through model
        with torch.no_grad():
            output = model(input, output_device="cpu")
            output = {k: v.squeeze(0).cpu().numpy() for k, v in output.items()}
        output["timestamps"] = segment["timestamps"]
        output["depths"] = segment["depths"]
        output["_pad_start"] = segment["_pad_start"]
        output["_pad_end"] = segment["_pad_end"]
        outputs.append(output)

    output = join_transect(outputs)
    output["is_upward_facing"] = is_upward_facing

    if autocrop_threshold >= 1:
        # If we are not doing auto-crop, return the current output
        return output

    # See how much of the depth we could crop away
    cs = ""
    if model.params["conditional"] and not force_unconditioned:
        if output["is_upward_facing"]:
            cs = "|upfacing"
        else:
            cs = "|downfacing"
    is_passive = output["p_is_passive" + cs] > 0.5
    depth_intv = abs(transect["depths"][1] - transect["depths"][0])
    if is_upward_facing:
        # Restrict the top based on the observed surface depths
        surface_depths = output["depths"][
            echofilter.utils.last_nonzero(output["p_is_above_surface" + cs] > 0.5, -1)
        ]
        # Redact passive regions
        if len(surface_depths[~is_passive]) > 0:
            surface_depths = surface_depths[~is_passive]
        # Find a good estimator of the maximum. Go for a robust estimate of
        # four sigma from the middle.
        pct = np.percentile(surface_depths, [2.275, 97.725])
        dev = (pct[1] - pct[0]) / 4
        new_crop_min = pct[0] - 2 * dev
        # Don't go higher than the minimum of the predicted surface depths
        new_crop_min = max(new_crop_min, np.min(surface_depths))
        # Offset minimum by at least 2m, to be sure we are capturing enough
        # surrounding content.
        new_crop_min -= max(2, 10 * depth_intv)
        new_crop_max = np.max(transect["depths"])
    else:
        new_crop_min = np.min(transect["depths"])
        # Restrict the bottom based on the observed bottom depths
        bottom_depths = output["depths"][
            echofilter.utils.first_nonzero(output["p_is_below_bottom" + cs] > 0.5, -1)
        ]
        # Redact passive regions
        if len(bottom_depths[~is_passive]) > 0:
            bottom_depths = bottom_depths[~is_passive]
        # Find a good estimator of the maximum. Go for a robust estimate of
        # four sigma from the middle.
        pct = np.percentile(bottom_depths, [2.275, 97.725])
        dev = (pct[1] - pct[0]) / 4
        new_crop_max = pct[1] + 2 * dev
        # Don't go deeper than the maximum of the predicted bottom depths
        new_crop_max = min(new_crop_max, np.max(bottom_depths))
        # Offset maximum by at least 2m, to be sure we are capturing enough
        # surrounding content.
        new_crop_max += max(2, 10 * depth_intv)

    if crop_min_depth is not None:
        new_crop_min = max(new_crop_min, crop_min_depth)
    if crop_max_depth is not None:
        new_crop_max = min(new_crop_max, crop_max_depth)

    current_height = abs(transect["depths"][-1] - transect["depths"][0])
    new_height = abs(new_crop_max - new_crop_min)
    if (current_height - new_height) / current_height <= autocrop_threshold:
        # No need to crop
        return output

    if verbose >= 1:
        print(
            "  Automatically zooming in on the {:.2f}m to {:.2f}m depth range"
            " and re-doing model inference.".format(new_crop_min, new_crop_max)
        )

    # Crop and run again; and don't autocrop a second time!
    return inference_transect(
        model,
        timestamps,
        depths,
        signals,
        device,
        image_height,
        facing=facing,
        crop_min_depth=new_crop_min,
        crop_max_depth=new_crop_max,
        autocrop_threshold=1,  # Don't crop again
        force_unconditioned=force_unconditioned,
        data_center=data_center,
        data_deviation=data_deviation,
        prenorm_nan_value=prenorm_nan_value,
        postnorm_nan_value=postnorm_nan_value,
        dtype=dtype,
        verbose=verbose,
    )


def import_lines_regions_to_ev(
    ev_fname,
    files,
    target_names=None,
    nearfield_depth=None,
    add_nearfield_line=True,
    lines_cutoff_at_nearfield=None,
    offsets=None,
    line_colors=None,
    line_thicknesses=None,
    ev_app=None,
    overwrite=False,
    common_notes="",
    verbose=1,
):
    """
    Write lines and regions to EV file.

    Parameters
    ----------
    ev_fname : str
        Path to Echoview file to import variables into.
    files : dict
        Mapping from output keys to filenames.
    target_names : dict, optional
        Mapping from output keys to output variable names.
    nearfield_depth : float, optional
        Depth at which nearfield line will be placed. By default, no
        nearfield line will be added, irrespective of ``add_nearfield_line``.
    add_nearfield_line : bool, default=True
        Whether to add a nearfield line.
    lines_cutoff_at_nearfield : list of str, optional
        Which lines (if any) should be clipped at the nearfield depth.
        By default, no lines will be clipped.
    offsets : dict, optional
        Amount of offset for each line.
    line_colors : dict, optional
        Mapping from output keys to line colours.
    line_thicknesses : dict, optional
        Mapping from output keys to line thicknesses.
    ev_app : win32com.client.Dispatch object, optional
        An object which can be used to interface with the Echoview application,
        as returned by :class:`win32com.client.Dispatch`. By default, a
        new instance of the application is opened (and closed on completion).
    overwrite : bool, default=False
        Whether existing lines with target names should be replaced.
        If a line with the target name already exists and ``overwrite=False``,
        the line is named with the current datetime to prevent collisions.
    common_notes : str, default=""
        Notes to include for every region.
    verbose : int, default=1
        Verbosity level.
    """
    if target_names is None:
        target_names = {}
    if lines_cutoff_at_nearfield is None:
        lines_cutoff_at_nearfield = {}
    if offsets is None:
        offsets = {}
    if line_colors is None:
        line_colors = {}
    if line_thicknesses is None:
        line_thicknesses = {}

    if verbose >= 2:
        print(f"Importing {len(files)} lines/regions into EV file {ev_fname}")

    # Assemble the color palette
    colors = get_color_palette(sort_colors=False)

    dtstr = datetime.datetime.now().isoformat(timespec="seconds")

    with echofilter.win.open_ev_file(ev_fname, ev_app) as ev_file:

        def change_line_color_thickness(line_name, color, thickness, ev_app=ev_app):
            if color is not None or thickness is not None:
                ev_app.Exec(f"{line_name} | UseDefaultLineDisplaySettings =| false")
            if color is not None:
                if color in colors:
                    color = colors[color]
                elif not isinstance(color, str):
                    pass
                elif "xkcd:" + color in colors:
                    color = colors["xkcd:" + color]
                color = hexcolor2rgb8(color)
                color = repr(color).replace(" ", "")
                if verbose >= 4:
                    print(
                        f"    Setting color of EV line {line_name} to {color}"
                        " (color active when line has Good status)"
                    )
                ev_app.Exec(f"{line_name} | CustomGoodLineColor =| {color}")
            if thickness is not None:
                if verbose >= 4:
                    print(
                        f"    Setting thickness of EV line {line_name} to {thickness}"
                    )
                ev_app.Exec(f"{line_name} | CustomLineDisplayThickness =| {thickness}")

        for key, fname in files.items():
            # Check the file exists
            fname_full = os.path.abspath(fname)
            if not os.path.isfile(fname_full):
                s = f"  Warning: File '{fname_full}' could not be found"
                s = echofilter.ui.style.warning_fmt(s)
                print(s)
                continue

            if os.path.splitext(fname)[1].lower() != ".evl":
                if verbose >= 3:
                    print(f"  Adding {key} to {ev_fname} from {fname}")

                # Import regions from the EVR file
                is_imported = ev_file.Import(fname_full)
                if not is_imported:
                    s = (
                        f"  Warning: Unable to import file '{fname_full}'"
                        "\n  Please consult Echoview for the Import error message."
                    )
                    s = echofilter.ui.style.warning_fmt(s)
                    print(s)
                continue

            if verbose >= 3:
                print(f"  Adding {key} line to {ev_fname}")

            # Import the line into Python now. We might need it now for
            # clipping, or later on for offsetting.
            ts, depths, statuses = echofilter.raw.loader.evl_loader(
                fname_full, return_status=True
            )
            line_status = echofilter.utils.mode(statuses)

            if nearfield_depth is None or key not in lines_cutoff_at_nearfield:
                # Import the original line straight into Echoview
                fname_loaded = fname_full
                is_imported = ev_file.Import(fname_loaded)
            else:
                if verbose >= 3:
                    print(
                        f"  Clipping {key} line at nearfield depth {nearfield_depth}m"
                    )
                # Edit the line, clipping as necessary
                if key == "bottom":
                    depths_clipped = np.minimum(depths, nearfield_depth)
                else:
                    depths_clipped = np.maximum(depths, nearfield_depth)

                # Export the edited line to a temporary file
                with tempfile.TemporaryDirectory() as tmpdirname:
                    temp_fname = os.path.join(tmpdirname, os.path.split(fname)[1])
                    echofilter.raw.loader.evl_writer(
                        temp_fname, ts, depths_clipped, status=line_status
                    )
                    # Import the edited line into the EV file
                    fname_loaded = temp_fname
                    is_imported = ev_file.Import(fname_loaded)

            if not is_imported:
                s = (
                    f"  Warning: Unable to import file '{fname_loaded}'"
                    "\n  Please consult Echoview for the Import error message."
                )
                s = echofilter.ui.style.warning_fmt(s)
                print(s)
                continue

            # Identify line just imported, which is the last variable
            variable = ev_file.Variables[ev_file.Variables.Count - 1]
            lines = ev_file.Lines
            line = lines.FindByName(variable.Name)
            if not line:
                s = (
                    "  Warning: Could not find line which was just imported with"
                    f" name '{variable.Name}'"
                    "\n  Ignoring and continuing processing."
                )
                s = echofilter.ui.style.warning_fmt(s)
                print(s)
                continue

            # Check whether we need to change the name of the line
            target_name = target_names.get(key, None)

            # Check if a line with this name already exists
            old_line = None if target_name is None else lines.FindByName(target_name)

            # Maybe try to overwrite existing line with new line
            successful_overwrite = False
            if old_line and overwrite:
                # Overwrite the old line
                if verbose >= 2:
                    print(
                        echofilter.ui.style.overwrite_fmt(
                            f"Overwriting existing line '{target_name}' with new"
                            f" {key} line output"
                        )
                    )
                try:
                    old_line_edit = old_line.AsLineEditable()
                    if old_line_edit:
                        # Overwrite the old line with the new line
                        old_line_edit.OverwriteWith(line)
                        # Delete the line we imported
                        lines.Delete(line)
                        # Update our reference to the line
                        line = old_line
                        successful_overwrite = True
                except Exception as err:
                    if verbose >= 3:
                        # Line is not editable
                        s = (
                            f"An error occurred while trying to overwrite existing line '{target_name}':"
                            f"\n{old_line_edit}"
                            f"\n{err}"
                        )
                        s = echofilter.ui.style.warning_fmt(s)
                        print(s)
                if not successful_overwrite:
                    try:
                        # Just delete the old line (seems like a simpler solution)
                        lines.Delete(old_line)
                        successful_overwrite = True
                    except Exception as err:
                        if verbose >= 3:
                            # Line could not be deleted
                            s = (
                                f"An error occurred while trying to delete existing line '{target_name}':"
                                f"\n{old_line}"
                                f"\n{err}"
                            )
                            s = echofilter.ui.style.warning_fmt(s)
                            print(s)
                if not successful_overwrite and verbose >= 0:
                    if verbose >= 0:
                        # Line is not editable
                        s = f"Existing line '{target_name}' cannot be overwritten."
                        s = echofilter.ui.style.warning_fmt(s)
                        print(s)

            if old_line and not successful_overwrite:
                # Change the name so there is no collision
                target_name += "_{}".format(dtstr)
                if verbose >= 1:
                    print(
                        f"Target line name '{target_names[key]}' already exists."
                        f" Will save new {key} line with name '{target_name}' instead."
                    )

            if target_name and not successful_overwrite:
                # Rename the line
                variable.ShortName = target_name
                line.Name = target_name

            if verbose >= 2:
                print(f"  Added {key} line '{line.Name}'")

            # Change the color and thickness of the line
            change_line_color_thickness(
                line.Name, line_colors.get(key), line_thicknesses.get(key)
            )
            # Add notes to the line
            notes = key.title() + " line"
            if len(common_notes) > 0:
                notes += "\n" + common_notes
            ev_app.Exec(
                "{} | Notes =| {}".format(line.Name, notes.replace("\n", "\r\n"))
            )

            # Handle offset line ----------------------------------------------
            if key not in offsets:
                continue

            # Remember references to original line
            original_line = line
            original_target_name = target_name

            # Generate an offset line
            offset = offsets[key]

            if verbose >= 3:
                print(f"  Adding {offset}m offset {key} line to {ev_fname}")

            if key == "bottom":
                # Offset is upward for bottom line, downward otherwise
                offset = -offset
            depths_offset = depths + offset
            if nearfield_depth is None or key not in lines_cutoff_at_nearfield:
                pass
            elif key == "bottom":
                depths_offset = np.minimum(depths_offset, nearfield_depth)
            else:
                depths_offset = np.maximum(depths_offset, nearfield_depth)

            # Export the edited line to a temporary file
            with tempfile.TemporaryDirectory() as tmpdirname:
                fname_noext, ext = os.path.splitext(fname)
                temp_fname = os.path.join(
                    tmpdirname,
                    os.path.split(fname_noext)[1] + "_offset" + ext,
                )
                echofilter.raw.loader.evl_writer(
                    temp_fname,
                    ts,
                    depths_offset,
                    status=line_status,
                )
                # Import the edited line into the EV file
                is_imported = ev_file.Import(temp_fname)

            if not is_imported:
                s = (
                    f"  Warning: Unable to import file '{temp_fname}'"
                    "\n  Please consult Echoview for the Import error message."
                )
                s = echofilter.ui.style.warning_fmt(s)
                print(s)
                continue

            # Identify line just imported, which is the last variable
            variable = ev_file.Variables[ev_file.Variables.Count - 1]
            line = lines.FindByName(variable.Name)  # Reference to new, offset, line
            if not line:
                s = (
                    "  Warning: Could not find line which was just imported with"
                    f" name '{variable.Name}'"
                    "\n  Ignoring and continuing processing."
                )
                s = echofilter.ui.style.warning_fmt(s)
                print(s)
                continue

            # Work out what we should call our new line
            target_name = target_names.get(key + "_offset")
            if target_name:
                pass
            elif not original_target_name:
                pass
            elif key in original_line.Name:
                target_name = original_line.Name.replace(key, key + "-offset", 1)
            elif key in target_names.get(key):
                target_name = target_names[key].replace(key, key + "-offset", 1)
            else:
                target_name = original_line.Name + "-offset"

            # Check if a line with this name already exists
            old_line = None if target_name is None else lines.FindByName(target_name)

            # Maybe try to overwrite existing line with new line
            successful_overwrite = False
            if old_line and overwrite:
                # Overwrite the old line
                if verbose >= 2:
                    print(
                        echofilter.ui.style.overwrite_fmt(
                            f"Overwriting existing line '{target_name}' with new"
                            f" {key} line output"
                        )
                    )
                try:
                    old_line_edit = old_line.AsLineEditable()
                    if old_line_edit:
                        # Overwrite the old line with the new line
                        old_line_edit.OverwriteWith(line)
                        # Delete the line we imported
                        lines.Delete(line)
                        # Update our reference to the line
                        line = old_line
                        successful_overwrite = True
                except Exception as err:
                    if verbose >= 3:
                        # Line is not editable
                        s = (
                            f"An error occurred while trying to overwrite existing line '{target_name}':"
                            f"\n{old_line_edit}"
                            f"\n{err}"
                        )
                        s = echofilter.ui.style.warning_fmt(s)
                        print(s)
                if not successful_overwrite:
                    try:
                        # Just delete the old line (seems like a simpler solution)
                        lines.Delete(old_line)
                        successful_overwrite = True
                    except Exception as err:
                        if verbose >= 3:
                            # Line could not be deleted
                            s = (
                                f"An error occurred while trying to delete existing line '{target_name}':"
                                f"\n{old_line}"
                                f"\n{err}"
                            )
                            s = echofilter.ui.style.warning_fmt(s)
                            print(s)
                if not successful_overwrite and verbose >= 0:
                    if verbose >= 0:
                        # Line is not editable
                        s = f"Existing line '{target_name}' cannot be overwritten."
                        s = echofilter.ui.style.warning_fmt(s)
                        print(s)

            if old_line and not successful_overwrite:
                # Change the name so there is no collision
                target_name += "_{}".format(dtstr)
                if verbose >= 1:
                    print(
                        f"Target line name '{target_names[key]}' already exists."
                        f" Will save new {key} line with name '{target_name}' instead."
                    )

            if target_name and not successful_overwrite:
                # Rename the line
                variable.ShortName = target_name
                line.Name = target_name

            if verbose >= 2:
                print(f"  Added {offset}m offset {key} line '{line.Name}'")

            # Change the color and thickness of the line
            change_line_color_thickness(
                line.Name,
                line_colors.get(key + "_offset", line_colors.get(key)),
                line_thicknesses.get(key + "_offset", line_thicknesses.get(key)),
            )
            # Add notes to the line
            notes = "{} line\nOffset: {:+g}m".format(key.title(), offset)
            if len(common_notes) > 0:
                notes += "\n" + common_notes
            ev_app.Exec(
                "{} | Notes =| {}".format(line.Name, notes.replace("\n", "\r\n"))
            )

        # Add nearfield line
        if nearfield_depth is not None and add_nearfield_line:
            key = "nearfield"
            if verbose >= 3:
                print(f"  Adding nearfield line at fixed depth {nearfield_depth}")
            lines = ev_file.Lines
            line = lines.CreateFixedDepth(nearfield_depth)

            # Check whether we need to change the name of the line
            target_name = target_names.get(key, None)

            # Check if a line with this name already exists
            old_line = None if target_name is None else lines.FindByName(target_name)

            # Maybe try to delete existing line
            successful_overwrite = False
            if old_line and overwrite:
                # Overwrite the old line
                if verbose >= 2:
                    print(
                        echofilter.ui.style.overwrite_fmt(
                            f"Deleting existing line '{target_name}'"
                        )
                    )
                successful_overwrite = lines.Delete(old_line)
                if not successful_overwrite and verbose >= 0:
                    # Line could not be deleted
                    print(
                        echofilter.ui.style.warning_fmt(
                            f"Existing line '{target_name}' could not be deleted"
                        )
                    )

            if old_line and not successful_overwrite:
                # Change the output name so there is no collision
                target_name += "_{}".format(dtstr)
                if verbose >= 1:
                    print(
                        f"Target line name '{target_names[key]}' already exists."
                        f" Will save new {key} line with name '{target_name}' instead."
                    )

            if target_name:
                # Rename the line
                line.Name = target_name

            if verbose >= 2:
                print(
                    f"  Added nearfield line '{line.Name}' at fixed depth {nearfield_depth}m"
                )

            # Change the color and thickness of the line
            change_line_color_thickness(
                line.Name, line_colors.get(key), line_thicknesses.get(key)
            )
            # Add notes to the line
            notes = key.title() + " line"
            if len(common_notes) > 0:
                notes += "\n" + common_notes
            ev_app.Exec(
                "{} | Notes =| {}".format(line.Name, notes.replace("\n", "\r\n"))
            )

        # Overwrite the EV file now the outputs have been imported
        ev_file.Save()


def get_color_palette(include_xkcd=True, sort_colors=True):
    """
    Provide a mapping of named colors from matplotlib.

    Parameters
    ----------
    include_xkcd : bool, default=True
        Whether to include the XKCD color palette in the output.
        Note that XKCD colors have ``"xkcd:"`` prepended to their names to
        prevent collisions with official named colors from CSS4.
        See https://xkcd.com/color/rgb/ and
        https://blog.xkcd.com/2010/05/03/color-survey-results/
        for the XKCD colors.
    sort_colors : bool, default=True
        Whether to sort the colors by hue. Otherwise the colors are grouped
        together by source, and maintain their default ordering (alphabetized).

    Returns
    -------
    colors : dict
        Mapping from names of colors as strings to color value, either as
        an RGB tuple (fractional, 0 to 1 range) or a hexadecimal string.
    """
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    if include_xkcd:
        colors.update(**mcolors.XKCD_COLORS)

    if not sort_colors:
        return colors

    # Sort colors by hue, saturation, value and name
    by_hsv = sorted(
        (tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(color))), name)
        for name, color in colors.items()
    )
    names = [name for hsv, name in by_hsv]
    colors = {name: colors[name] for name in names}

    return colors


def hexcolor2rgb8(color):
    """
    Map hexadecimal colors to uint8 RGB.

    Parameters
    ----------
    color : str
        A hexadecimal color string, with leading ``"#"``.
        If the input is not a string beginning with ``"#"``, it is returned as-is
        without raising an error.

    Returns
    -------
    tuple
        RGB color tuple, in uint8 format (0--255).
    """
    if isinstance(color, str) and color[0] == "#":
        color = mcolors.to_rgba(color)[:3]
    if (
        isinstance(color, tuple)
        and any(isinstance(c, float) for c in color)
        or all(c <= 1 for c in color)
    ):
        color = tuple(max(0, min(255, int(np.round(c * 255)))) for c in color)
    return color


if __name__ == "__main__":
    main()
