"""
U-Net model.
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

import functools

import torch
from torch import nn

from . import modules


class Down(nn.Module):
    """
    Downscaling layer, downsampling by a factor of two in one or more dimensions.
    """

    def __init__(self, mode="max", compress_dims=True):
        super(Down, self).__init__()

        compress_dims = modules.utils._pair(compress_dims)
        kernel_sizes = modules.utils._pair(
            2 if compress_dim else 1 for compress_dim in compress_dims
        )

        if mode == "max":
            self.pool = nn.MaxPool2d(kernel_sizes)
        elif mode == "avg":
            self.pool = nn.AvgPool2d(kernel_sizes)
        else:
            raise ValueError("Unsupported pooling method: {}".format(mode))

    def forward(self, x):
        return self.pool(x)


class Up(nn.Module):
    """
    Upscaling layer, upsampling by a factor of two in one or more dimensions.
    """

    def __init__(self, in_channels=None, up_dims=True, mode="bilinear"):
        super(Up, self).__init__()

        up_dims = modules.utils._pair(up_dims)
        kernel_sizes = modules.utils._pair(2 if up_dim else 1 for up_dim in up_dims)

        # If conv mode, use a transposed convolution to increase the size
        # Otherwise, use one of the nn.Upsample modes:
        # {"nearest", "linear", "bilinear", "bicubic"}
        if "conv" in mode:
            if in_channels is None:
                raise ValueError(
                    "Number of channels must be provided if upscaling with "
                    "transposed convolution."
                )
            self.up = nn.ConvTranspose2d(
                in_channels,
                in_channels,
                kernel_size=kernel_sizes,
                stride=kernel_sizes,
            )
        else:
            self.up = nn.Upsample(
                scale_factor=kernel_sizes, mode=mode, align_corners=True
            )

    def forward(self, x):
        return self.up(x)


class UNetBlock(nn.Module):
    """
    Create a (cascading set of) UNet block(s).

    Each block performs the steps:
        - Store input to be used in skip connection
        - Down step
        - Horizontal block
        - <Recursion>
        - Up step
        - Concatenate with skip connection
        - Horizontal block

    Where <Recursion> is a call generating a child UNetBlock instance.

    Parameters
    ----------
    in_channels : int
        Number of input channels to this block.
    horizontal_block_factory : callable
        A :class:`torch.nn.Module` constructor or function which returns a
        block of layers. The resulting module must accept ``in_channels`` and
        ``out_channels`` as its first two arguments.
    n_block : int, optional
        The number of nested UNetBlocks to use. Default is ``1`` (no nesting).
    block_expansion_factor : int or float, optional
        Expansion factor for the number of channels between nested UNetBlocks.
        Default is ``2``.
    expand_only_on_down : bool, optional
        Whether to exand the number of channels only when one of the spatial
        dimensions is compressed. Default is ``False``.
    blocks_per_downsample : int or sequence, optional
        How many blocks to include between each downsample operation. This can
        be a tuple of values for each spatial dimension, or an int which
        uses the same value for each spatial dimension. Default is ``1``.
    blocks_before_first_downsample : int or sequence, optional
        How many blocks to include before the first spatial downsampling
        occurs. Default is ``1``.
    always_include_skip_connection : bool, optional
        If ``True``, a skip connection is included even if no dimensions were
        downsampled in this block. Default is ``True``.
    deepest_inner : {callable, "horizontal_block", "identity", None}, optional
        A layer which should be applied at the deepest part of the network,
        before the first upsampling step. The parameter should either be a
        pre-instantiated layer, or the string ``"horizontal_block"``, to indicate
        an additional block as generated by the ``horizontal_block_factory``.
        If it is the string ``"identity"`` or ``None`` (default), no additional
        layer is included at the deepest point before upsampling begins.
    downsampling_modes : {"max", "avg", "stride"} or sequence, optional
        The downsampling mode to use. If this is a string, the same
        downsampling mode is used for every downsampling step. If it is
        a sequence, it should contain a string for each downsampling step.
        If the input sequence is too short, the final value will be used
        for all remaining downsampling steps. Default is ``"max"``.
    upsampling_modes : str or sequence, optional
        The upsampling mode to use. If this is a string, it must be ``"conv"``,
        or something supported by :class:`torch.nn.Upsample`; the same
        upsampling mode is used for every upsampling step. If it is
        a sequence, it should contain a string for each upsampling step.
        If the input sequence is too short, the final value will be used
        for all remaining upsampling steps. Default is ``"bilinear"``.
    _i_block : int, optional
        The current block number. Used internally to track recursion.
        Default is ``0``.
    _i_down : int, optional
        Used internally to track downsampling depth. Default is ``0``.

    Notes
    -----
    This class is defined recursively, and will instantiate itself as its own
    child until the number of blocks has been satisfied.
    """

    def __init__(
        self,
        in_channels,
        horizontal_block_factory,
        n_block=1,
        block_expansion_factor=2,
        expand_only_on_down=False,
        blocks_per_downsample=1,
        blocks_before_first_downsample=0,
        always_include_skip_connection=True,
        deepest_inner="identity",
        downsampling_modes="max",
        upsampling_modes="bilinear",
        _i_block=0,
        _i_down=0,
    ):
        super(UNetBlock, self).__init__()

        # Ensure these variables are a tuple of length two
        blocks_per_downsample = modules.utils._pair(blocks_per_downsample)
        blocks_before_first_downsample = modules.utils._pair(
            blocks_before_first_downsample
        )

        # Check which downsampling and upsampling mode we are using for this
        # layer (may be the same for every layer)
        if isinstance(downsampling_modes, str):
            downsampling_mode = downsampling_modes
        elif _i_down >= len(downsampling_modes):
            downsampling_mode = downsampling_modes[-1]
        else:
            downsampling_mode = downsampling_modes[_i_down]

        if isinstance(upsampling_modes, str):
            upsampling_mode = upsampling_modes
        elif _i_down >= len(upsampling_modes):
            upsampling_mode = upsampling_modes[-1]
        else:
            upsampling_mode = upsampling_modes[_i_down]

        # Check which dimensions need to be compressed with this block
        compress_dims = tuple(
            _i_block >= i0 and (_i_block - i0) % k == 0
            for i0, k in zip(blocks_before_first_downsample, blocks_per_downsample)
        )
        compress_any_dims = any(compress_dims)

        # Determine whether we are increasing the number of channels, and
        # if so what to
        if expand_only_on_down and not compress_any_dims:
            out_channels = in_channels
        else:
            out_channels = int(max(1, round(in_channels * block_expansion_factor)))

        # Downsamling step. If the mode is "stride", this is incorporated
        # into the horizontal block with a strided convolution. If there
        # is no need to downsample, it will be the Identity function.
        stride = 1
        if not compress_any_dims:
            self.down = nn.Identity()
        elif downsampling_mode == "stride":
            self.down = nn.Identity()
            stride = tuple(2 if compress_dim else 1 for compress_dim in compress_dims)
        else:
            self.down = Down(mode=downsampling_mode, compress_dims=compress_dims)

        # First horizontal block. It might begin with a downsampling stride,
        # and might increase the number of channels.
        self.horizontal_block_a = horizontal_block_factory(
            in_channels,
            out_channels,
            stride=stride,
        )

        # In the sequence, the inner step comes next. But we will define it
        # once we have finished defining everything else in this UNet block.

        # Upsampling step. Does the inverse of the Down step, using some
        # method.
        if not compress_any_dims:
            self.up = nn.Identity()
        else:
            self.up = Up(
                in_channels=out_channels, up_dims=compress_dims, mode=upsampling_mode
            )

        if compress_any_dims or always_include_skip_connection:
            # Concatenation step
            self.concatenate = modules.FlexibleConcat2d()
            b_in_channels = in_channels + out_channels
        else:
            # No concatenation step
            self.concatenate = None
            b_in_channels = out_channels

        # Second horizontal block. Takes both the skip connection and the
        # upsampled data as its input.
        self.horizontal_block_b = horizontal_block_factory(b_in_channels, in_channels)

        if _i_block + 1 < n_block:
            # Recurse deeper! Call this class again, but with the
            # block counter increased.
            self.nested = UNetBlock(
                out_channels,
                horizontal_block_factory,
                n_block=n_block,
                block_expansion_factor=block_expansion_factor,
                expand_only_on_down=expand_only_on_down,
                blocks_per_downsample=blocks_per_downsample,
                blocks_before_first_downsample=blocks_before_first_downsample,
                always_include_skip_connection=always_include_skip_connection,
                deepest_inner=deepest_inner,
                downsampling_modes=downsampling_modes,
                upsampling_modes=upsampling_modes,
                _i_block=_i_block + 1,
                _i_down=_i_down + compress_any_dims,
            )
        elif callable(deepest_inner):
            self.nested = deepest_inner
        elif deepest_inner is None or deepest_inner.lower() == "identity":
            # End recursion, by doing nothing for the inner loop.
            self.nested = nn.Identity()
        elif deepest_inner == "horizontal_block":
            # End recursion, by doing an extra regular block.
            self.nested = horizontal_block_factory(out_channels, out_channels)
        else:
            raise ValueError(
                "Unsupported deepest_inner value: {}".format(deepest_inner)
            )

    def forward(self, input):
        x = self.down(input)
        x = self.horizontal_block_a(x)
        x = self.nested(x)
        x = self.up(x)
        if self.concatenate is not None:
            x = self.concatenate(x, input)
        x = self.horizontal_block_b(x)
        return x


class UNet(nn.Module):
    """
    UNet model.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    initial_channels : int, optional
        Number of latent channels to output from the initial convolution
        facing the input layer. Default is ``32``.
    bottleneck_channels : int, optional
        Number of channels to output from the first block, before the first
        unet downsampling step can occur. Default is the same as
        ``initial_channels``.
    n_block : int, optional
        Number of blocks, both up and down. Default is ``4``.
    unet_expansion_factor : int or float, optional
        Channel expansion factor between unet blocks. Default is ``2``.
    expand_only_on_down : bool, optional
        Whether to only apply ``unet_expansion_factor`` on unet blocks which
        actually containg a down/up sampling component, and not on vanilla
        blocks. Default is ``False``.
    blocks_per_downsample : int or sequence, optional
        Block interval between dowsampling steps in the unet. If this is
        a sequence, it corresponds to the number of blocks for each spatial
        dimension. Default is ``1``.
    blocks_before_first_downsample : int, optional
        Number of blocks to use before and after the main unet structure.
        Must be at least ``1``. Default is ``1``.
    always_include_skip_connection : bool, optional
        If ``True``, a skip connection is included between all blocks equally
        far from the start and end of the UNet. If ``False``, skip connections
        are only used between downsampling and upsampling operations. Default
        is ``True``.
    deepest_inner : {callable, "horizontal_block", "identity", None}, optional
        A layer which should be applied at the deepest part of the network,
        before the first upsampling step. The parameter should either be a
        pre-instantiated layer, or the string ``"horizontal_block"``, to indicate
        an additional block as generated by the ``horizontal_block_factory``.
        If it is the string ``"identity"`` or ``None`` (default), no additional
        layer is included at the deepest point before upsampling begins.
    intrablock_expansion : int or float, optional
        Channel expansion factor within inverse residual block. Default is ``6``.
    se_reduction : int or float, optional
        Channel reduction factor within squeeze and excite block.
        Default is ``4``.
    downsampling_modes : {"max", "avg", "stride"} or sequence, optional
        The downsampling mode to use. If this is a string, the same
        downsampling mode is used for every downsampling step. If it is
        a sequence, it should contain a string for each downsampling step.
        If the input sequence is too short, the final value will be used
        for all remaining downsampling steps. Default is ``"max"``.
    upsampling_modes : str or sequence, optional
        The upsampling mode to use. If this is a string, it must be ``"conv"``,
        or something supported by :class:`torch.nn.Upsample`; the same
        upsampling mode is used for every upsampling step. If it is
        a sequence, it should contain a string for each upsampling step.
        If the input sequence is too short, the final value will be used
        for all remaining upsampling steps. Default is ``"bilinear"``.
    depthwise_separable_conv : bool, optional
        Whether to use depthwise separable convolutions in the MBConv block.
        Otherwise, the depth and pointwise convolutions are fused together
        into a regular convolution. Default is ``True``.
    residual : bool, optional
        Whether to use a residual architecture for the MBConv blocks.
        Default is ``True``.
    actfn : str, optional
        Name of the activation function to use. Default is ``"InplaceReLU"``.
    kernel_size : int, optional
        Size of convolution kernel to use. Default is ``5``.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        initial_channels=32,
        bottleneck_channels=None,
        n_block=4,
        unet_expansion_factor=2,
        expand_only_on_down=False,
        blocks_per_downsample=1,
        blocks_before_first_downsample=1,
        always_include_skip_connection=True,
        deepest_inner="identity",
        intrablock_expansion=6,
        se_reduction=4,
        downsampling_modes="max",
        upsampling_modes="bilinear",
        depthwise_separable_conv=True,
        residual=True,
        actfn="InplaceReLU",
        kernel_size=5,
    ):
        super(UNet, self).__init__()

        if bottleneck_channels is None:
            bottleneck_channels = initial_channels

        blocks_before_first_downsample = modules.utils._pair(
            blocks_before_first_downsample
        )

        if any(b < 1 for b in blocks_before_first_downsample):
            raise ValueError(
                "An initial block is hard coded. Number of blocks before first"
                " downsample must be at least 1."
            )

        self.in_channels = in_channels
        self.out_channels = out_channels

        actfn_factory = modules.activations.str2actfnfactory(actfn)

        horizontal_block_factory = functools.partial(
            modules.MBConv,
            expansion=intrablock_expansion,
            se_reduction=se_reduction,
            fused=not depthwise_separable_conv,
            residual=residual,
            actfn=actfn_factory,
            kernel_size=kernel_size,
        )

        self.initial_conv = nn.Sequential(
            modules.Conv2dSame(in_channels, initial_channels, kernel_size=kernel_size),
            nn.BatchNorm2d(initial_channels),
            actfn_factory(),
        )
        self.first_block = horizontal_block_factory(
            initial_channels,
            bottleneck_channels,
            expansion=1,
        )
        self.main_blocks = UNetBlock(
            bottleneck_channels,
            horizontal_block_factory,
            n_block=n_block,
            block_expansion_factor=unet_expansion_factor,
            expand_only_on_down=expand_only_on_down,
            blocks_per_downsample=blocks_per_downsample,
            blocks_before_first_downsample=tuple(
                b - 1 for b in blocks_before_first_downsample
            ),
            always_include_skip_connection=always_include_skip_connection,
            deepest_inner=deepest_inner,
            downsampling_modes=downsampling_modes,
            upsampling_modes=upsampling_modes,
        )
        self.final_block = horizontal_block_factory(bottleneck_channels, out_channels)

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.first_block(x)
        x = self.main_blocks(x)
        x = self.final_block(x)
        return x
