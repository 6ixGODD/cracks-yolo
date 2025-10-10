from __future__ import annotations

import typing as t

import torch
import torch.nn as nn
import torch.nn.common_types as ct
import torch.nn.functional as F
from ultralytics.nn.modules import Bottleneck
from ultralytics.nn.modules import C3


def autopad(k: int | t.List[int],
            p: int | t.List[int] | None = None,
            d: int = 1) -> int | t.List[int]:
    """Automatically calculates padding to maintain 'same' output shape for
    convolutions. Adjusts for optional dilation to ensure output size matches
    input size.

    Args:
        k: Kernel size (int or list). Size of the convolution kernel.
        p: Padding (int or list, optional). If None, auto-calculates padding.
        d: Dilation rate (int, default=1). Spacing between kernel elements.

    Returns:
        int or list: Calculated padding value(s) to maintain spatial dimensions.

    Example:
        For k=3, d=1: returns 1 (pad by 1 on each side)
        For k=3, d=2: kernel becomes effective size 5, returns 2
    """
    # Adjust kernel size based on dilation: effective_kernel = d * (k - 1) + 1
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]

    # Calculate padding as half the kernel size (standard 'same' padding)
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class ConvAWS2d(nn.Conv2d):
    """Adaptive Weight Standardization (AWS) Convolution.

    Normalizes convolution weights to have zero mean and unit variance per
    output channel. This improves training stability and generalization,
    especially in deeper networks.

    Reference:
        "Weight Standardization" - Normalizes weights instead of activations.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: ct._size_2_t,
                 stride: ct._size_2_t = 1,
                 padding: str | ct._size_2_t = 0,
                 dilation: ct._size_2_t = 1,
                 groups: int = 1,
                 bias: bool = True) -> None:
        """Initializes AWS convolution with learnable scale (gamma) and shift
        (beta) parameters.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Size of the convolution kernel.
            stride: Stride of the convolution. Defaults to 1.
            padding: Padding added to input. Defaults to 0.
            dilation: Spacing between kernel elements. Defaults to 1.
            groups: Number of blocked connections from input to output.
                Defaults to 1.
            bias: If True, adds a learnable bias. Defaults to True.
        """
        super().__init__(in_channels,
                         out_channels,
                         kernel_size,
                         stride=stride,
                         padding=padding,
                         dilation=dilation,
                         groups=groups,
                         bias=bias)
        # Learnable scale parameter (gamma) for each output channel, initialized to 1
        self.register_buffer("weight_gamma", torch.ones(self.out_channels, 1, 1, 1))
        # Learnable shift parameter (beta) for each output channel, initialized to 0
        self.register_buffer("weight_beta", torch.zeros(self.out_channels, 1, 1, 1))

    def _get_weight(self, weight: torch.Tensor) -> torch.Tensor:
        """Standardizes convolution weights to zero mean and unit variance.
        Then applies learned affine transformation: gamma * normalized_weight +
        beta.

        Args:
            weight: Original convolution weights of shape (C_out, C_in, K, K).

        Returns:
            Standardized and scaled weights.
        """
        # Compute mean across spatial and input channel dimensions for each output channel
        weight_mean = (weight.mean(dim=1, keepdim=True).mean(dim=2,
                                                             keepdim=True).mean(dim=3,
                                                                                keepdim=True))
        # Center the weights (zero mean)
        weight -= weight_mean

        # Compute standard deviation for each output channel (flatten spatial + channel dims)
        std = torch.sqrt(weight.view(weight.size(0), -1).var(dim=1) + 1e-5).view(-1, 1, 1, 1)
        # Normalize to unit variance
        weight /= std

        # Apply learnable affine transformation: scale and shift
        weight = self.weight_gamma * weight + self.weight_beta
        return weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with weight standardization applied on-the-fly.

        Args:
            x: Input tensor of shape (B, C_in, H, W).

        Returns:
            Convolution output with standardized weights.
        """
        # Get standardized weights
        weight = self._get_weight(self.weight)
        # Perform convolution with standardized weights (bias=None handled separately)
        return super()._conv_forward(x, weight, None)

    def _load_from_state_dict(self, state_dict: t.Dict[str, torch.Tensor], prefix: str,
                              local_metadata: t.Dict[str, t.Any], strict: bool,
                              missing_keys: t.List[str], unexpected_keys: t.List[str],
                              error_msgs: t.List[str]) -> None:
        """Custom state dict loading to initialize gamma/beta from existing
        weights if needed. This ensures compatibility when loading weights
        trained without AWS.

        Process:
            1. Try to load from state dict
            2. If gamma/beta not in state dict, compute them from current
            weights
            3. gamma = std of weights, beta = mean of weights

        Args:
            state_dict: Dictionary containing model parameters.
            prefix: Prefix for parameter names.
            local_metadata: Metadata for loading.
            strict: Whether to strictly enforce key matching.
            missing_keys: List to store missing keys.
            unexpected_keys: List to store unexpected keys.
            error_msgs: List to store error messages.
        """
        # Sentinel value to detect if gamma was loaded from state_dict
        self.weight_gamma.data.fill_(-1)

        # Load state dict normally
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys,
                                      unexpected_keys, error_msgs)

        # If gamma was successfully loaded (mean > 0), we're done
        if self.weight_gamma.data.mean() > 0:
            return

        # Otherwise, initialize gamma and beta from current weights
        weight = self.weight.data

        # Compute mean for beta initialization
        weight_mean = weight.data.mean(dim=1, keepdim=True).mean(dim=2,
                                                                 keepdim=True).mean(dim=3,
                                                                                    keepdim=True)
        self.weight_beta.data.copy_(weight_mean)

        # Compute std for gamma initialization
        std = torch.sqrt(weight.view(weight.size(0), -1).var(dim=1) + 1e-5).view(-1, 1, 1, 1)
        self.weight_gamma.data.copy_(std)


class SAConv2d(ConvAWS2d):
    """Switchable Atrous Convolution (SAC).

    Dynamically switches between two convolution branches with different
    receptive fields:
    - Small receptive field: Standard convolution (dilation=d)
    - Large receptive field: Dilated convolution (dilation=3*d)

    The switch is learned based on input content, allowing the network to
    adaptively select the appropriate receptive field for different spatial
    contexts.

    Also includes:
    - Pre-context: Global context before convolution
    - Post-context: Global context after convolution
    - Weight standardization (inherited from ConvAWS2d)

    Reference:
        "Detectors: Detecting Objects with Recursive Feature Pyramid and
        Switchable Atrous Convolution"
    """

    default_act = nn.SiLU()  # SiLU (Swish) activation

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 s: int = 1,
                 p: int | None = None,
                 g: int = 1,
                 d: int = 1,
                 act: bool | nn.Module = True,
                 bias: bool = True) -> None:
        """Initializes Switchable Atrous Convolution with context modules and
        switching mechanism.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Kernel size. Defaults to 3.
            s: Stride. Defaults to 1.
            p: Padding. If None, auto-calculated. Defaults to None.
            g: Groups for grouped convolution. Defaults to 1.
            d: Base dilation rate. Defaults to 1.
            act: Activation function. True uses SiLU, False/None uses Identity,
                or provide custom nn.Module. Defaults to True.
            bias: Whether to use bias in convolution. Defaults to True.
        """
        super().__init__(in_channels,
                         out_channels,
                         kernel_size,
                         stride=s,
                         padding=autopad(kernel_size, p, d),
                         dilation=d,
                         groups=g,
                         bias=bias)

        # ============ Switching Mechanism ============
        # 1x1 conv that outputs a scalar per spatial location (acts as attention weight)
        # Initialized to output 1.0 everywhere (favoring small receptive field initially)
        self.switch = nn.Conv2d(self.in_channels, 1, kernel_size=1, stride=s, bias=True)
        self.switch.weight.data.fill_(0)  # Weight=0 makes output depend only on bias
        self.switch.bias.data.fill_(1)  # Bias=1 means switch starts at 1.0 (small RF)

        # ============ Dual Receptive Field Weights ============
        # weight_diff: learnable difference between small and large receptive field kernels
        # Large kernel = small kernel + weight_diff
        self.weight_diff = nn.Parameter(torch.Tensor(self.weight.size()))
        self.weight_diff.data.zero_()  # Initially, both kernels are identical

        # ============ Context Modules ============
        # Pre-context: Adds global context (via global pooling) before convolution
        # 1x1 conv initialized to zero (identity at start of training)
        self.pre_context = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1, bias=True)
        self.pre_context.weight.data.fill_(0)
        self.pre_context.bias.data.fill_(0)

        # Post-context: Adds global context after convolution
        # 1x1 conv initialized to zero (identity at start of training)
        self.post_context = nn.Conv2d(self.out_channels,
                                      self.out_channels,
                                      kernel_size=1,
                                      bias=True)
        self.post_context.weight.data.fill_(0)
        self.post_context.bias.data.fill_(0)

        # ============ Normalization and Activation ============
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = (self.default_act
                    if act is True else act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass implementing switchable atrous convolution with context.

        Process:
            1. Pre-context: Add global context to input
            2. Compute switch weights based on local average
            3. Compute small receptive field output (standard dilation)
            4. Compute large receptive field output (3x dilation)
            5. Interpolate between outputs using switch weights
            6. Post-context: Add global context to output
            7. Apply batch norm and activation

        Args:
            x: Input tensor of shape (B, C_in, H, W).

        Returns:
            Output tensor of shape (B, C_out, H, W).
        """
        # ============ Step 1: Pre-context ============
        # Global average pooling to get channel-wise global information
        avg_x = F.adaptive_avg_pool2d(x, output_size=1)  # (B, C, 1, 1)
        # Transform global context
        avg_x = self.pre_context(avg_x)  # (B, C, 1, 1)
        # Broadcast and add to input (adds global context to every spatial location)
        avg_x = avg_x.expand_as(x)  # (B, C, H, W)
        x = x + avg_x

        # ============ Step 2: Compute Switch Weights ============
        # Compute local average using 5x5 window (reflects local spatial context)
        avg_x = F.pad(x, pad=(2, 2, 2, 2), mode="reflect")  # Pad to maintain size
        avg_x = F.avg_pool2d(avg_x, kernel_size=5, stride=1, padding=0)  # (B, C, H, W)
        # Switch values in [0, 1] range (though not explicitly constrained)
        switch = self.switch(avg_x)  # (B, 1, H, W)

        # ============ Step 3: Small Receptive Field Branch ============
        # Use standardized base weights with original dilation
        weight = self._get_weight(self.weight)
        out_s = super()._conv_forward(x, weight, None)  # Small RF output

        # ============ Step 4: Large Receptive Field Branch ============
        # Temporarily modify padding and dilation (3x larger)
        ori_p = self.padding
        ori_d = self.dilation
        self.padding = tuple(3 * p for p in self.padding)  # 3x padding
        self.dilation = tuple(3 * d for d in self.dilation)  # 3x dilation

        # Add weight difference to create large RF kernel
        weight += self.weight_diff
        out_l = super()._conv_forward(x, weight, None)  # Large RF output

        # Restore original padding and dilation
        self.padding = ori_p
        self.dilation = ori_d

        # ============ Step 5: Interpolate Between Branches ============
        # Weighted combination: switch=1 → small RF, switch=0 → large RF
        out = switch * out_s + (1 - switch) * out_l

        # ============ Step 6: Post-context ============
        # Global average pooling on output
        avg_x = F.adaptive_avg_pool2d(out, output_size=1)  # (B, C_out, 1, 1)
        # Transform global context
        avg_x = self.post_context(avg_x)
        # Broadcast and add (adds global output context)
        avg_x = avg_x.expand_as(out)
        out = out + avg_x

        # ============ Step 7: Normalize and Activate ============
        return self.act(self.bn(out))


# ============================================================================
# Ultralytics Bottleneck Reference (from ultralytics.nn.modules)
# ============================================================================
# class Bottleneck(nn.Module):
#     """Standard bottleneck block with residual connection.
#
#     Structure: 1x1 Conv (compress) -> 3x3 Conv (extract) -> residual add.
#     Commonly used in ResNet-style architectures.
#
#     Args:
#         c1 (int): Input channels.
#         c2 (int): Output channels.
#         shortcut (bool): Whether to use shortcut connection.
#         g (int): Groups for convolutions.
#         k (tuple): Kernel sizes for convolutions, e.g., (1, 3) or (3, 3).
#         e (float): Expansion ratio for hidden channels.
#     """


class BottleneckSAC(Bottleneck):
    """Bottleneck block with Switchable Atrous Convolution (SAC).

    Extends the standard Ultralytics Bottleneck by replacing the second
    convolution layer (cv2) with SAConv2d, enabling adaptive receptive field
    selection for better multi-scale feature extraction.

    Inherits from ultralytics.nn.modules.Bottleneck and overrides cv2 with SAC.
    """

    def __init__(self,
                 c1: int,
                 c2: int,
                 shortcut: bool = True,
                 g: int = 1,
                 k: tuple[int, int] = (1, 3),
                 e: float = 0.5) -> None:
        """Initializes bottleneck layer with SAC for adaptive receptive fields.

        Args:
            c1: Number of input channels.
            c2: Number of output channels.
            shortcut: Whether to add shortcut connection. Only applied when
                c1 == c2. Defaults to True.
            g: Groups for grouped convolution in SAConv2d. Defaults to 1.
            k: Tuple of kernel sizes (k1, k2) for cv1 and cv2 respectively.
                Defaults to (1, 3).
            e: Channel expansion factor for hidden channels. Defaults to 0.5.
        """
        # Initialize parent Bottleneck (sets up cv1, cv2, and self.add flag)
        super().__init__(c1, c2, shortcut, g, k, e)

        # Override cv2: Replace standard Conv with SAConv2d
        # This enables adaptive receptive field switching in the second conv
        self.cv2 = SAConv2d(
            int(c2 * e),  # Input channels to cv2 (hidden channels)
            c2,  # Output channels
            k[1],  # Kernel size (typically 3)
            1,  # Stride
            g=g  # Groups
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through bottleneck with SAC and optional residual.

        Process:
            1. Pass through cv1 (1x1 conv for channel compression)
            2. Pass through cv2 (SAConv2d for adaptive feature extraction)
            3. Add residual connection if self.add is True

        Args:
            x: Input tensor of shape (B, C1, H, W).

        Returns:
            Output tensor of shape (B, C2, H, W).
        """
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


# ============================================================================
# Ultralytics C3 Reference (from ultralytics.nn.modules)
# ============================================================================
# class C3(nn.Module):
#     """Cross Stage Partial (CSP) Bottleneck with 3 convolutions.
#
#     Splits input into two branches:
#     - Branch 1: Multiple bottleneck blocks
#     - Branch 2: Direct 1x1 conv bypass
#     Concatenates and fuses branches with final 1x1 conv.
#
#     Args:
#         c1 (int): Input channels.
#         c2 (int): Output channels.
#         n (int): Number of Bottleneck blocks.
#         shortcut (bool): Whether to use shortcut connections.
#         g (int): Groups for convolutions.
#         e (float): Expansion ratio.
#     """


class C3SAC(C3):
    """CSP Bottleneck with 3 convolutions using Switchable Atrous Convolution.

    Extends the standard Ultralytics C3 module by replacing standard bottleneck
    blocks with BottleneckSAC, enabling adaptive receptive field selection
    within the CSP structure. This is particularly useful for handling
    multi-scale features in object detection tasks.

    Inherits from ultralytics.nn.modules.C3 and overrides the bottleneck
    sequence (self.m) with BottleneckSAC blocks.
    """

    def __init__(self,
                 c1: int,
                 c2: int,
                 n: int = 1,
                 shortcut: bool = True,
                 g: int = 1,
                 e: float = 0.5) -> None:
        """Initializes CSP Bottleneck with SAC blocks for adaptive receptive
        fields.

        Args:
            c1: Number of input channels.
            c2: Number of output channels.
            n: Number of BottleneckSAC blocks to stack. Defaults to 1.
            shortcut: Whether to use shortcut connections in bottlenecks.
                Defaults to True.
            g: Groups for grouped convolution in SAConv2d. Defaults to 1.
            e: Channel expansion factor for hidden channels. Defaults to 0.5.
        """
        # Initialize parent C3 module (sets up cv1, cv2, cv3, and default self.m)
        super().__init__(c1, c2, n, shortcut, g, e)

        c_ = int(c2 * e)  # Hidden channels

        # Override bottleneck sequence: Replace standard Bottleneck with BottleneckSAC
        # Each BottleneckSAC uses e=1.0 to maintain channel dimensions within the sequence
        self.m = nn.Sequential(*(BottleneckSAC(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
