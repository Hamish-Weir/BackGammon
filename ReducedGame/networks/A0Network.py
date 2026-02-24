import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.Board import BOARD_SIZE, PRIOR_SIZE

class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)     # Convolution           (1)
        out = self.bn1(out)     # Batch Normalisation   (2)
        out = self.relu(out)   # ReLu                  (3)
        out = self.conv2(out)   # Convolution           (4)
        out = self.bn2(out)     # Batch Normalisation   (5)
        out = out + x           # skip connection       (6)
        out = self.relu(out)   # ReLu                  (7)
        return out

class A0Network(nn.Module):
    """
    Residual network for 11x11 board with policy and value heads.

    Args:
        board_size: int (expected 11)
    Forward returns:
        - policy:   probabilities (B, policy_output_size) (softmax over logits)
        - value:    scalar in [-1,1] (B, 1)
    """
    def __init__(
        self,
        input_size: int = BOARD_SIZE+4,
        output_size: int = PRIOR_SIZE,
        in_channels: int = 1,
        channels: int = 128,
        num_res_blocks: int = 10,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        # Initial Convolution -> Batch Norm -> ReLu
        self.conv_initial   = nn.Conv1d(in_channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_initial     = nn.BatchNorm1d(channels) 
        self.relu           = nn.ReLU(inplace=True)

        # Residual tower
        self.res_blocks     = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_res_blocks)])

        # Policy head
        self.policy_conv = nn.Conv1d(channels, 2, kernel_size=1, stride=1, padding=0, bias=False) 
        self.policy_bn = nn.BatchNorm1d(2) 
        self.policy_fc = nn.Linear(2 * input_size, (output_size)) 

        # Value head
        self.value_conv = nn.Conv1d(channels, 1, kernel_size=1, stride=1, padding=0, bias=False) 
        self.value_bn = nn.BatchNorm1d(1) 
        self.value_fc1 = nn.Linear(1 * input_size, 256)
        self.value_fc2 = nn.Linear(256, 1) 
        # torch.tanh() (7)

        # initialize weights (following common conv initialization)
        self._initialize_weights()

    def _initialize_weights(self):
        # Kaiming for conv, linear; 
        # BatchNorm weights = 1, bias = 0
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> (float):
        """
        x: tensor shape (B, 1, 11, 11) or (B, 11, 11)
        """
        # Allow single-sample input
        if x.dim() == 1:
            x = x.unsqueeze(0)  # (1, INPUT_SIZE)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, INPUT_SIZE)

        assert x.dim() == 3 and x.size(1) == 1 and x.size(2) == self.input_size, \
            f"Expected input shape (B, 2, {self.input_size}), got {tuple(x.shape)}"

        # initial
        trunk = self.conv_initial(x)    # (1)
        trunk = self.bn_initial(trunk)  # (2)
        trunk = self.relu(trunk)        # (3)

        # residual tower
        trunk = self.res_blocks(trunk)

        # Policy head
        p = self.policy_conv(trunk)             # (1)
        p = self.policy_bn(p)                   # (2)
        p = self.relu(p)                        # (3)
        p = p.view(p.size(0), -1)               # flatten
        policy_logits = self.policy_fc(p)       # (4)

        # Value head
        v = self.value_conv(trunk)  # (1)
        v = self.value_bn(v)        # (2)
        v = self.relu(v)            # (3)
        v = v.view(v.size(0), -1)   # flatten
        v = self.value_fc1(v)       # (4)
        v = self.relu(v)            # (5)
        v = self.value_fc2(v)       # (6)
        v = torch.tanh(v)           # (7)

        return v, policy_logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
