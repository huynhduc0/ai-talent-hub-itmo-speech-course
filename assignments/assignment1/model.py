import torch
from torch import nn
import torch.nn.functional as F

class SpeechCNN(nn.Module):
    def __init__(self, in_channels=80, num_classes=2, groups=1):
        """
        Args:
            in_channels: Equals to n_mels
            num_classes: 2 for binary classification ("YES" / "NO")
            groups: Controls the torch.nn.Conv1d groups parameter.
        """
        super().__init__()
        
        # We ensure groups divides into in_channels and out_channels cleanly depending on experimentation.
        # But in_channels might be 20, 40, or 80.
        # A common divisor for (20, 40, 80) and 32 / 64 / 128 is not straightforward if groups > 1
        # groups needs to divide in_channels and out_channels.
        # So we might need to adjust channels to be divisible by 16.
        # Let's adjust in_channels using a dummy 1x1 conv if groups requirement fails.
        self.groups = groups
        
        # Project n_mels to a fixed dimension that is divisible by 16 (max groups requirement)
        # 16 is max groups from {2, 4, 8, 16}.
        base_channels = 32
        self.input_proj = nn.Conv1d(in_channels, base_channels, kernel_size=1)
        
        self.conv1 = nn.Conv1d(base_channels, base_channels, kernel_size=3, padding=1, groups=groups)
        self.bn1 = nn.BatchNorm1d(base_channels)
        
        self.conv2 = nn.Conv1d(base_channels, base_channels*2, kernel_size=3, padding=1, groups=groups)
        self.bn2 = nn.BatchNorm1d(base_channels*2)
        
        self.conv3 = nn.Conv1d(base_channels*2, base_channels*4, kernel_size=3, padding=1, groups=groups)
        self.bn3 = nn.BatchNorm1d(base_channels*4)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc1 = nn.Linear(base_channels*4, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # x shape should be (B, C, T) where C = n_mels
        x = self.input_proj(x)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, 2)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, 2)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def calculate_model_metrics(model, input_size=(1, 80, 100)):
    """ Returns (num_params, flops). Handles FLOPs using thop if available. """
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    flops = 0
    try:
        from thop import profile
        dummy_in = torch.randn(*input_size).to(next(model.parameters()).device)
        macs, _ = profile(model, inputs=(dummy_in, ), verbose=False)
        flops = macs * 2
    except ImportError:
        pass
    return params, flops
