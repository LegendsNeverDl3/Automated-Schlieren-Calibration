import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """Depthwise-Separable Convolution block with BatchNorm and Mish/ReLU activation."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)

class SpatialFrequencyAttention(nn.Module):
    """Attention module emphasizing periodic vertical fringe features (Sobel/fringe resonance)."""
    def __init__(self, channels):
        super().__init__()
        # Horizontal 1D conv pooling to capture vertical standing wave column structure
        self.vertical_pool = nn.AdaptiveAvgPool2d((None, 1))
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        att = self.vertical_pool(x)
        att = self.fc(att)
        return x * att

class SpatioTemporalWaveNet(nn.Module):
    """
    Lightweight Deep Neural Network designed for Schlieren standing wave detection.
    
    Inputs:
        Tensor of shape (B, 2, 128, 128):
            Channel 0: Temporal Mean frame (stationary acoustic standing wave field)
            Channel 1: Temporal Variance map (turbulent thermal convection activity)
            
    Outputs:
        quality_score: Scalar in [0.0, 1.0] predicting wave clarity/contrast.
        regime_logits: 3-class logits [0: Under-Cutoff, 1: Optimal Cutoff, 2: Over-Cutoff].
    """
    def __init__(self, in_channels=2, num_regimes=3):
        super().__init__()
        
        # Initial stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2, bias=False),  # 128 -> 64
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Convolutional stages
        self.stage1 = ConvBlock(32, 48, stride=2)   # 64 -> 32
        self.attn1 = SpatialFrequencyAttention(48)
        
        self.stage2 = ConvBlock(48, 64, stride=2)   # 32 -> 16
        self.attn2 = SpatialFrequencyAttention(64)
        
        self.stage3 = ConvBlock(64, 96, stride=2)   # 16 -> 8
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Shared dense representation
        self.fc_shared = nn.Sequential(
            nn.Linear(96, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
        # Head 1: Wave Quality Regression (0.0 to 1.0)
        self.quality_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Head 2: Cutoff Regime Classification (Under / Optimal / Over)
        self.regime_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_regimes)
        )

    def forward(self, x):
        feat = self.stem(x)
        feat = self.attn1(self.stage1(feat))
        feat = self.attn2(self.stage2(feat))
        feat = self.stage3(feat)
        
        pooled = self.global_pool(feat).flatten(1)
        shared = self.fc_shared(pooled)
        
        quality = self.quality_head(shared).squeeze(-1)
        regime_logits = self.regime_head(shared)
        
        return quality, regime_logits

if __name__ == "__main__":
    # Quick architecture validation
    model = SpatioTemporalWaveNet()
    dummy_input = torch.randn(4, 2, 128, 128)
    q, r = model(dummy_input)
    print(f"Model initialized successfully!")
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Quality shape: {q.shape} (values: {q.detach().numpy().round(3)})")
    print(f"Regime shape:  {r.shape}")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {num_params:,} (~{num_params*4/1024:.1f} KB)")
