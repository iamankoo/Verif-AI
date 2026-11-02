class SRMLayer(nn.Module):
    """Fixed (non-trainable) high-pass residual filters. Kernels are those of the original project."""

    def __init__(self):
        super().__init__()
        k = np.array([
            [[-1, 2, -1], [2, -4, 2], [-1, 2, -1]],
            [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
            [[0, -1, 0], [-1, 4, -1], [0, -1, 0]],
        ], dtype=np.float32) / 4.0
        self.kernel = nn.Parameter(torch.tensor(k).view(3, 1, 3, 3).repeat(1, 3, 1, 1), requires_grad=False)

    def forward(self, x):
        return F.conv2d(x, self.kernel, stride=1, padding=1)


class TwoStreamNet(nn.Module):
    """RGB stream + SRM-noise stream (two EfficientNet-B0), concatenated 2x1280 -> 512 -> 1 -> sigmoid.
    Attribute names match the original checkpoints, so they load with strict=True."""

    def __init__(self, pretrained: bool = False):
        super().__init__()
        make = EfficientNet.from_pretrained if pretrained else EfficientNet.from_name
        self.spatial_net = make("efficientnet-b0")
        self.srm_layer = SRMLayer()
        self.noise_net = make("efficientnet-b0")
        self.fc = nn.Sequential(nn.Linear(1280 * 2, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, 1), nn.Sigmoid())

    def logits(self, x):
        """Pre-sigmoid score. forward(x) == sigmoid(logits(x)); used for numerically stable training."""
        f1 = F.adaptive_avg_pool2d(self.spatial_net.extract_features(x), 1).flatten(1)
        f2 = F.adaptive_avg_pool2d(self.noise_net.extract_features(self.srm_layer(x)), 1).flatten(1)
        return self.fc[:-1](torch.cat((f1, f2), dim=1))

    def forward(self, x):
        return torch.sigmoid(self.logits(x))
