import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.mbv4 import mobilenetv4_conv_small


class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_channels, affine=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layer(x)


class TinyFeat(nn.Module):
    def __init__(self):
        super().__init__()

        # bb
        self.backbone = mobilenetv4_conv_small()

        self.block_fusion = nn.Sequential(
            ConvBN(64, 64, stride=1),
            nn.Conv2d(64, 64, 1, padding=0)
        )

        self.heatmap_head = nn.Sequential(
            ConvBN(64, 64, 1, padding=0),
            ConvBN(64, 64, 1, padding=0),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )

        self.keypoint_head = nn.Sequential(
            ConvBN(64, 64, 1, padding=0),
            ConvBN(64, 64, 1, padding=0),
            ConvBN(64, 64, 1, padding=0),
            nn.Conv2d(64, 65, 1),
        )

        ########### ⬇️ Fine Matcher MLP ⬇️ ###########
        self.fine_matcher = nn.Sequential(
            nn.Linear(128, 512),
            nn.BatchNorm1d(512, affine=False),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512, affine=False),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512, affine=False),
            nn.ReLU(inplace=True),
            nn.Linear(512, 64),
        )

    def _unfold2d(self, x, ws=2):
        """
            Unfolds tensor in 2D with desired ws (window size) and concat the channels
        """
        B, C, H, W = x.shape
        x = x.unfold(2,  ws, ws).unfold(3, ws, ws)                             \
            .reshape(B, C, H//ws, W//ws, ws**2)
        return x.permute(0, 1, 4, 2, 3).reshape(B, -1, H//ws, W//ws)

    def forward(self, x):
        """
            input:
                x -> torch.Tensor(B, C, H, W) grayscale or rgb images
            return:
                feats     ->  torch.Tensor(B, 64, H/8, W/8) dense local features
                keypoints ->  torch.Tensor(B, 65, H/8, W/8) keypoint logit map
                heatmap   ->  torch.Tensor(B,  1, H/8, W/8) reliability map

        """
        x1, x2, x3, x4, x5 = self.backbone(x)
        # print(x1.shape, x2.shape, x3.shape, x4.shape, x5.shape)

        # # main backbone
        # x1 = self.block1(x)
        # x2 = self.block2(x1 + self.skip1(x))
        # x3 = self.block3(x2)
        # x4 = self.block4(x3)
        # x5 = self.block5(x4)

        # # pyramid fusion
        x4 = F.interpolate(x4, (x3.shape[-2], x3.shape[-1]), mode='bilinear')
        x5 = F.interpolate(x5, (x3.shape[-2], x3.shape[-1]), mode='bilinear')
        feats = self.block_fusion(x3 + x4 + x5)

        # heads
        heatmap = self.heatmap_head(feats)  # Reliability map
        keypoints = self.keypoint_head(
            self._unfold2d(x, ws=8))  # Keypoint map logits

        return feats, keypoints, heatmap
