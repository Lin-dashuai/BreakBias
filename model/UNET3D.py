import torch
import torch.nn as nn
import torch.nn.functional as F


def get_norm_layer(normalization, num_features):
    """
    Returns a normalization layer based on the input string.
    """
    if normalization == 'batchnorm':
        return nn.BatchNorm3d(num_features)
    elif normalization == 'instancenorm':
        return nn.InstanceNorm3d(num_features)
    elif normalization is None or normalization.lower() == 'none':
        return nn.Identity()
    else:
        raise ValueError(f"Unknown normalization type: {normalization}")


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, normalization='batchnorm'):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            get_norm_layer(normalization, mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            get_norm_layer(normalization, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, normalization='batchnorm'):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels, normalization=normalization)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, trilinear=True, normalization='batchnorm'):
        super().__init__()
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, mid_channels=in_channels // 2,
                                   normalization=normalization)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, normalization=normalization)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Pad x1 to the size of x2
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# =========================================================================
# The main UNet3D class, now fully compatible with VNet's parameters
# =========================================================================
class UNet3D(nn.Module):
    def __init__(self, n_channels, n_classes, n_filters=16,
                 normalization='batchnorm', has_dropout=True, dropout_p=0.5, trilinear=True):
        super(UNet3D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.trilinear = trilinear

        self.inc = DoubleConv(n_channels, n_filters, normalization=normalization)
        self.down1 = Down(n_filters, n_filters * 2, normalization=normalization)
        self.down2 = Down(n_filters * 2, n_filters * 4, normalization=normalization)
        self.down3 = Down(n_filters * 4, n_filters * 8, normalization=normalization)
        factor = 2 if trilinear else 1
        self.down4 = Down(n_filters * 8, n_filters * 16 // factor, normalization=normalization)

        self.up1 = Up(n_filters * 16, n_filters * 8 // factor, trilinear, normalization=normalization)
        self.up2 = Up(n_filters * 8, n_filters * 4 // factor, trilinear, normalization=normalization)
        self.up3 = Up(n_filters * 4, n_filters * 2 // factor, trilinear, normalization=normalization)
        self.up4 = Up(n_filters * 2, n_filters, trilinear, normalization=normalization)
        self.outc = OutConv(n_filters, n_classes)

        # Dropout layers controlled by has_dropout, mirroring VNet's behavior
        self.dropout1 = nn.Dropout3d(p=dropout_p) if has_dropout else nn.Identity()
        self.dropout2 = nn.Dropout3d(p=dropout_p) if has_dropout else nn.Identity()

    def encoder(self, x):
        features = []
        x1 = self.inc(x);
        features.append(x1)
        x2 = self.down1(x1);
        features.append(x2)
        x3 = self.down2(x2);
        features.append(x3)
        x4 = self.down3(x3);
        features.append(x4)
        x5 = self.down4(x4);
        features.append(x5)
        return features

    # ===================== [NEW METHOD ADDED] =====================
    # 这个方法专门用于DDFP，模拟VNet的行为
    def decoder(self, features):
        # features 列表包含了所有编码器的输出
        # features[0] -> x1, features[1] -> x2, ..., features[4] -> x5 (最深层)
        x1, x2, x3, x4, x5 = features[0], features[1], features[2], features[3], features[4]

        # 应用dropout在瓶颈层（与forward方法保持一致）
        x5_drop = self.dropout1(x5)

        # 执行标准的U-Net解码流程
        x = self.up1(x5_drop, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x_up = self.up4(x, x1)

        # 应用末端dropout（与forward方法保持一致）
        x_up_drop = self.dropout2(x_up)

        logits = self.outc(x_up_drop)
        return logits

    def forward(self, x):
        # 1. 调用 encoder 获取所有特征
        features = self.encoder(x)

        # 2. 调用 decoder 获取最终预测
        logits = self.decoder(features)

        # 3. 返回符合要求的字典
        return {'pred': logits, 'features': features}