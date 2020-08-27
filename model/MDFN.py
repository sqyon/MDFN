import torch
import torch.nn as nn
import torch.nn.functional as F


def get_model(param):
    return MultiDimensionFusionNetwork(param)


class DynamicFiltersBranch(nn.Module):
    def __init__(self, channel_num, df_size, sr_rate):
        super(DynamicFiltersBranch, self).__init__()
        n = channel_num
        self.conv1 = nn.Conv3d(n, sr_rate * sr_rate * n, kernel_size=1, stride=1, padding=0, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(sr_rate)
        self.conv2 = nn.Conv3d(n, df_size * df_size, kernel_size=1, stride=1, padding=0, bias=True)
        self.relu0 = nn.PReLU()
        self.relu1 = nn.PReLU()
        self.df_size = df_size
        self.sr_rate = sr_rate
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform_(m.weight, a=0.25, mode='fan_in', nonlinearity='leaky_relu')
                m.bias.data.zero_()

    def forward(self, x):
        batch, channel, height_view, width_view, height, width = list(x.shape)
        x = x.reshape(batch, channel, height_view * width_view, height, width)
        x = self.relu0(self.conv1(x))
        channel = x.shape[1]
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(batch * height_view * width_view, channel, height, width)
        x = self.pixel_shuffle(x)
        channel, height, width = list(x.shape[1:])
        x = x.reshape(batch, height_view * width_view, channel, height, width)
        x = x.permute(0, 2, 1, 3, 4)
        x = self.relu1(self.conv2(x))
        x = x.permute(0, 1, 3, 4, 2)
        x = x.reshape(batch, 1, height_view, width_view, height, width, self.df_size * self.df_size)
        x = F.softmax(x, dim=6)
        x = x.reshape(batch, 1, height_view, width_view, height, width, self.df_size, self.df_size)
        return x


class ResidualBranch(nn.Module):
    def __init__(self, channel_num, sr_rate):
        super(ResidualBranch, self).__init__()
        self.sr_rate = sr_rate
        self.conv1 = nn.Conv3d(channel_num, channel_num, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv2 = nn.Conv3d(channel_num, sr_rate * sr_rate, kernel_size=1, stride=1, padding=0, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(sr_rate)
        self.relu0 = nn.PReLU()
        self.relu1 = nn.PReLU()
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform_(m.weight, a=0.25, mode='fan_in', nonlinearity='leaky_relu')
                m.bias.data.zero_()

    def forward(self, x):
        batch, channel, height_view, width_view, height, width = list(x.shape)
        x = x.reshape(batch, channel, height_view * width_view, height, width)
        x = self.relu0(self.conv1(x))
        x = self.relu1(self.conv2(x))
        channel = x.shape[1]
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(batch * height_view * width_view, channel, height, width)
        x = self.pixel_shuffle(x)
        channel, height, width = list(x.shape[1:])
        x = x.reshape(batch, height_view, width_view, channel, height, width)
        x = x.permute(0, 3, 1, 2, 4, 5)
        return x


class MultiDimensionFusionBlock(nn.Module):
    def __init__(self, channel_num):
        super(MultiDimensionFusionBlock, self).__init__()
        n = channel_num
        assert n % 4 == 0, 'channel_num of AllDimBlock should be even.'
        self.channel = n // 4
        kerner_size_s, padding_s = (1, 3, 3), (0, 1, 1)
        kerner_size_a, padding_a = (3, 3, 1), (1, 1, 0)

        self.xy_conv = nn.Conv3d(n, self.channel, kernel_size=kerner_size_s, stride=1, padding=padding_s, bias=True)
        self.uv_conv = nn.Conv3d(n, self.channel, kernel_size=kerner_size_a, stride=1, padding=padding_a, bias=True)
        self.relu0 = nn.PReLU()
        self.relu1 = nn.PReLU()
        self.vy_conv = nn.Conv3d(n, self.channel, kernel_size=kerner_size_s, stride=1, padding=padding_s, bias=True)
        self.ux_conv = nn.Conv3d(n, self.channel, kernel_size=kerner_size_a, stride=1, padding=padding_a, bias=True)
        self.relu2 = nn.PReLU()
        self.relu3 = nn.PReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform_(m.weight, a=0.25, mode='fan_in', nonlinearity='leaky_relu')
                m.bias.data.zero_()

    def forward(self, x):
        batch, channel, height_view, width_view, height, width = list(x.shape)
        n = self.channel

        x_xy = x.reshape(batch, channel, height_view * width_view, height, width)
        x_xy = self.relu0(self.xy_conv(x_xy))
        x_xy = x_xy.reshape(batch, n, height_view, width_view, height, width)

        x_uv = x.reshape(batch, channel, height_view, width_view, height * width)
        x_uv = self.relu1(self.uv_conv(x_uv))
        x_uv = x_uv.reshape(batch, n, height_view, width_view, height, width)

        x = x.permute(0, 1, 2, 4, 3, 5)

        x_vy = x.reshape(batch, channel, height_view * height, width_view, width)
        x_vy = self.relu2(self.vy_conv(x_vy))
        x_vy = x_vy.reshape(batch, n, height_view, height, width_view, width)
        x_vy = x_vy.permute(0, 1, 2, 4, 3, 5)

        x_ux = x.reshape(batch, channel, height_view, height, width_view * width)
        x_ux = self.relu3(self.ux_conv(x_ux))
        x_ux = x_ux.reshape(batch, n, height_view, height, width_view, width)
        x_ux = x_ux.permute(0, 1, 2, 4, 3, 5)

        ret = torch.cat([x_vy, x_ux, x_xy, x_uv], dim=1)

        return ret


class MultiDimensionFusionArchitecture(nn.Module):
    def __init__(self, channel_num, block_num, in_channel):
        super(MultiDimensionFusionArchitecture, self).__init__()
        self.pre_conv = nn.Conv3d(in_channel, channel_num, kernel_size=3, padding=1, bias=True)
        self.relu = nn.PReLU()
        channel_num = channel_num
        layers = list()
        for i in range(block_num):
            layers.append(MultiDimensionFusionBlock(channel_num))
        self.seqn = nn.Sequential(*layers)
        self.channel_num = channel_num

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform_(m.weight, a=0.25, mode='fan_in', nonlinearity='leaky_relu')
                m.bias.data.zero_()

    def get_channel(self):
        return self.channel_num

    def forward(self, x):
        batch, channel, height_view, width_view, height, width = list(x.shape)
        x = x.reshape(batch, channel, height_view * width_view, height, width)
        x = self.relu(self.pre_conv(x))
        channel = x.shape[1]
        x = x.reshape(batch, channel, height_view, width_view, height, width)
        x = self.seqn(x)
        return x


class MultiDimensionFusionNetwork(nn.Module):
    def __init__(self, params):
        super(MultiDimensionFusionNetwork, self).__init__()
        self.df_size = params['df_size']
        self.sr_rate = params['sr_rate']

        self.MDFA = MultiDimensionFusionArchitecture(80, 8, 1)
        self.DFB = DynamicFiltersBranch(80, self.df_size, self.sr_rate)

        self.RB = ResidualBranch(80, self.sr_rate)

    def filter_convolution(self, dynamic_filter, X):
        batch, channel, height_view, width_view, height, width = list(X.shape)
        pad_size = (self.df_size - 1) // 2
        x_pad = X.permute(0, 4, 5, 1, 2, 3)
        x_pad = x_pad.reshape(batch * height * width, channel, height_view, width_view)
        x_pad = F.pad(x_pad, pad=[pad_size, pad_size, pad_size, pad_size], mode='replicate')
        # batch*height*width, channel, height_view + pad_size * 2, width_view + pad_size * 2
        x_multiplier = torch.zeros_like(dynamic_filter, device=dynamic_filter.device)
        for i in range(self.df_size):
            for j in range(self.df_size):
                # N*H*W, C, Vh, Vw
                upsampled = x_pad[:, :, i:height_view + i, j:width_view + j]
                upsampled = upsampled.reshape(batch, height, width, channel, height_view, width_view)
                upsampled = upsampled.permute(0, 4, 5, 3, 1, 2)
                upsampled = upsampled.reshape(batch * height_view * width_view, channel, height, width)
                upsampled = F.interpolate(upsampled, scale_factor=self.sr_rate, mode='nearest')
                upsampled = upsampled.reshape(
                    batch, height_view, width_view, channel, height * self.sr_rate, width * self.sr_rate)
                upsampled = upsampled.permute(0, 3, 1, 2, 4, 5)
                # N, C, Vh, Vw, H, W
                x_multiplier[:, :, :, :, :, :, i, j] = upsampled
        product = torch.mul(x_multiplier, dynamic_filter)
        product = product.reshape(
            batch, channel, height_view, width_view, height * self.sr_rate, width * self.sr_rate, -1)
        product = torch.sum(product, dim=6)
        return product

    def forward(self, X):
        X = X['input']
        all_feature = self.MDFA(X)
        residual = self.RB(all_feature)

        dynamic_filter = self.DFB(all_feature)
        residual += self.filter_convolution(dynamic_filter, X)
        return {'output': residual}


if __name__ == '__main__':
    param = {'df_size': 5, 'sr_rate': 2}
    net = MultiDimensionFusionNetwork(param)
    # net = net.cuda()
    x = torch.randn(2, 1, 7, 7, 32, 32)
    input = {'input': x}
    # flops, params = profile(net, inputs=(x,))
    # cflops, cparams = clever_format([flops, params], "%.3f")
    # print(f'flops:{flops} - params:{params} - {cflops} - {cparams}')
    y = net(input)
    print(y['output'].shape)
