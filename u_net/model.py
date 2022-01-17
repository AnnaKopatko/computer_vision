import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class double_conv(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, 3, 1, 1, bias = False),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_dim, output_dim, 3, 1, 1),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class u_net(nn.Module):
    def __init__(self, input_dim = 3, output_dim =1, features = [64, 128, 256, 512]):
        super(u_net, self).__init__()
        self.downsampling = nn.ModuleList()
        self.upsampling = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride = 2)

        for feature in features:
            self.downsampling.append(double_conv(input_dim, feature))
            input_dim = feature

        for feature in features[::-1]:
            self.upsampling.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride = 2))
            self.upsampling.append(double_conv(feature*2, feature))

            self.bottleneck = double_conv(features[-1], features[-1]*2)
            self.final_conv = nn.Conv2d(features[0], output_dim, kernel_size=1)

    def forward(self, x):

        skip_connections = []

        for down_layer in self.downsampling:
            x = down_layer(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for index in range(0, len(self.upsampling), 2):
            x = self.upsampling[index](x)
            skip_connection = skip_connections[index//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size = skip_connection.shape[2:])
            concat_skip = torch.concat((skip_connection, x), dim= 1)
            x = self.upsampling[index+1](concat_skip)

        return self.final_conv(x)



# def test():
#     x = torch.randn((3, 1, 161, 161))
#
#     model = u_net(input_dim=1, output_dim=1)
#     preds = model(x)
#     print(x.shape)
#     assert preds.shape ==x.shape
#
# test()