import torch
import torch.nn as nn


config = [(32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4],  # To this point is Darknet-53
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S"]
#tuple: (out_channels, kernel, stride)
#list: B - residual block, number of repeats
#"s" - scale prediction block and computing the yolo loss
#"u" - upsampling the feature map and concat with a previous layer



class CNN_block(nn.Module):
    def __init__(self, in_channels, out_channels, batch_act = True, **kwargs):
        super(CNN_block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias = not batch_act, **kwargs)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.use_bn_act = batch_act
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        conved = self.conv(x)
        if self.use_bn_act:
            return self.leaky_relu(self.batch_norm(conved))
        else:
            return conved



class Residual_Block(nn.Module):
    def __init__(self, channels, use_residual = True, num_repeats =1):
        super(Residual_Block, self).__init__()
        self.layers = nn.ModuleList()
        for repeat in range(num_repeats):
            self.layers+=[nn.Sequential(CNN_block(channels, channels//2, kernel_size = 1), CNN_block(channels//2, channels, kernel_size =3, padding = 1))]

        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
                x = layer(x) + x if self.use_residual else layer(x)
        return x


class scale_prediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(scale_prediction, self).__init__()
        self.prediction = nn.Sequential(CNN_block(in_channels, 2*in_channels, kernel_size = 3, padding = 1),
                                        CNN_block(2*in_channels, (num_classes + 5)*3, batch_act=False, kernel_size =1))
        self.num_classes = num_classes

    def forward(self, x):
        #old dimentionality??: [num_of_pictures, (num_classes + 5)*3, num_boxes, 13 grid, 13grid]
        return self.prediction(x).reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3]).permute(0, 1, 3, 4, 2)
        #we want dimentionality num_of_pictures, num_boxes, 13 grid, 13grid, (num_classes + 5)



class YOLO_v3(nn.Module):
    def __init__(self, in_channels = 3, num_classes = 20):
        super(YOLO_v3, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.layers = self._create_conv_layers()

    def forward(self, x):
        #we are goinf to have one output for each scale prediction
        outputs = []

        #where we concat the layers, skip connections for concats
        route_connections = []

        for layer in self.layers:
            if isinstance(layer, scale_prediction):
                #we want to do the prediction, add the output and then after this we want to continue with the netwok
                outputs.append(layer(x))
                continue

            x = layer(x)
            if isinstance(layer, Residual_Block) and layer.num_repeats ==8:
                #we wanr to add the roots
                route_connections.append(x)

            elif isinstance(layer, nn.Upsample):
                #if we are using upsample we want to concat with the last root connection
                x = torch.cat([x,  route_connections[-1]], dim = 1)
                route_connections.pop()

        return outputs



    def _create_conv_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels
        for module in config:
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(CNN_block(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = 1 if kernel_size ==3 else 0))
                in_channels = out_channels

            elif isinstance(module, list):
                num_repeats = module[1]
                layers.append(Residual_Block(in_channels, num_repeats = num_repeats))

            elif isinstance(module, str):
                if module=="S":
                    layers+=[Residual_Block(in_channels, use_residual=False, num_repeats = 1), CNN_block(in_channels, in_channels//2, kernel_size = 1),
                             scale_prediction(in_channels//2, self.num_classes)]
                    in_channels = in_channels//2

                elif module == "U":
                    layers.append(nn.Upsample(scale_factor = 2))

                    #this is where we concat, so we have more channels
                    in_channels = in_channels*3

        return layers


if __name__ == "__main__":
    num_classes = 20
    image_size = 416
    model = YOLO_v3(num_classes=num_classes)
    x = torch.randn((2, 3, image_size, image_size))
    out = model(x)
    assert model(x)[0].shape == (2, 3, 13, 13, num_classes+5)
    assert model(x)[1].shape == (2, 3, image_size//16, image_size//16, num_classes+5)
    assert model(x)[2].shape==(2, 3, image_size//8, image_size//8, num_classes+5)

