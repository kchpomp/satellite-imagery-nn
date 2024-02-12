import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

########################################################################################################################
############################################### RESNET101 RESIDUAL CLASS ###############################################
########################################################################################################################
class ResidualBlock(nn.Module):
    def __init__(self, num_layers, in_channels, out_channels, identity_downsample=None, stride=1):
        assert num_layers in [18, 34, 50, 101, 152], "should be a a valid architecture"
        super(ResidualBlock, self).__init__()
        self.num_layers = num_layers
        if self.num_layers > 34:
            self.expansion = 4
        else:
            self.expansion = 1
        # ResNet50, 101, and 152 include additional layer of 1x1 kernels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        if self.num_layers > 34:
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        else:
            # for ResNet18 and 34, connect input directly to (3x3) kernel (skip first (1x1))
            self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x
        if self.num_layers > 34:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x
#**********************************************************************************************************************#


########################################################################################################################
#################################### RESNET101 BACKBONE CLASS FOR DEEPLABv3 MODEL ######################################
########################################################################################################################
class ResNet(nn.Module):
    def __init__(self, num_layers, block, image_channels, num_classes):
        assert num_layers in [18, 34, 50, 101, 152], f'ResNet{num_layers}: Unknown architecture! Number of layers has ' \
                                                     f'to be 18, 34, 50, 101, or 152 '
        super(ResNet, self).__init__()
        if num_layers < 50:
            self.expansion = 1
        else:
            self.expansion = 4
        if num_layers == 18:
            layers = [2, 2, 2, 2]
        elif num_layers == 34 or num_layers == 50:
            layers = [3, 4, 6, 3]
        elif num_layers == 101:
            layers = [3, 4, 23, 3]
        else:
            layers = [3, 8, 36, 3]
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNetLayers
        self.layer1 = self.make_layers(num_layers, block, layers[0], intermediate_channels=64, stride=1)
        self.layer2 = self.make_layers(num_layers, block, layers[1], intermediate_channels=128, stride=2)
        self.layer3 = self.make_layers(num_layers, block, layers[2], intermediate_channels=256, stride=2)
        self.layer4 = self.make_layers(num_layers, block, layers[3], intermediate_channels=512, stride=2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # print("ResNet101 Input: ", x.size())
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # print("Input to first Block: ", x.size())
        x = self.layer1(x)
        x = self.layer2(x)
        # print("Level 2 features size: ", x.size())
        x2 = self.dropout(x)
        x = self.layer3(x)
        x3 = self.dropout(x)
        # print("Level 3 features size: ", x.size())
        x = self.layer4(x)
        # print("Level 4 features size: ", x.size())
        x4 = self.dropout(x)
        return x2, x3, x4

    def make_layers(self, num_layers, block, num_residual_blocks, intermediate_channels, stride):
        layers = []

        identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, intermediate_channels*self.expansion, kernel_size=1, stride=stride),
                                            nn.BatchNorm2d(intermediate_channels*self.expansion))
        layers.append(block(num_layers, self.in_channels, intermediate_channels, identity_downsample, stride))
        self.in_channels = intermediate_channels * self.expansion # 256
        for i in range(num_residual_blocks - 1):
            layers.append(block(num_layers, self.in_channels, intermediate_channels)) # 256 -> 64, 64*4 (256) again
        return nn.Sequential(*layers)
    
def ResNet101(img_channels=6, num_classes=6):
    return ResNet(101, ResidualBlock, img_channels, num_classes)

def ResNet50(img_channels=6, num_classes=6):
    return ResNet(50, ResidualBlock, img_channels, num_classes)

def ResNet152(img_channels=6, num_classes=6):
    return ResNet(152, ResidualBlock, img_channels, num_classes)
#**********************************************************************************************************************#

########################################################################################################################
########################################## SEPARABLE CONVOLUTION FOR JPU CLASS #########################################
########################################################################################################################
class SeparableConv2d(nn.Module):
    def __init__(
            self, inplanes, planes, kernel_size=3,
            stride=1, padding=1, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(
            inplanes, inplanes, kernel_size,
            stride, padding, dilation, groups=inplanes, bias=bias)
        self.bn = nn.BatchNorm2d(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x
#**********************************************************************************************************************#


########################################################################################################################
################################################# JPU FOR FASTFCN CLASS ################################################
########################################################################################################################
class JPU(nn.Module):
    """
    Joint Pyramid Upsampling Module proposed in:
    H. Wu et al., FastFCN: Rethinking Dilated Convolution in the Backbone for Semantic Segmentation
    https://arxiv.org/abs/1903.11816
    """

    def __init__(self, in_channels, width=512):
        super(JPU, self).__init__()
        """
        Args:
            in channels: tuple. in ascending order
        """

        self.convs = []
        self.dilations = []

        self.conv0 = nn.Sequential(
            nn.Conv2d(
                in_channels[0], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True)
        )
        self.dilation0 = nn.Sequential(
            SeparableConv2d(
                3 * width, width, kernel_size=3,
                padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True)
        )
        self.convs.append(self.conv0)
        self.dilations.append(self.dilation0)

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels[1], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True)
        )
        self.dilation1 = nn.Sequential(
            SeparableConv2d(
                3 * width, width, kernel_size=3,
                padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True)
        )
        self.convs.append(self.conv1)
        self.dilations.append(self.dilation1)

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels[2], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True)
        )
        self.dilation2 = nn.Sequential(
            SeparableConv2d(
                3 * width, width, kernel_size=3,
                padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True)
        )

        self.convs.append(self.conv2)
        self.dilations.append(self.dilation2)

        self.dilation3 = nn.Sequential(
            SeparableConv2d(
                3 * width, width, kernel_size=3,
                padding=8, dilation=8, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True)
        )

        self.dilations.append(self.dilation3)

    def forward(self, *inputs):
        """
        Args:
            inputs: tuple. in order from high resolution feature to low resolution feature
        """
        feats = []

        for input, conv in zip(inputs, self.convs):
            feats.append(conv(input))

        _, _, h, w = feats[0].shape

        for i in range(1, len(feats)):
            feats[i] = F.interpolate(
                feats[i], size=(h, w), mode='bilinear', align_corners=True
            )

        feat = torch.cat(feats, dim=1)

        outputs = []

        for dilation in self.dilations:
            outputs.append(
                dilation(feat)
            )

        outputs = torch.cat(outputs, dim=1)

        return outputs
#**********************************************************************************************************************#


class FastFCN(nn.Module):
    """
        Fast FCN with VGG backbone.
        Originally, Fast FCN has ResNet101 as a backbone.
        Please refer to:
            H. Wu et al., FastFCN: Rethinking Dilated Convolution in the Backbone for Semantic Segmentation
            https://arxiv.org/abs/1903.11816
    """

    def __init__(self, in_channels, num_classes, jpu_in_channels=(512, 1024, 2048), width=512):
        super().__init__()

        self.backbone = ResNet152(img_channels=in_channels, num_classes=num_classes)

        self.jpu = JPU(jpu_in_channels, width)

        self.Conv_end1 = nn.Conv2d(
            (len(jpu_in_channels)+1) * width, 256, 3, 1, 1, bias=False)
        self.bn_end1 = nn.BatchNorm2d(256)
        self.conv_end2 = nn.Conv2d(
            in_channels=256, out_channels=num_classes, kernel_size=1, stride=1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        _, _, h, w = x.shape

        x2, x3, x4 = self.backbone(x)

        x = self.jpu(x2, x3, x4)
        # print("X size after JPU: ", x.size())
        x = self.Conv_end1(x)
        # print("X size after first output convolution: ", x.size())
        x = self.bn_end1(x)
        x = self.conv_end2(x)
        # print("X size after second output convolution: ", x.size())
        result = F.interpolate(
            x, size=(h, w), mode='bilinear', align_corners=True)

        return self.dropout(result)
    
# random_data = torch.rand((4, 6, 128, 128))

# fastfcn = FastFCN(in_channels=6, num_classes=6)

# result = fastfcn(random_data)
# print(result.size())