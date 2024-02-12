import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, num_layers, block, image_channels):
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

        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * self.expansion, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # print("ResNet101 Input: ", x.size())
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # print("Input to first Block: ", x.size())
        x = self.layer1(x)
        low_level_features = self.dropout(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.dropout(x)

        # x = self.avgpool(x)
        # x = x.reshape(x.shape[0], -1)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)
        return x, low_level_features

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
###################################### ATROUS SPATIAL PYRAMID POOLING CLASS ############################################
########################################################################################################################

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)
    

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256, rates=[6, 12, 18, 24]):
        super(ASPP, self).__init__()

        # self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        self.conv_input = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.aspp_conv_1 = ASPPConv(in_channels, out_channels, rates[0])

        self.aspp_conv_2 = ASPPConv(in_channels, out_channels, rates[1])

        self.aspp_conv_3 = ASPPConv(in_channels, out_channels, rates[2])

        self.aspp_conv_4 = ASPPConv(in_channels, out_channels, rates[3])

        self.aspp_pooling = ASPPPooling(in_channels, out_channels)

        self.out_conv = nn.Sequential(
            nn.Conv2d(6 * out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
            )

    def forward(self, x):

        x1 = self.conv_input(x)
        # print("Input to ASPP and ASPP Pooling size: ", x.size())
        x2 = self.aspp_conv_1(x)
        x3 = self.aspp_conv_2(x)
        x4 = self.aspp_conv_3(x)
        x5 = self.aspp_conv_4(x)
        x6 = self.aspp_pooling(x)

        out = torch.cat([x1, x2, x3, x4, x5, x6], dim=1)
        out = self.out_conv(out)
        # print("ASPP Output Size: ", out.size())
        return out
#**********************************************************************************************************************#


########################################################################################################################
####################################### DECODING BLOCK FOR DEEPLABV3+ CLASS ############################################
########################################################################################################################
class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, upsample_factor=8):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nn.ReLU()

        self.up = nn.Upsample(scale_factor=upsample_factor, mode='bilinear', align_corners=True)

        self.conv2 = nn.Conv2d(
#             in_channels // 4 + out_channels
            320
            , out_channels
            , 3
            , padding=1
            , bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()

    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu1(low_level_feat)
        x = self.up(x)
        # print("Decoder X size after upsampling", x.size())
        x = torch.cat((x, low_level_feat), dim=1)
        # print("Decoder X size after concatenation of X and LLF: ", x.size())
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        # print("Decoder Output size: ", x.size())
        return x
#**********************************************************************************************************************#


########################################################################################################################
########################################### DEEPLAB V3 PLUS CLASS ######################################################
########################################################################################################################
class DeepLabV3Plus(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(DeepLabV3Plus, self).__init__()

        # Define ResNet101 backbone
        self.backbone = ResNet101(img_channels=in_channels, num_classes=num_classes)
        # self.backbone = ResNet50(img_channels=in_channels, num_classes=num_classes)
        # self.backbone = ResNet152(img_channels=in_channels, num_classes=num_classes)
        
        # Define ASPP module
        self.aspp = ASPP(in_channels=2048)
        self.dropout = nn.Dropout(0.5)

        # Define Decoder module
#         self.decoder = Decoder(in_channels=256, out_channels=48, atrous_rates=[12, 24, 36])
        self.decoder = Decoder(in_channels = 256, out_channels = 256)


        # Define final convolutional layer
        self.final_conv = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):

        x, low_level_features = self.backbone(x)

        aspp_out = self.aspp(x)

        decoder_out = self.decoder(aspp_out, low_level_features)

        x = F.interpolate(decoder_out, size=[128, 128], mode='bilinear', align_corners=True)
        x = self.final_conv(x)
        return self.dropout(x)
#**********************************************************************************************************************#


# random_data = torch.rand((4, 6, 128, 128))

# deep_lab = DeepLabV3Plus(6, 6)

# result = deep_lab(random_data)
# print(result.size())