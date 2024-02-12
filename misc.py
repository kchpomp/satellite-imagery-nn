import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from init_weights import init_weights
import torchvision.transforms.functional as TF


#==================== Function that allows to print a progress bar ====================#
#------------------------- Parameters: ------------------------------------------------#
#----- iteration: Required - current iteration                                    -----#
#----- total: Required - total iterations                                         -----#
#----- prefix: Optional - prefix string                                           -----#
#----- suffix: Optional - suffix string                                           -----#
#----- decimals: Optional - positive number of decimals in percent complete       -----#
#----- length: Optional - character length of bar                                 -----#
#----- fill: Optional - bar fill character                                        -----#
#--------------------------------------------------------------------------------------# 

def PrintProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='$'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)

    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    
    if iteration == total:
        print()  
#--------------------------------------------------------------------------------------#


#----------------------------------------------------------------------------------------------------------------------#
#-------------------------------------- BLOCK OF CLASSES NEEDED FOR U-NET MODIFIED MODELS -----------------------------#
#----------------------------------------------------------------------------------------------------------------------#


########################################################################################################################
######################################### CLASS FOR DOWNSAMPLE CONVOLUTION BLOCK #######################################
########################################################################################################################
class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch, dilation=1):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False, dilation=dilation),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.6),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False, dilation=dilation),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.6)
            )

    def forward(self, x):
        x = self.conv(x)
        return x
#**********************************************************************************************************************#


########################################################################################################################
####################################### CLASS FOR 3D DOWNSAMPLE CONVOLUTION BLOCK ######################################
########################################################################################################################
class unetConv2exp(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        super(unetConv2exp, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_batchnorm:
            for i in range(1, n+1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.BatchNorm2d(out_size),
                                     nn.ReLU(inplace=True),)
                setattr(self, 'conv%d'%i, conv)
                in_size = out_size

        else:
            for i in range(1, n+1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.ReLU(inplace=True),)
                setattr(self, 'conv%d'%i, conv)
                in_size = out_size

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n+1):
            conv = getattr(self, 'conv%d'%i)
            x = conv(x)

        return x
#**********************************************************************************************************************#


########################################################################################################################
########################################## CLASS FOR UPSAMPLE CONVOLUTION BLOCK ########################################
########################################################################################################################
class up_conv(nn.Module):
    def __init__(self, in_ch, out_ch, dilation=1):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False, dilation=dilation),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.6) # was 0.45
        )

    def forward(self, x):
        x = self.up(x)
        return x
#**********************************************************************************************************************#


########################################################################################################################
######################################## CLASS FOR 2D UPSAMPLE CONVOLUTION BLOCK #######################################
########################################################################################################################
class UpsampleCustom(nn.Module):
    """ nn.Upsample is deprecated """

    def __init__(self, scale_factor, mode="bilinear"):
        super(UpsampleCustom, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x

class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv):
        super(unetUp, self).__init__()
        self.conv = unetConv2(in_size, out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1)
        else:
            self.up = UpsampleCustom(scale_factor=2)

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('unetConv2') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2]
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))
#**********************************************************************************************************************#


########################################################################################################################
########################################## CLASS U-NET DEEP SUPERVISION BLOCK ##########################################
########################################################################################################################
class UnetDsv(nn.Module):
    def __init__(self, in_size, out_size, scale_factor):
        super(UnetDsv, self).__init__()
        self.dsv = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0),
                                 UpsampleCustom(scale_factor=scale_factor, mode='bilinear'), )

    def forward(self, input):
        return self.dsv(input)
#**********************************************************************************************************************#


########################################################################################################################
############################################### CLASS FOR 2D GATING SIGNAL #############################################
########################################################################################################################
class UnetGridGatingSignal2(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, is_batchnorm=True):
        super(UnetGridGatingSignal2, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size, 1, 0),
                                       nn.BatchNorm2d(out_size),
                                       nn.ReLU(inplace=True),
                                       )
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size, 1, 0),
                                       nn.ReLU(inplace=True),
                                       )

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        return outputs
#**********************************************************************************************************************#


########################################################################################################################
########################################## CLASS FOR 2D GRID ATTENTION BLOCK ###########################################
########################################################################################################################
class _GridAttentionBlockND(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels=None, dimension=3, mode='concatenation',
                 sub_sample_factor=(2,2,2)):
        super(_GridAttentionBlockND, self).__init__()

        assert dimension in [2, 3]
        assert mode in ['concatenation', 'concatenation_debug', 'concatenation_residual']

        # Downsampling rate for the input featuremap
        if isinstance(sub_sample_factor, tuple): self.sub_sample_factor = sub_sample_factor
        elif isinstance(sub_sample_factor, list): self.sub_sample_factor = tuple(sub_sample_factor)
        else: self.sub_sample_factor = tuple([sub_sample_factor]) * dimension

        # Default parameter set
        self.mode = mode
        self.dimension = dimension
        self.sub_sample_kernel_size = self.sub_sample_factor

        # Number of channels (pixel dimensions)
        self.in_channels = in_channels
        self.gating_channels = gating_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            bn = nn.BatchNorm3d
            self.upsample_mode = 'trilinear'
        elif dimension == 2:
            conv_nd = nn.Conv2d
            bn = nn.BatchNorm2d
            self.upsample_mode = 'bilinear'
        else:
            raise NotImplemented

        # Output transform
        self.W = nn.Sequential(
            conv_nd(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0),
            bn(self.in_channels),
        )

        # Theta^T * x_ij + Phi^T * gating_signal + bias
        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=self.sub_sample_kernel_size, stride=self.sub_sample_factor, padding=0, bias=False)
        self.phi = conv_nd(in_channels=self.gating_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0, bias=True)
        self.psi = conv_nd(in_channels=self.inter_channels, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

        # Initialise weights
        for m in self.children():
            init_weights(m, init_type='kaiming')

        # Define the operation
        if mode == 'concatenation':
            self.operation_function = self._concatenation
        elif mode == 'concatenation_debug':
            self.operation_function = self._concatenation_debug
        elif mode == 'concatenation_residual':
            self.operation_function = self._concatenation_residual
        else:
            raise NotImplementedError('Unknown operation function.')


    def forward(self, x, g):
        '''
        :param x: (b, c, t, h, w)
        :param g: (b, g_d)
        :return:
        '''

        output = self.operation_function(x, g)
        return output

    def _concatenation(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        # phi   => (b, g_d) -> (b, i_c)
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
        #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, thw) -> (b, i_c, t/s1, h/s2, w/s3)
        phi_g = F.interpolate(self.phi(g), size=theta_x_size[2:], mode=self.upsample_mode)
        f = F.relu(theta_x + phi_g, inplace=True)

        #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
        sigm_psi_f = F.sigmoid(self.psi(f))

        # upsample the attentions and multiply
        sigm_psi_f = F.interpolate(sigm_psi_f, size=input_size[2:], mode=self.upsample_mode)
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)

        return W_y, sigm_psi_f

    def _concatenation_debug(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        # phi   => (b, g_d) -> (b, i_c)
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
        #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, thw) -> (b, i_c, t/s1, h/s2, w/s3)
        phi_g = F.interpolate(self.phi(g), size=theta_x_size[2:], mode=self.upsample_mode)
        f = F.softplus(theta_x + phi_g)

        #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
        sigm_psi_f = F.sigmoid(self.psi(f))

        # upsample the attentions and multiply
        sigm_psi_f = F.interpolate(sigm_psi_f, size=input_size[2:], mode=self.upsample_mode)
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)

        return W_y, sigm_psi_f


    def _concatenation_residual(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        # phi   => (b, g_d) -> (b, i_c)
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
        #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, thw) -> (b, i_c, t/s1, h/s2, w/s3)
        phi_g = F.interpolate(self.phi(g), size=theta_x_size[2:], mode=self.upsample_mode)
        f = F.relu(theta_x + phi_g, inplace=True)

        #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
        f = self.psi(f).view(batch_size, 1, -1)
        sigm_psi_f = F.softmax(f, dim=2).view(batch_size, 1, *theta_x.size()[2:])

        # upsample the attentions and multiply
        sigm_psi_f = F.interpolate(sigm_psi_f, size=input_size[2:], mode=self.upsample_mode)
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)

        return W_y, sigm_psi_f


class GridAttentionBlock2D(_GridAttentionBlockND):
    def __init__(self, in_channels, gating_channels, inter_channels=None, mode='concatenation',
                 sub_sample_factor=(2,2,2)):
        super(GridAttentionBlock2D, self).__init__(in_channels,
                                                   inter_channels=inter_channels,
                                                   gating_channels=gating_channels,
                                                   dimension=2, mode=mode,
                                                   sub_sample_factor=sub_sample_factor,
                                                   )
#**********************************************************************************************************************#


########################################################################################################################
############################################### MULTI ATTENTION BLOCK CLASS ############################################
########################################################################################################################
class MultiAttentionBlock(nn.Module):
    def __init__(self, in_size, gating_channels, inter_channels, mode, sub_sample_factor):
        super(MultiAttentionBlock, self).__init__()
        self.gate_block_1 = GridAttentionBlock2D(in_channels=in_size, gating_channels=gating_channels,
                                                 inter_channels=inter_channels, mode=mode,
                                                 sub_sample_factor= sub_sample_factor)
        self.combine_gates = nn.Sequential(nn.Conv2d(in_size, in_size, kernel_size=1, stride=1, padding=0),
                                           nn.BatchNorm2d(in_size),
                                           nn.ReLU(inplace=True)
                                           )

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('GridAttentionBlock2D') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, input, gating_signal):
        gate_1, attention_1 = self.gate_block_1(input, gating_signal)

        return self.combine_gates(gate_1), attention_1

#**********************************************************************************************************************#


########################################################################################################################
################################################## ATTENTION MODULE CLASS ##############################################
########################################################################################################################
class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        # print("F_g", F_g)
        # print("F_g type", type(F_g))
        # print("F_g", F_l)
        # print("F_g type", type(F_l))
        # print("F_int", F_int)
        # print("F_int type", type(F_int))

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = psi * x 
        return out
#**********************************************************************************************************************#


########################################################################################################################
################################################### RECURRENT BLOCK ####################################################
########################################################################################################################
class Recurrent_block(nn.Module):
    def __init__(self, ch_out, t=2):
        super(Recurrent_block, self).__init__()
        self.t = t
        self.ch_out = ch_out

        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.1)
        )

    def forward(self, x):
        for i in range(self.t):

            if i == 0:
                x1 = self.conv(x)

            x1 = self.conv(x + x1)
        return x1
#**********************************************************************************************************************#


########################################################################################################################
################################################### RRCNN BLOCK ########################################################
########################################################################################################################
class RRCNN_block(nn.Module):
    '''
    Updated on 21.02.2023 according to https://github.com/LeeJunHyun/Image_Segmentation/issues/73
    '''
    def __init__(self,ch_in,ch_out,t=2):
        super(RRCNN_block,self).__init__()

    #     # self.Conv_1x1 = nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=1,padding=0)
        self.Conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.1)
        )
        # self.activate = nn.ReLU(inplace=True)


    # def forward(self,x):
    #     a1 = self.Conv(x)
    #     a2 = self.RCNN(x)
    #     out = self.activate(a1 + a2)
    #     return out
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out,t=t),
            Recurrent_block(ch_out,t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        x = self.Conv(x)
        x1 = self.RCNN(x)
        out = x+x1
        # out = self.activate(x1+x2)
        # out = self.activate(x + x1)
        return out


class single_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(single_conv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

    def forward(self,x):
        x = self.conv(x)
        return
#**********************************************************************************************************************#


########################################################################################################################
################################################### UNET2+ BLOCK #######################################################
########################################################################################################################
class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        super(unetConv2, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        
        if is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.BatchNorm2d(out_size),
                                     nn.ReLU(inplace=True))
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size
        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                      nn.ReLU(inplace=True))
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')


    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)
        return x
#**********************************************************************************************************************#


########################################################################################################################
################################################### UNET2+ BLOCK #######################################################
########################################################################################################################
class unetUp_origin(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, n_concat=2):
        super(unetUp_origin, self).__init__()
        # self.conv = unetConv2(out_size*2, out_size, False)
        if is_deconv:
            self.conv = unetConv2(in_size + (n_concat - 2) * out_size, out_size, False)
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1)
        else:
            self.conv = unetConv2(in_size + (n_concat - 2) * out_size, out_size, False)
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('unetConv2') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs0, *input):
        # print(self.n_concat)
        # print(input)
        outputs0 = self.up(inputs0)
        for i in range(len(input)):
            outputs0 = torch.cat([outputs0, input[i]], 1)
        return self.conv(outputs0)
#**********************************************************************************************************************#


########################################################################################################################
########################################### NESTED DENSE CONVOLUTIONAL BLOCK ###########################################
########################################################################################################################
class NestedBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        return out
#**********************************************************************************************************************#


########################################################################################################################
############################################## U-NET3+ CONVOLUTIONAL BLOCK #############################################
########################################################################################################################
class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        super(unetConv2, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.BatchNorm2d(out_size),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)

        return x
#**********************************************************************************************************************#


########################################################################################################################
########################################### U-NET 3D DEEP SUPER VISION BLOCK ###########################################
########################################################################################################################
class UnetDsv3(nn.Module):
    def __init__(self, in_size, out_size, scale_factor):
        super(UnetDsv3, self).__init__()
        self.dsv = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size=1, stride=1, padding=0),
                                 nn.Upsample(scale_factor=scale_factor, mode='trilinear'), )

    def forward(self, input):
        return self.dsv(input)
#**********************************************************************************************************************#


########################################################################################################################
########################################### DICE LOSS FOR SEGMENTATION TASKS ###########################################
########################################################################################################################
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice
#**********************************************************************************************************************#


########################################################################################################################
########################################### FUNCTION TO GET METRICS FROM LOG ###########################################
########################################################################################################################
def get_metric(epoch_range, metrics, mode, metric):
    values = []
    for i in range(epoch_range):
        base = metrics[i][f'epoch_{i}']
        if metric == 'loss':
            value = base['loss']
        else:
            value = base['metrics'][f'{mode}_{metric}']
        values.append(value)
        
    return values
#**********************************************************************************************************************#


########################################################################################################################
############################################ FUNCTION TO BUILD METRICS PLOT ############################################
########################################################################################################################
def build_metric_plot(epoch_range, train_logs, val_logs, metrics_name, model_type):
        sns.set_style("dark")
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.plot(range(1, epoch_range + 1), 
        get_metric(epoch_range, train_logs, 'train', metrics_name), label=f'Training {metrics_name}', 
        color="red", linewidth=3)
        ax.plot(range(1, epoch_range + 1), 
        get_metric(epoch_range, val_logs, 'val', metrics_name), label=f'Validation {metrics_name}', 
        color="orange", linewidth=3)
        ax.set_title(model_type, fontsize = 18)
        ax.set_ylabel(f"{metrics_name}", fontsize=15)
        ax.set_xlabel("Epochs", fontsize=15)
        plt.legend()
        plt.savefig(model_type + f"_{metrics_name}_graph.png")
#**********************************************************************************************************************#


########################################################################################################################
########################################### VANILLA U-NET DOUBLE CONV BLOCK ############################################
########################################################################################################################
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation = 1):
        super(DoubleConv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation,stride=1, bias=False,
                     dilation = dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding = dilation, stride = 1, bias=False,
                     dilation = dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
#**********************************************************************************************************************#


########################################################################################################################
############################################### CLASS FOR  TVERSKY LOSS ################################################
########################################################################################################################
class TverskyLoss(nn.Module):
    """Tversky Loss.

    .. seealso::
        Salehi, Seyed Sadegh Mohseni, Deniz Erdogmus, and Ali Gholipour. "Tversky loss function for image segmentation
        using 3D fully convolutional deep networks." International Workshop on Machine Learning in Medical Imaging.
        Springer, Cham, 2017.

    Args:
        alpha (float): Weight of false positive voxels.
        beta  (float): Weight of false negative voxels.
        smooth (float): Epsilon to avoid division by zero, when both Numerator and Denominator of Tversky are zeros.

    Attributes:
        alpha (float): Weight of false positive voxels.
        beta  (float): Weight of false negative voxels.
        smooth (float): Epsilon to avoid division by zero, when both Numerator and Denominator of Tversky are zeros.

    Notes:
        - setting alpha=beta=0.5: Equivalent to DiceLoss.
        - default parameters were suggested by https://arxiv.org/pdf/1706.05721.pdf .
    """

    def __init__(self, alpha=0.7, beta=0.3, smooth=1.0):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth


    def tversky_index(self, y_pred, y_true):
        """Compute Tversky index.

        Args:
            y_pred (torch Tensor): Prediction.
            y_true (torch Tensor): Target.

        Returns:
            float: Tversky index.
        """
        # Compute TP
        y_true = y_true.float()
        tp = torch.sum(y_true * y_pred)
        # Compute FN
        fn = torch.sum(y_true * (1 - y_pred))
        # Compute FP
        fp = torch.sum((1 - y_true) * y_pred)
        # Compute Tversky for the current class, see Equation 3 of the original paper
        numerator = tp + self.smooth
        denominator = tp + self.alpha * fp + self.beta * fn + self.smooth
        tversky_label = numerator / denominator
        return tversky_label


    def forward(self, input, target):
        n_classes = input.shape[1]
        tversky_sum = 0.
        

        # TODO: Add class_of_interest?
        for i_label in range(n_classes):
            # Get samples for a given class
            y_pred, y_true = input[:, i_label, ], target[:, i_label, ]
            # Compute Tversky index
            tversky_sum += self.tversky_index(y_pred, y_true)

        return - tversky_sum / n_classes
#**********************************************************************************************************************#


########################################################################################################################
############################################### CLASS FOR  TVERSKY LOSS ################################################
########################################################################################################################
class FocalTverskyLoss(TverskyLoss):
    """Focal Tversky Loss.

    .. seealso::
        Abraham, Nabila, and Naimul Mefraz Khan. "A novel focal tversky loss function with improved attention u-net for
        lesion segmentation." 2019 IEEE 16th International Symposium on Biomedical Imaging (ISBI 2019). IEEE, 2019.

    Args:
        alpha (float): Weight of false positive voxels.
        beta  (float): Weight of false negative voxels.
        gamma (float): Typically between 1 and 3. Control between easy background and hard ROI training examples.
        smooth (float): Epsilon to avoid division by zero, when both Numerator and Denominator of Tversky are zeros.

    Attributes:
        gamma (float): Typically between 1 and 3. Control between easy background and hard ROI training examples.

    Notes:
        - setting alpha=beta=0.5 and gamma=1: Equivalent to DiceLoss.
        - default parameters were suggested by https://arxiv.org/pdf/1810.07842.pdf .
    """

    def __init__(self, alpha=0.7, beta=0.3, gamma=1.33, smooth=1.0):
        super(FocalTverskyLoss, self).__init__()
        self.gamma = gamma
        self.tversky = TverskyLoss(alpha=alpha, beta=beta, smooth=smooth)


    def forward(self, input, target):
        n_classes = input.shape[1]
        focal_tversky_sum = 0.
        input = input.cpu()
        target = target.cpu()
        true_1_hot = torch.eye(n_classes)[target.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(input, dim=1)
        true_1_hot = true_1_hot.type(target.type())
        dims = (0,) + tuple(range(2, target.ndimension()))
        # tversky_index = self.tversky.tversky_index(probas, true_1_hot)

        # TODO: Add class_of_interest?
        for i_label in range(n_classes):
            # Get samples for a given class
            # y_pred, y_true = input[:, i_label, ], target[:, i_label, ]
            # Compute Tversky index
            tversky_index = self.tversky.tversky_index(probas, true_1_hot)
            # Compute Focal Tversky loss, Equation 4 in the original paper
            focal_tversky_sum += torch.pow(1 - tversky_index, exponent=1 / self.gamma)

        return focal_tversky_sum / n_classes
#**********************************************************************************************************************#


def tversky_loss_new(true, logits, alpha=0.7, beta=0.3, gamma=1.33, eps=1e-7):
    """Computes the Tversky loss [1].
    Args:
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        alpha: controls the penalty for false positives.
        beta: controls the penalty for false negatives.
        eps: added to the denominator for numerical stability.
    Returns:
        tversky_loss: the Tversky loss.
    Notes:
        alpha = beta = 0.5 => dice coeff
        alpha = beta = 1 => tanimoto coeff
        alpha + beta = 1 => F beta coeff
    References:
        [1]: https://arxiv.org/abs/1706.05721
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    fps = torch.sum(probas * (1 - true_1_hot), dims)
    fns = torch.sum((1 - probas) * true_1_hot, dims)
    num = intersection
    denom = intersection + (alpha * fps) + (beta * fns)
    tversky_loss = (num / (denom + eps)).mean()
    return torch.pow((1 - tversky_loss), exponent=1/ gamma)