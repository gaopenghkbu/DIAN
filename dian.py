import torch
import torch.nn as nn
from torch.nn import init
from resnet import resnet50, resnet18
import torch.nn.functional as F
import numpy as np
from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2dPack as DCN


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out
# #####################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)

class visible_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(visible_module, self).__init__()

        model_v = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.visible = model_v

    def forward(self, x):
        x = self.visible.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)
        return x

class thermal_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(thermal_module, self).__init__()

        model_t = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.thermal = model_t

    def forward(self, x):
        x = self.thermal.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)
        return x

class base_resnet(nn.Module):
    def __init__(self, arch='resnet50'):
        super(base_resnet, self).__init__()

        model_base = resnet50(pretrained=True,
                              last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = model_base

    def forward(self, x):
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        return x
    
class OrthogonalFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.01)
                init.zeros_(m.bias.data)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
                init.zeros_(m.bias.data)
            elif isinstance(m, nn.BatchNorm1d):
                init.normal_(m.weight.data, 1.0, 0.01)
                init.zeros_(m.bias.data)
                
    def forward(self, local_feat, global_feat):
        #print('global_feat', global_feat.shape)
        #print(global_feat.shape)
        if len(global_feat.shape) == 1:
            global_feat = global_feat.unsqueeze(0)
        #print(global_feat.shape)
        global_feat_norm = torch.norm(global_feat, p=2, dim=1)
        projection = torch.bmm(global_feat.unsqueeze(1), torch.flatten(
            local_feat, start_dim=2))
        
        projection = torch.bmm(global_feat.unsqueeze(
            2), projection).view(local_feat.size())
        
        projection = projection / \
            (global_feat_norm * global_feat_norm).view(-1, 1, 1, 1)
        orthogonal_comp = local_feat - projection
        global_feat = global_feat.unsqueeze(-1).unsqueeze(-1)
        return torch.cat([global_feat.expand(orthogonal_comp.size()), orthogonal_comp], dim=1)
 
class MultiScale(nn.Module):
    def __init__(self, in_channel, out_channel, dilation_rates=[3, 5, 7, 9], factor=2):
        super().__init__()
        self.dilated_convs = [
            nn.Conv2d(in_channel, int(out_channel/4),
                      kernel_size=3, dilation=rate, padding=rate)
            for rate in dilation_rates
        ]
        
        # self.gap_branch = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv2d(in_channel, int(out_channel/4), kernel_size=1),
        #     nn.ReLU(),
        #     nn.Upsample(size=(size, size), mode='bilinear')
        # )
        if factor == 2:
            self.up = nn.AvgPool2d((2,2))
        else:
            self.up = nn.Identity()
        # self.dilated_convs.append(self.gap_branch)
        self.dilated_convs = nn.ModuleList(self.dilated_convs)
        self.init_weights()
        
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.01)
                init.zeros_(m.bias.data)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
                init.zeros_(m.bias.data)
            elif isinstance(m, nn.BatchNorm1d):
                init.normal_(m.weight.data, 1.0, 0.01)
                init.zeros_(m.bias.data)
                
    def forward(self, x):
        local_feat = []
        for dilated_conv in self.dilated_convs:
            local_feat.append(self.up(dilated_conv(x)))
            
        local_feat = torch.cat(local_feat, dim=1)
        return local_feat
 
class LocalBranch(nn.Module):
    def __init__(self, in_channel, out_channel, hidden_channel, factor):
        super().__init__()
        #size = 512/8 = 64
        self.multi_atrous = MultiScale(in_channel=in_channel, out_channel=hidden_channel, factor=factor)
        self.conv1x1_1 = nn.Conv2d(hidden_channel, out_channel, kernel_size=1)
        
        self.conv1x1_2 = nn.Conv2d(out_channel, out_channel, kernel_size=1, bias=False)
        self.conv1x1_3 = nn.Conv2d(out_channel, out_channel, kernel_size=1)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channel)
        self.softplus = nn.Softplus()
        self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.01)
                init.zeros_(m.bias.data)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
                init.zeros_(m.bias.data)
            elif isinstance(m, nn.BatchNorm1d):
                init.normal_(m.weight.data, 1.0, 0.01)
                init.zeros_(m.bias.data)
                
    def forward(self, x):
        #concat后的local branch
        local_feat = self.multi_atrous(x)
        #一共用三个卷积去干这个事情
        local_feat = self.conv1x1_1(local_feat)
        local_feat = self.relu(local_feat)
        
        local_feat = self.conv1x1_2(local_feat)
        local_feat = self.bn(local_feat)

        attention_map = self.relu(local_feat)
        attention_map = self.conv1x1_3(attention_map)
        attention_map = self.softplus(attention_map)

        local_feat = F.normalize(local_feat, p=2, dim=1)
        local_feat = local_feat * attention_map

        return local_feat       

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, requires_grad=False):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p, requires_grad=requires_grad)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

class DolgNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, factor=2):
        super().__init__()
     
        self.orthogonal_fusion = OrthogonalFusion()
        self.local_branch = LocalBranch(input_dim, output_dim,hidden_dim,factor=factor)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gem_pool = GeM()
        self.fc_1 = nn.Linear(output_dim, output_dim)
        self.fc_2 = nn.Linear(int(2*output_dim), output_dim)
        self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.01)
                init.zeros_(m.bias.data)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
                init.zeros_(m.bias.data)
            elif isinstance(m, nn.BatchNorm1d):
                init.normal_(m.weight.data, 1.0, 0.01)
                init.zeros_(m.bias.data)
                
                
    def forward(self, x_low, x_high):
        #output = self.cnn(x)

        # local_feat = self.local_branch(output[0])  # ,hidden_channel,16,16
        # global_feat = self.fc_1(self.gem_pool(output[1]).squeeze())  # ,1024
        #print('x_low',x_low.shape)
        local_feat = self.local_branch(x_low)
        #print('local_feat===', local_feat.shape)
        
        global_feat = self.fc_1(self.gem_pool(x_high).squeeze())
        #print('global_feat===', global_feat.shape)
        feat = self.orthogonal_fusion(local_feat, global_feat)
        feat = self.gap(feat).squeeze()
        feat = self.fc_2(feat).unsqueeze(-1).unsqueeze(-1)
        
        return feat * x_high + x_high
    
class Decouple(nn.Module):
    def __init__(self, in_x_channels, kernel_size):
        super(Decouple, self).__init__()
        self.unfold = nn.Unfold(kernel_size=3, dilation=kernel_size, padding=kernel_size, stride=1)
        self.fuse = nn.Conv2d(in_x_channels, in_x_channels, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.01)
                init.zeros_(m.bias.data)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
                init.zeros_(m.bias.data)
            elif isinstance(m, nn.BatchNorm1d):
                init.normal_(m.weight.data, 1.0, 0.01)
                init.zeros_(m.bias.data)
                
    def forward(self, x, y):
        N, xC, xH, xW = x.size()
        kernel   = y.reshape([N, xC, 3 ** 2, xH, xW])
        unfold_x = self.unfold(x).reshape([N, xC, -1, xH, xW])
        out = (unfold_x * kernel).sum(2)
        out = self.lrelu(self.fuse(out))
        return out
    
class Dynamic_kernel(nn.Module):
    def __init__(self, in_channels) -> None:
        super().__init__()
        self.psa = ParallelAttention(channel=in_channels)
        self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.01)
                init.zeros_(m.bias.data)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
                init.zeros_(m.bias.data)
            elif isinstance(m, nn.BatchNorm1d):
                init.normal_(m.weight.data, 1.0, 0.01)
                init.zeros_(m.bias.data)
                
    def forward(self, x):
        x = self.psa(x)
        return x
    
class ParallelAttention(nn.Module):

    def __init__(self, channel=512):
        super().__init__()
        self.ch_wv=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.ch_wq=nn.Conv2d(channel,1,kernel_size=(1,1))
        self.softmax_channel=nn.Softmax(1)
        self.softmax_spatial=nn.Softmax(-1)
        self.ch_wz=nn.Conv2d(channel//2,channel,kernel_size=(1,1))
        self.ln=nn.LayerNorm(channel)
        self.sigmoid=nn.Sigmoid()
        self.sp_wv=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.sp_wq=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.agp=nn.AdaptiveAvgPool2d((1,1))
        self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.01)
                init.zeros_(m.bias.data)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
                init.zeros_(m.bias.data)
            elif isinstance(m, nn.BatchNorm1d):
                init.normal_(m.weight.data, 1.0, 0.01)
                init.zeros_(m.bias.data)
                
    def forward(self, x):
        b, c, h, w = x.size()

        #Channel-only Self-Attention
        channel_wv=self.ch_wv(x) #bs,c//2,h,w
        channel_wq=self.ch_wq(x) #bs,1,h,w
        channel_wv=channel_wv.reshape(b,c//2,-1) #bs,c//2,h*w
        channel_wq=channel_wq.reshape(b,-1,1) #bs,h*w,1
        channel_wq=self.softmax_channel(channel_wq)
        channel_wz=torch.matmul(channel_wv,channel_wq).unsqueeze(-1) #bs,c//2,1,1
        channel_weight=self.sigmoid(self.ln(self.ch_wz(channel_wz).reshape(b,c,1).permute(0,2,1))).permute(0,2,1).reshape(b,c,1,1) #bs,c,1,1
        channel_out=channel_weight*x + x

        #Spatial-only Self-Attention
        spatial_wv=self.sp_wv(x) #bs,c//2,h,w
        spatial_wq=self.sp_wq(x) #bs,c//2,h,w
        spatial_wq=self.agp(spatial_wq) #bs,c//2,1,1
        spatial_wv=spatial_wv.reshape(b,c//2,-1) #bs,c//2,h*w
        spatial_wq=spatial_wq.permute(0,2,3,1).reshape(b,1,c//2) #bs,1,c//2
        spatial_wq=self.softmax_spatial(spatial_wq)
        spatial_wz=torch.matmul(spatial_wq,spatial_wv) #bs,1,h*w
        spatial_weight=self.sigmoid(spatial_wz.reshape(b,1,h,w)) #bs,1,h,w
        spatial_out=spatial_weight*x + x
        out=spatial_out+channel_out
        return out
    
class IEDK2(nn.Module):
    def __init__(self, in_channels, mid_channels, dataset): # [32, 16, 32] [64, 32, 32] [128, 64, 32] [256, 128, 32]
        super().__init__()
        self.in_channels = in_channels
        self.dataset = dataset
        self.mid_channels = mid_channels
        if self.in_channels != self.mid_channels:
            self.conv = nn.Conv2d(in_channels, mid_channels, 1, 1, 0)
        # self.offset   = OffsetBlock(in_channels, offset_channels)
        self.generate_kernel = nn.Sequential(DCN(in_channels, in_channels, 3, stride=1, padding=1, dilation=1, deformable_groups=8),
                           nn.LeakyReLU(negative_slope=0.2, inplace=True),
                           Dynamic_kernel(in_channels),
                           nn.Conv2d(in_channels, mid_channels * 3 **2, 1, 1, 0)
        )

        # self.enchance0 = Branch_enhance(channel=1024, factor=1)
        self.enchance1 = Branch_enhance(channel=in_channels)
        self.enchance3 = Branch_enhance(channel=in_channels)
        self.enchance5 = Branch_enhance(channel=in_channels)
        
        self.lrelu    = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.branch_1 = Decouple(mid_channels, kernel_size=1)
        self.branch_3 = Decouple(mid_channels, kernel_size=3)
        self.branch_5 = Decouple(mid_channels, kernel_size=5)
        # self.fusion   = nn.Conv2d(mid_channels*4, in_channels, 1, 1, 0)
        # self.attention = GlobalAttention(in_channels=in_channels*4)
        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.01)
                init.zeros_(m.bias.data)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
                init.zeros_(m.bias.data)
            elif isinstance(m, nn.BatchNorm1d):
                init.normal_(m.weight.data, 1.0, 0.01)
                init.zeros_(m.bias.data)
                
    def forward(self, x):
        # x0 = self.enchance0(x)
        x0 = x
        # x_offset = self.offset(x)  
        y = self.generate_kernel(x)

        # print(y)
        if self.in_channels != self.mid_channels:
            x  = self.conv(x)  #  通道维度进行压缩，低维度通道压缩为一半，高维度保持不变
            
        if self.dataset == 'sysu':
            x1 = self.enchance1(self.branch_1(x, y))
            x3 = self.enchance3(self.branch_3(x, y))
            x5 = self.enchance5(self.branch_5(x, y))
        else:
            x1 = self.branch_1(x,y)
            x3 = self.branch_3(x,y)
            x5 = self.branch_5(x,y)
            
        
        out = torch.cat((x0, x1, x3, x5), dim=0)
        return out
    
class Branch_enhance(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(in_channels=channel,out_channels=channel, kernel_size=3, groups=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=channel,out_channels=channel, kernel_size=3, groups=2, padding=1)
        )

        self.transform = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1)
        )
        
        self.act = nn.LeakyReLU(negative_slope=0.2)

        self.ch_wv=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.ch_wq=nn.Conv2d(channel,1,kernel_size=(1,1))
        self.softmax_channel=nn.Softmax(1)
        self.softmax_spatial=nn.Softmax(-1)
        self.ch_wz=nn.Conv2d(channel//2,channel,kernel_size=(1,1))
        self.ln=nn.LayerNorm(channel)
        self.sigmoid=nn.Sigmoid()
        self.sp_wv=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.sp_wq=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.agp=nn.AdaptiveAvgPool2d((1,1))
        self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.01)
                init.zeros_(m.bias.data)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
                init.zeros_(m.bias.data)
            elif isinstance(m, nn.BatchNorm1d):
                init.normal_(m.weight.data, 1.0, 0.01)
                init.zeros_(m.bias.data)

    def forward(self,x):
        b, c, h, w = x.size()
        x0 = x
        x = self.shared(x)
        
        channel_wv=self.ch_wv(x) #bs,c//2,h,w
        channel_wq=self.ch_wq(x) #bs,1,h,w
        channel_wv=channel_wv.reshape(b,c//2,-1) #bs,c//2,h*w
        channel_wq=channel_wq.reshape(b,-1,1) #bs,h*w,1
        channel_wq=self.softmax_channel(channel_wq)
        channel_wz=torch.matmul(channel_wv,channel_wq).unsqueeze(-1) #bs,c//2,1,1
        channel_weight=self.ch_wz(channel_wz).reshape(b,c,1).permute(0,2,1).permute(0,2,1).reshape(b,c,1,1) #bs,c,1,1
        channel_out=channel_weight*x
        
        #Spatial-only Self-Attention
        spatial_wv=self.sp_wv(x) #bs,c//2,h,w
        spatial_wq=self.sp_wq(x) #bs,c//2,h,w
        spatial_wq=self.agp(spatial_wq) #bs,c//2,1,1
        spatial_wv=spatial_wv.reshape(b,c//2,-1) #bs,c//2,h*w
        spatial_wq=spatial_wq.permute(0,2,3,1).reshape(b,1,c//2) #bs,1,c//2
        spatial_wq=self.softmax_spatial(spatial_wq)
        spatial_wz=torch.matmul(spatial_wq,spatial_wv) #bs,1,h*w
        spatial_weight=spatial_wz.reshape(b,1,h,w) #bs,1,h,w
        spatial_out=spatial_weight*x
        out=spatial_out+channel_out
        
        #transform
        out = self.transform(out)
        out = self.act(out+x)
        final_out = x0+out
        return final_out
    
class embed_net(nn.Module):
    def __init__(self,  class_num, dataset, arch='resnet50'):
        super(embed_net, self).__init__()

        self.thermal_module = thermal_module(arch=arch)
        self.visible_module = visible_module(arch=arch)

        self.base_resnet = base_resnet(arch=arch)
        
        self.dataset = dataset
        if self.dataset == 'regdb': # For regdb dataset, we remove the MFA3 block and layer4.
            pool_dim = 2048
            self.DK = Dynamic_kernel(in_channels=1024, mid_channels=1024, dataset='regdb')
        else:
            pool_dim = 2048
        # self.enhance_visible = RCBpaper(n_feat=1024)
        # self.enhance_thermal = RCBpaper(n_feat=1024)
            self.DEE = IEDK2(in_channels=1024, mid_channels=1024, dataset='sysu')
            self.MFA0 = DolgNet(input_dim=64, hidden_dim=16, output_dim=256, factor=1)
            self.MFA1 = DolgNet(input_dim=256, hidden_dim=64, output_dim=512, factor=2)
            self.MFA2 = DolgNet(input_dim=512, hidden_dim=128, output_dim=1024, factor=2)
        #self.MFA3 = DolgNet(input_dim=1024, hidden_dim=256, output_dim=2048, factor=1)
        ###
        # self.MFA1 = AdaMScaleBlock(in_channel=256, out_channel=256)
        # self.MFA2 = AdaMScaleBlock(in_channel=512, out_channel=512)
        # self.MFA3 = AdaMScaleBlock(in_channel=1024, out_channel=1024)
        # self.MFA4 = AdaMScaleBlock(in_channel=2048, out_channel=2048)

        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier = nn.Linear(pool_dim, class_num, bias=False)
        self.classifier.apply(weights_init_classifier)
        
        self.l2norm = Normalize(2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x1, x2, modal=0):
        if modal == 0:
            x1 = self.visible_module(x1)
            x2 = self.thermal_module(x2)
           
            x = torch.cat((x1, x2), 0)
        elif modal == 1:
            x = self.visible_module(x1)
           
        elif modal == 2:
            x = self.thermal_module(x2)
            


  
        if self.dataset=='sysu':
            x_ = x
            x = self.base_resnet.base.layer1(x_)
            x_ = self.MFA0(x_low=x_, x_high=x)
            x = self.base_resnet.base.layer2(x_)
            x_ = self.MFA1(x_low=x_, x_high=x)

            x = self.base_resnet.base.layer3(x_)
            x_ = self.MFA2(x_low=x_, x_high=x)

            x_ = self.DK(x_)
            x = self.base_resnet.base.layer4(x_)       
            
        else:
            x = self.base_resnet.base.layer1(x)
            x = self.base_resnet.base.layer2(x)
            x = self.base_resnet.base.layer3(x)
            #x = self.DEE(x)
            x = self.base_resnet.base.layer4(x)
            x = torch.cat((x,x,x,x), dim=0)
        
        

        
        xp = self.avgpool(x)
        x_pool = xp.view(xp.size(0), xp.size(1))
        
        feat = self.bottleneck(x_pool)

        if self.training:
            # xps = xp.view(xp.size(0), xp.size(1), xp.size(2)).permute(0, 2, 1)
            # xp1, xp2, xp3 = torch.chunk(xps, 3, 0)
            # xpss = torch.cat((xp2, xp3), 1)
            # loss_ort = torch.triu(torch.bmm(xpss, xpss.permute(0, 2, 1)), diagonal = 1).sum() / (xp.size(0))
            # print('x_pool.shape', x_pool.shape)
            return x_pool, self.classifier(feat), 0
        else:
            #print('x_pool.shape', x_pool.shape)
            return self.l2norm(x_pool), self.l2norm(feat)
        