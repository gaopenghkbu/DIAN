import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
from torch.autograd import Variable

class PLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(PLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=0.2)

    def forward(self, inputs, targets):
        ft1, ft2, ft3, ft4 = torch.chunk(inputs, 4, 0)
        lb1, lb2, lb3, lb4 = torch.chunk(targets, 4, 0)
        
        lb_num = len(lb1.unique())    
        lbs = lb1.unique() 

        n = lbs.size(0)   

        ft1 = ft1.chunk(lb_num, 0)
        ft2 = ft2.chunk(lb_num, 0)
        ft3 = ft3.chunk(lb_num, 0)
        ft4 = ft4.chunk(lb_num, 0)
        for i in ft1:
            print('iiiii',i.shape)
        center1 = []
        center2 = []
        center3 = []
        center4 = []
        for i in range(lb_num):
            center1.append(torch.mean(ft1[i], dim=0, keepdim=True))
            center2.append(torch.mean(ft2[i], dim=0, keepdim=True))
            center3.append(torch.mean(ft3[i], dim=0, keepdim=True))
            center4.append(torch.mean(ft4[i], dim=0, keepdim=True))

        ft1 = torch.cat(center1)
        ft2 = torch.cat(center2)
        ft3 = torch.cat(center3)
        ft4 = torch.cat(center4)
        
        print('ft1',ft1.shape) #ft1 torch.Size([6, 2048])

        dist_13 = pdist_torch(ft1, ft3)
        dist_23 = pdist_torch(ft2, ft3)
        dist_33 = pdist_torch(ft3, ft3)
        dist_11 = pdist_torch(ft1, ft1)

        dist_14 = pdist_torch(ft1, ft4)
        dist_24 = pdist_torch(ft2, ft4)
        dist_44 = pdist_torch(ft4, ft4)
        dist_22 = pdist_torch(ft2, ft2)

        mask = lbs.expand(n, n).eq(lbs.expand(n, n).t())
        
        dist_ap_123, dist_an_123, dist_ap_124, dist_an_124, dist_an_33, dist_an_44, dist_an_11, dist_an_22 = [], [], [], [], [], [], [], []
        for i in range(n):
            dist_ap_123.append(dist_23[i][mask[i]].max().unsqueeze(0))
            dist_an_123.append(dist_13[i][mask[i]].min().unsqueeze(0))
            dist_an_33.append(dist_33[i][mask[i] == 0].min().unsqueeze(0))
            dist_an_11.append(dist_11[i][mask[i] == 0].min().unsqueeze(0))

            dist_ap_124.append(dist_14[i][mask[i]].max().unsqueeze(0))
            dist_an_124.append(dist_24[i][mask[i]].min().unsqueeze(0))
            dist_an_44.append(dist_44[i][mask[i] == 0].min().unsqueeze(0))
            dist_an_22.append(dist_22[i][mask[i] == 0].min().unsqueeze(0))

        dist_ap_123 = torch.cat(dist_ap_123)
        dist_an_123 = torch.cat(dist_an_123).detach()
        dist_an_33 = torch.cat(dist_an_33)
        dist_an_11 = torch.cat(dist_an_11)

        dist_ap_124 = torch.cat(dist_ap_124)
        dist_an_124 = torch.cat(dist_an_124).detach()
        dist_an_44 = torch.cat(dist_an_44)
        dist_an_22 = torch.cat(dist_an_22)

        loss_123 = self.ranking_loss(dist_an_123, dist_ap_123, torch.ones_like(dist_an_123)) + (self.ranking_loss(dist_an_33, dist_ap_123, torch.ones_like(dist_an_33)) + self.ranking_loss(dist_an_11, dist_ap_123, torch.ones_like(dist_an_33))) * 0.5
        loss_124 = self.ranking_loss(dist_an_124, dist_ap_124, torch.ones_like(dist_an_124)) + (self.ranking_loss(dist_an_44, dist_ap_124, torch.ones_like(dist_an_44)) + self.ranking_loss(dist_an_22, dist_ap_124, torch.ones_like(dist_an_44))) * 0.5
        return (loss_123 + loss_124)/2

class CPMLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(CPMLoss, self).__init__()
        self.margin = margin
        
        #看y代表的意思
        #ln=max(0,−y∗(x1−x2)+margin)
        

        self.ranking_loss = nn.MarginRankingLoss(margin=0.2)

    def forward(self, inputs, targets):
        # print('inputs---', inputs.shape)
        # print('targets---', targets.shape)
        ft1, ft2, ft3, ft4 = torch.chunk(inputs, 4, 0)
        lb1, lb2, lb3, lb4 = torch.chunk(targets, 4, 0)
        
        lb_num = len(lb1.unique())    
        lbs = lb1.unique() 
        #print('lb_num', lb_num)
        n = lbs.size(0)   
        #print('n===',n)
        #ft1 v1
        ft1 = ft1.chunk(lb_num, 0)
        
        #ft2 v2
        ft2 = ft2.chunk(lb_num, 0)
        
        #ft3 v3
        ft3 = ft3.chunk(lb_num, 0)
        
        #ft4 v4
        ft4 = ft4.chunk(lb_num, 0)
        
        
        center1 = []
        center2 = []
        center3 = []
        center4 = []
        for i in range(lb_num):
            center1.append(torch.mean(ft1[i], dim=0, keepdim=True))
            center2.append(torch.mean(ft2[i], dim=0, keepdim=True))
            center3.append(torch.mean(ft3[i], dim=0, keepdim=True))
            center4.append(torch.mean(ft4[i], dim=0, keepdim=True))

        ft1 = torch.cat(center1)
        ft2 = torch.cat(center2)
        ft3 = torch.cat(center3)
        ft4 = torch.cat(center4)

        # V跟V+
        dist_13 = pdist_torch(ft1, ft3)
        
        # N跟V+
        dist_23 = pdist_torch(ft2, ft3)
        
        # V+跟V+
        dist_33 = pdist_torch(ft3, ft3)
        
        # V跟N
        dist_12 = pdist_torch(ft1, ft2)
        
        #V跟V
        dist_11 = pdist_torch(ft1, ft1)

        #V跟N+
        dist_14 = pdist_torch(ft1, ft4)
        
        #N跟N+
        dist_24 = pdist_torch(ft2, ft4)
        
        #N+跟N+
        dist_44 = pdist_torch(ft4, ft4)
        
        #N跟N
        dist_22 = pdist_torch(ft2, ft2)
        
        #V+跟N+
        dist_34 = pdist_torch(ft3, ft4)

        mask = lbs.expand(n, n).eq(lbs.expand(n, n).t())
        
        #dist_ap_123, dist_an_123, dist_ap_124, dist_an_124, dist_an_33, dist_an_44, dist_an_11, dist_an_22, dist_an_13, dist_an_24, dist_an_12, dist_an_14, dist_an_23, dist_an_34 = [], [], [], [], [], [], [], [], [], [], [], [], [], []
        #dist_an_V_Vp, dist_an_N_Np = [], []
        dist_ap_v_vp, dist_ap_v_np, dist_ap_n_np, dist_ap_n_vp= [], [], [], []
        dist_an_vv, dist_an_vn, dist_an_vvp, dist_an_vnp, dist_an_nn, dist_an_nvp, dist_an_nnp, dist_an_vpvp, dist_an_vpnp, dist_an_npnp = [], [], [], [], [], [], [], [], [], []
        for i in range(n):
            '''
            #N跟V+ id相同中距离最大的
            dist_ap_123.append(dist_23[i][mask[i]].max().unsqueeze(0))
            
            ### V跟V+ id相同距离最小的
            #### dist_an_123.append(dist_13[i][mask[i]].min().unsqueeze(0))
            dist_an_123.append(dist_13[i][mask[i]].min().unsqueeze(0))
            #V跟N+ id相同距离最大的
            dist_ap_124.append(dist_14[i][mask[i]].max().unsqueeze(0))
            
            # N跟N+ id相同距离最小的
            dist_an_124.append(dist_24[i][mask[i]].min().unsqueeze(0))
            '''
            #V跟V+距离最大的
            dist_ap_v_vp.append(dist_13[i][mask[i]].max().unsqueeze(0))
            
            #N跟N+距离最大的
            dist_ap_n_np.append(dist_24[i][mask[i]].max().unsqueeze(0))
            
            
            #V跟N+距离最大的// V跟N+id相同距离最小
            dist_ap_v_np.append(dist_14[i][mask[i]].min().unsqueeze(0))
            
            #N跟V+距离最大的 // N跟V+id相同距离最小的
            dist_ap_n_vp.append(dist_23[i][mask[i]].min().unsqueeze(0))
            
            
            #V跟V中 id不同距离最小的
            dist_an_vv.append(dist_11[i][mask[i] == 0].min().unsqueeze(0))
            
            #V跟N中 id不同距离最小的
            dist_an_vn.append(dist_12[i][mask[i] == 0].min().unsqueeze(0))
            
            #V跟V+ id不同距离最小
            dist_an_vvp.append(dist_13[i][mask[i] == 0].min().unsqueeze(0))
            
            #V跟N+ id不同距离最小
            dist_an_vnp.append(dist_14[i][mask[i] == 0].min().unsqueeze(0))
            
            #N跟N id不同距离最小的
            dist_an_nn.append(dist_22[i][mask[i] == 0].min().unsqueeze(0))
            
            #N跟V+ id不同距离最小
            dist_an_nvp.append(dist_23[i][mask[i] == 0].min().unsqueeze(0))
            
            #N跟N+ id不同距离最小的
            dist_an_nnp.append(dist_24[i][mask[i] == 0].min().unsqueeze(0))
            
            #V+跟V+ id不同距离最小的
            dist_an_vpvp.append(dist_33[i][mask[i] == 0].min().unsqueeze(0))
            
            #N+跟v+ id不同距离最小的
            dist_an_vpnp.append(dist_34[i][mask[i] == 0].min().unsqueeze(0))
            
            #N+跟N+ id不同距离最小的
            dist_an_npnp.append(dist_44[i][mask[i] == 0].min().unsqueeze(0))
            
            
        #论文中把N跟V+相同id拉近， V跟V+相同id拉远   
        dist_ap_v_vp = torch.cat(dist_ap_v_vp)
        dist_ap_n_np = torch.cat(dist_ap_n_np)
        dist_ap_v_np = torch.cat(dist_ap_v_np).detach()
        dist_ap_n_vp = torch.cat(dist_ap_n_vp).detach()
        # dist_an_123 = torch.cat(dist_an_123).detach()
        # dist_an_33 = torch.cat(dist_an_33)
        # dist_an_11 = torch.cat(dist_an_11)
        dist_an_vv = torch.cat(dist_an_vv)
        dist_an_vn = torch.cat(dist_an_vn)
        dist_an_vvp = torch.cat(dist_an_vvp)
        dist_an_vnp = torch.cat(dist_an_vnp)
        
        dist_an_nn = torch.cat(dist_an_nn)
        dist_an_nvp = torch.cat(dist_an_nvp)
        dist_an_nnp = torch.cat(dist_an_nnp)
        
        dist_an_vpvp = torch.cat(dist_an_vpvp)
        dist_an_vpnp = torch.cat(dist_an_vpnp)
        
        dist_an_npnp = torch.cat(dist_an_npnp)
        # dist_an_124 = torch.cat(dist_an_124).detach()
        # dist_an_44 = torch.cat(dist_an_44)
        # dist_an_22 = torch.cat(dist_an_22)

        #当y=1，x1>x2
        #V跟V+ id相同的距离要大于 N跟V+id相同                                                    V+跟V+id不同近距离最小的 大于 N跟V+id相同                                                    V跟Vid不同 大于 N跟V+id相同
        #loss_123 = self.ranking_loss(dist_an_123, dist_ap_123, torch.ones_like(dist_an_123)) + (self.ranking_loss(dist_an_33, dist_ap_123, torch.ones_like(dist_an_33)) + self.ranking_loss(dist_an_11, dist_ap_123, torch.ones_like(dist_an_33))) * 0.5
        #N跟N+ id相同的距离要大于 V跟N+id相同                                                    N+跟N+id不同近距离最小的 大于 V跟N+id相同                                                    N跟Nid不同 大于 V跟N+id相同
        #loss_124 = self.ranking_loss(dist_an_124, dist_ap_124, torch.ones_like(dist_an_124)) + (self.ranking_loss(dist_an_44, dist_ap_124, torch.ones_like(dist_an_44)) + self.ranking_loss(dist_an_22, dist_ap_124, torch.ones_like(dist_an_44))) * 0.5
        
        #N跟V+id相同距离最小要大于， V跟V+id相同距离最大
        loss123 = self.ranking_loss(dist_ap_n_vp, dist_ap_v_vp, torch.ones_like(dist_ap_v_vp)) + (self.ranking_loss(dist_an_vpvp, dist_ap_v_vp, torch.ones_like(dist_an_vpvp)) + self.ranking_loss(dist_an_vv, dist_ap_v_vp, torch.ones_like(dist_an_vv))) * 0.5
        #V跟N+id相同距离最小要大于， N跟N+id相同距离最大
        loss124 = self.ranking_loss(dist_ap_n_vp, dist_ap_n_np, torch.ones_like(dist_ap_n_np)) + (self.ranking_loss(dist_an_npnp, dist_ap_n_np, torch.ones_like(dist_an_npnp)) + self.ranking_loss(dist_an_nn, dist_ap_n_np, torch.ones_like(dist_an_nn))) * 0.5
        
        #这是第一种方案效果不行
        # loss_v_vp = self.ranking_loss(dist_an_vvp, dist_ap_v_vp, torch.ones_like(dist_ap_v_vp)) + 0.5 * (self.ranking_loss(dist_an_nvp, dist_ap_v_vp, torch.ones_like(dist_ap_v_vp)) + self.ranking_loss(dist_an_vpvp, dist_ap_v_vp, torch.ones_like(dist_ap_v_vp)) + self.ranking_loss(dist_an_vpnp, dist_ap_v_vp, torch.ones_like(dist_ap_v_vp)))\
        #         #    +self.ranking_loss(dist_an_23, dist_ap_123, torch.ones_like(dist_ap_123)) \
        #         #    +self.ranking_loss(dist_an_33, dist_ap_123, torch.ones_like(dist_ap_123)) + self.ranking_loss(dist_an_34, dist_ap_123, torch.ones_like(dist_ap_123))
                   
        # loss_n_np = self.ranking_loss(dist_an_nnp, dist_ap_n_np, torch.ones_like(dist_ap_n_np)) + 0.5 * (self.ranking_loss(dist_an_npnp, dist_ap_n_np, torch.ones_like(dist_ap_n_np)) + self.ranking_loss(dist_an_vnp, dist_ap_n_np, torch.ones_like(dist_ap_n_np)) + self.ranking_loss(dist_an_vpnp, dist_ap_n_np, torch.ones_like(dist_ap_n_np)))
                   
        # loss_v_np = self.ranking_loss(dist_an_nnp, dist_ap_v_np, torch.ones_like(dist_ap_v_np)) + 0.5 * (self.ranking_loss(dist_an_vnp, dist_ap_v_np, torch.ones_like(dist_ap_v_np)) + self.ranking_loss(dist_an_npnp, dist_ap_v_np, torch.ones_like(dist_ap_v_np)) + self.ranking_loss(dist_an_vpnp, dist_ap_v_np, torch.ones_like(dist_ap_v_np)))
        
        # loss_n_vp = self.ranking_loss(dist_an_vvp, dist_ap_n_vp, torch.ones_like(dist_ap_n_vp)) + 0.5 * (self.ranking_loss(dist_an_nvp, dist_ap_n_vp, torch.ones_like(dist_ap_n_vp)) + self.ranking_loss(dist_an_vpnp, dist_ap_n_vp, torch.ones_like(dist_ap_n_vp)) + self.ranking_loss(dist_an_vpvp, dist_ap_n_vp, torch.ones_like(dist_ap_n_vp)))
        # return (loss_123 + loss_124)/2
        
        #尝试第二种方案
        #N跟V+id相同距离最小要大于， V跟V+id相同距离最大
        loss123 = self.ranking_loss(dist_ap_n_vp, dist_ap_v_vp, torch.ones_like(dist_ap_v_vp)) + (self.ranking_loss(dist_an_vpvp, dist_ap_v_vp, torch.ones_like(dist_an_vpvp)) + self.ranking_loss(dist_an_vv, dist_ap_v_vp, torch.ones_like(dist_an_vv))) * 0.5
        #V跟N+id相同距离最小要大于， N跟N+id相同距离最大
        loss124 = self.ranking_loss(dist_ap_n_vp, dist_ap_n_np, torch.ones_like(dist_ap_n_np)) + (self.ranking_loss(dist_an_npnp, dist_ap_n_np, torch.ones_like(dist_an_npnp)) + self.ranking_loss(dist_an_nn, dist_ap_n_np, torch.ones_like(dist_an_nn))) * 0.5
        return (loss123+loss124)/2

class OriTripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    
    Args:
    - margin (float): margin for triplet.
    """
    
    def __init__(self, batch_size, margin=0.3):
        super(OriTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        
        # compute accuracy
        correct = torch.ge(dist_an, dist_ap).sum().item()
        return loss

class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature).cuda())			# 超参数 温度
        
        #主对角线是0， 其余位置是1
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 4, batch_size * 4, dtype=bool).cuda()).float())		# 主对角线为0，其余位置全为1的mask矩阵

    
    def forward(self, emb_i, emb_j):
        z_i = F.normalize(emb_i, dim=1)     # (bs, dim)  --->  (bs, dim)
        z_j = F.normalize(emb_j, dim=1) 
        
        representations = torch.cat([z_i, z_j], dim=0)          # repre: (2*bs, dim)
        #print(representations.unsqueeze(1).shape) # 48 1 2048
        #print(representations.unsqueeze(0).shape) # 1 48 2048
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)      # simi_mat: (2*bs, 2*bs)
        
        #print(similarity_matrix)
        #print()
        sim_ij_bs1 = torch.diag(similarity_matrix, self.batch_size)         # bs
        #print('sim_ij', sim_ij)
        sim_ji_bs1 = torch.diag(similarity_matrix, -self.batch_size)        # bs
        
        sim_ij_bs2 = torch.diag(similarity_matrix,  2*self.batch_size)
        sim_ji_bs2 = torch.diag(similarity_matrix, -2*self.batch_size)
        
        sim_ij_bs3 = torch.diag(similarity_matrix,  3*self.batch_size)
        sim_ji_bs3 = torch.diag(similarity_matrix, -3*self.batch_size)
        #print()
        #print('sim_ji', sim_ji)
        positives1 = torch.cat([sim_ij_bs1, sim_ji_bs1], dim=0)                  # 2*bs
        positives2 = torch.cat([sim_ij_bs2, sim_ji_bs2], dim=0) 
        positives3 = torch.cat([sim_ij_bs3, sim_ji_bs3], dim=0) 
        
        nominator1 = torch.exp(positives1 / self.temperature)             # 2*bs
        nominator2 = torch.exp(positives2 / self.temperature)
        nominator3 = torch.exp(positives3 / self.temperature)
        # print('nominator1',nominator1.shape)
        # print('nominator2',nominator2.shape)
        # print('nominator3',nominator3.shape)
        # print()
        
        # print('negatives_mask',self.negatives_mask.shape)
        # print('similarity', similarity_matrix.shape)
        
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)             # 2*bs, 2*bs
        
        #print('denominator', denominator.shape)
        # loss_partial1 = -torch.log(nominator1 / torch.sum(denominator))
        # loss_partial2 = -torch.log(nominator2 / torch.sum(denominator, dim=1))
        # loss_partial3 = -torch.log(nominator3 / torch.sum(denominator, dim=1))
        nominator = torch.sum(nominator1) + torch.sum(nominator2) + torch.sum(nominator3)
        denominator = torch.sum(denominator) - nominator
        loss_partial = -torch.log(nominator/denominator)
        loss = torch.sum(loss_partial) / (4 * self.batch_size)
        return loss


class Frequency_ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.loss = ContrastiveLoss(batch_size=batch_size, temperature=temperature) 
        #self.loss03 = ContrastiveLoss(batch_size=batch_size, temperature=temperature)
        #self.loss05 = ContrastiveLoss(batch_size=batch_size, temperature=temperature)
    def forward(self, feat1, labels):
        ft_v, ft_n, ft_v1, ft_n1 = torch.chunk(feat1, 4, 0)
        lb_v, lb_n, lb_v1, lb_n1 = torch.chunk(labels, 4, 0) 
        lb_num = len(lb_v.unique())
        lbs = lb_v.unique()
        
        ft_v = ft_v.chunk(lb_num, 0)
        ft_n = ft_n.chunk(lb_num, 0)
        ft_v1 = ft_v1.chunk(lb_num, 0)
        ft_n1 = ft_n1.chunk(lb_num, 0)
        center_v = []
        center_n = []
        center_v1 = []
        center_n1 = []
    
        for i in range(lb_num):
            center_v.append(torch.mean(ft_v[i], dim=0, keepdim=True))
            center_n.append(torch.mean(ft_n[i], dim=0, keepdim=True))
            center_v1.append(torch.mean(ft_v1[i], dim=0, keepdim=True))
            center_n1.append(torch.mean(ft_n1[i], dim=0, keepdim=True))
            
        ft_v = torch.cat(center_v) #6 2048
        ft_n = torch.cat(center_n) #6 2048
        ft_v1 = torch.cat(center_v1) #6 2048
        ft_n1 = torch.cat(center_n1) #6 2048
      
        emb1 = torch.cat((ft_v, ft_n), dim=0) #24 2048
        emb2 = torch.cat((ft_v1, ft_n1), dim=0)
        return self.loss(emb1, emb2)
        
# Adaptive weights
def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6 # avoid division by zero
    W = torch.exp(diff) * mask / Z
    return W

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

class TripletLoss_WRT(nn.Module):
    """Weighted Regularized Triplet'."""

    def __init__(self):
        super(TripletLoss_WRT, self).__init__()
        self.ranking_loss = nn.SoftMarginLoss()

    def forward(self, inputs, targets, normalize_feature=False):
        if normalize_feature:
            inputs = normalize(inputs, axis=-1)
        dist_mat = pdist_torch(inputs, inputs)

        N = dist_mat.size(0)
        # shape [N, N]
        is_pos = targets.expand(N, N).eq(targets.expand(N, N).t()).float()
        is_neg = targets.expand(N, N).ne(targets.expand(N, N).t()).float()

        # `dist_ap` means distance(anchor, positive)
        # both `dist_ap` and `relative_p_inds` with shape [N, 1]
        dist_ap = dist_mat * is_pos
        dist_an = dist_mat * is_neg

        weights_ap = softmax_weights(dist_ap, is_pos)
        weights_an = softmax_weights(-dist_an, is_neg)
        furthest_positive = torch.sum(dist_ap * weights_ap, dim=1)
        closest_negative = torch.sum(dist_an * weights_an, dim=1)

        y = furthest_positive.new().resize_as_(furthest_positive).fill_(1)
        loss = self.ranking_loss(closest_negative - furthest_positive, y)


        # compute accuracy
        correct = torch.ge(closest_negative, furthest_positive).sum().item()
        return loss, correct
        
def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim = 1, keepdim = True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim = 1, keepdim = True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    # dist_mtx = dist_mtx.clamp(min = 1e-12)
    dist_mtx = dist_mtx.clamp(min = 1e-12).sqrt()
    return dist_mtx    


def pdist_np(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using cpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = np.square(emb1).sum(axis = 1)[..., np.newaxis]
    emb2_pow = np.square(emb2).sum(axis = 1)[np.newaxis, ...]
    dist_mtx = -2 * np.matmul(emb1, emb2.T) + emb1_pow + emb2_pow
    # dist_mtx = np.sqrt(dist_mtx.clip(min = 1e-12))
    return dist_mtx