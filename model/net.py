import numpy as np

import torch
import torch.nn as nn
import torch.functional as F
from collections import OrderedDict

from model.modules import *


def factory(type, in_channels):
    if type == 'level_0':
        return CMD240x240(in_channels, True)
    elif type == 'level_1':
        return CMD120x120(in_channels=in_channels, bn=True)
    elif type == 'level_2':
        return CMD60x60(in_channels=in_channels, bn=True)
    elif type == 'level_3':
        return CMDTop(in_channels=in_channels, bn=True)
    elif type == 'level_4':
        return CMDTop(in_channels=in_channels, bn=True)
    assert 0, 'Correspondence Map Decoder bad creation: ' + type



def MutualMatching(corr4d):
    # mutual matching
    batch_size,ch,fs1,fs2,fs3,fs4 = corr4d.size()

    corr4d_B=corr4d.view(batch_size,fs1*fs2,fs3,fs4) # [batch_idx,k_A,i_B,j_B]
    corr4d_A=corr4d.view(batch_size,fs1,fs2,fs3*fs4)

    # get max
    corr4d_B_max,_=torch.max(corr4d_B,dim=1,keepdim=True)
    corr4d_A_max,_=torch.max(corr4d_A,dim=3,keepdim=True)

    eps = 1e-5
    corr4d_B=corr4d_B/(corr4d_B_max+eps)
    corr4d_A=corr4d_A/(corr4d_A_max+eps)

    corr4d_B=corr4d_B.view(batch_size,1,fs1,fs2,fs3,fs4)
    corr4d_A=corr4d_A.view(batch_size,1,fs1,fs2,fs3,fs4)

    corr4d=corr4d*(corr4d_A*corr4d_B) # parenthesis are important for symmetric output 
        
    return corr4d


class DGCNet(nn.Module):
    """
    Original DGC-net model
    """
    def __init__(self, mask=False):
        super(DGCNet, self).__init__()

        self.mask = mask

        self.pyramid = VGGPyramid()
        # L2 feature normalisation
        self.l2norm = FeatureL2Norm()
        # Correlation volume
        self.corr = CorrelationVolume()

        if self.mask:
            self.matchability_net = MatchabilityNet(in_channels=128, bn=True)

        # create a hierarchy of correspondence decoders
        map_dim = 2
        N_out = [x + map_dim for x in [128, 128, 256, 512, 225]]

        for i, in_chan in enumerate(N_out):
            self.__dict__['_modules']['reg_' + str(i)] = factory('level_' + str(i), in_chan) 


    def forward(self, x1, x2):
        """
        x1 - target image
        x2 - source image
        """

        target_pyr = self.pyramid(x1)
        source_pyr = self.pyramid(x2)

        # do feature normalisation
        feat_top_pyr_trg = self.l2norm(target_pyr[-1])
        feat_top_pyr_src = self.l2norm(source_pyr[-1])

        # do correlation
        corr1 = self.corr(feat_top_pyr_trg, feat_top_pyr_src)
        corr1 = self.l2norm(F.relu(corr1))

        b, c, h, w = corr1.size()
        init_map = torch.FloatTensor(b, 2, h, w).zero_().cuda()
        est_grid = self.__dict__['_modules']['reg_4'](x1=corr1, x3=init_map)

        estimates_grid = [est_grid]

        '''
        create correspondence map decoder, upsampler for each level of
        the feature pyramid
        '''
        for k in reversed(range(4)):
            p1, p2 = target_pyr[k], source_pyr[k]
            est_map = F.interpolate(input=estimates_grid[-1], scale_factor=2, mode='bilinear', align_corners=False)

            p1_w = F.grid_sample(p1, est_map.transpose(1,2).transpose(2,3))
            est_map = self.__dict__['_modules']['reg_' + str(k)](x1=p1_w, x2=p2, x3=est_map)
            estimates_grid.append(est_map)

        # matchability mask
        matchability = None
        if self.mask:
            matchability = self.matchability_net(x1=p1_w, x2=p2)

        return estimates_grid, matchability


class DGCNCCCNet(nn.Module):
    """
    Modified DGC-Net model with only one decoder proposed in the report/paper
    """
    def __init__(self,
                use_cuda=True,
                ncons_kernel_sizes=[3,3,3],
                ncons_channels=[10,10,1]
                 ):
        super(DGCNCCCNet, self).__init__()
        # Feature pyramid
        self.pyramid = VGG16Pyramid()
        
        # L2 feature normalization
        self.featureL2Norm = FeatureL2Norm()
        
        # Correlation volume computation
        self.FeatureCorrelation = FeatureCorrelation(shape='4D',normalization=False)

        self.NeighConsensus = NeighConsensus(use_cuda=use_cuda,
                                             kernel_sizes=ncons_kernel_sizes,
                                             channels=ncons_channels)

        # Convolutional Upsampler
        self.upsampler = Upsampler()

        flow_dim = 2
        N_out = [x + flow_dim for x in [128, 128, 256, 512, 225]] # Too bad to be true
        self.__dict__['_modules']['reg_4'] = FlowRegressor_Top2Levels(N_out[4], batch_norm=True)
        self.__dict__['_modules']['reg_0'] = CCDecoder(N_out[0], batch_norm=True)

    def forward(self, x1, x2):
        """
        Forward pass of the pyramid-based affine transformation assessment.
        
        x1 - unlatered image
        x2 - transformed image
        
        """
        pyramid_out1 = self.pyramid(x1)
        pyramid_out2 = self.pyramid(x2)
        p1_top = pyramid_out1[-1]
        p2_top = pyramid_out2[-1]

        # do feature normalization 
        feature_top_pyr1 = self.featureL2Norm(pyramid_out1[-1])
        feature_top_pyr2 = self.featureL2Norm(pyramid_out2[-1])

        # feature correlation
        corr4d = self.FeatureCorrelation(feature_top_pyr1,feature_top_pyr2)
        
        # run match processing model 
        corr4d = MutualMatching(corr4d)
        corr4d = self.NeighConsensus(corr4d)
        corr4d = MutualMatching(corr4d)

        # reshape new to old shape
        batch_size = corr4d.size(0)
        feature_size = corr4d.size(2)
        correlation_B_Avec = corr4d.view(batch_size,
                                         feature_size * feature_size,
                                         feature_size,
                                         feature_size)
        correlation_A_Bvec = corr4d.view(batch_size,
                                         feature_size,
                                         feature_size,
                                         feature_size * feature_size).permute(0, 3, 1, 2)

        estimates_grid_B_A = self.flow_est_opt(correlation_B_Avec, pyramid_out1, pyramid_out2)
        estimates_grid_A_B = self.flow_est_opt(correlation_A_Bvec, pyramid_out2, pyramid_out1)

        return estimates_grid_B_A, estimates_grid_A_B

    def flow_est_opt(self, correlation, pyramid_out1, pyramid_out2):
        # do positive matches normalization
        correlation = self.featureL2Norm(F.relu(correlation))

        b, c, h, w = correlation.size()
        init_flow = Variable(torch.FloatTensor(b, 2, h, w).zero_()).cuda()
        est_grid = self.__dict__['_modules']['reg_4'](x1=correlation, init_flow=init_flow)

        estimates_grid = [est_grid]

        for i in range(3, -1, -1):
            p1, p2 = pyramid_out1[i], pyramid_out2[i]
            est_flow = self.upsampler(estimates_grid[-1])
            p1_w = F.grid_sample(p1, est_flow.transpose(1,2).transpose(2,3))
            est_flow = self.__dict__['_modules']['reg_0'](x1=p1_w, x2=p2, init_flow=est_flow)
            estimates_grid.append(est_flow)

        return estimates_grid