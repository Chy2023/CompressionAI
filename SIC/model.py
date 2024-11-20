from compressai.models import Cheng2020Anchor
from compressai.layers import (
    MaskedConv2d,
    conv1x1,
    ResidualBlock,
    ResidualBlockUpsample,
    subpel_conv3x3,
)
from compressai.entropy_models import GaussianConditional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import warnings
from compressai.ans import BufferedRansEncoder, RansDecoder

import os
import sys
sys.path.append(".")
sys.path.append("./YOLOv3")
from YOLOv3.pytorchyolo.models import load_model


class Choi2022(Cheng2020Anchor):
    '''
    Args:
        N(int):channel number of y and z
        groups(list):partition of y into y0,y1
        r(list):upsampling factor
        M(int):channel number of task feature
    '''
    def __init__(self,N=192,groups=[128,64],r=[2,1,1,1],M=256,**kwargs):
        super().__init__(N=N,**kwargs)
        assert sum(groups)==N
        self.context_prediction=nn.ModuleList([MaskedConv2d(groups[i],2*groups[i],kernel_size=5,padding=2,stride=1)
                                 for i in range(0,2)])
        self.entropy_parameters=nn.ModuleList([nn.Sequential(
            conv1x1(groups[i]*4,groups[i]*10//3),
            nn.LeakyReLU(inplace=True),
            conv1x1(groups[i]*10//3,groups[i]*8//3),
            nn.LeakyReLU(inplace=True),
            conv1x1(groups[i]*8//3,groups[i]*2),
        ) for i in range(0,2)])
        self.gaussian_conditional=nn.ModuleList([GaussianConditional(None) for i in range(0,2)])
        self.LST=nn.Sequential(
            ResidualBlock(groups[0],M),
            ResidualBlockUpsample(M,M,r[0]),
            ResidualBlock(M,M),
            ResidualBlockUpsample(M,M,r[1]),
            ResidualBlock(M,M),
            ResidualBlockUpsample(M,M,r[2]),
            ResidualBlock(M,M),
            subpel_conv3x3(M,M,r[3]),
        )
        self.groups=groups
        self.yolo=self.yolov3()
        
        
    def forward(self,x):
        y=self.g_a(x)
        padding=1
        flag=False
        if y.size(2)%4!=0:
            y=F.pad(y, (padding, padding, padding, padding))
            flag=True
        z=self.h_a(y)
        z_hat,z_likelihoods=self.entropy_bottleneck(z)
        params=self.h_s(z_hat)
        params=torch.split(params,[i*2 for i in self.groups],dim=1)
        y_=torch.split(y,self.groups,dim=1)
        y_hat=[self.gaussian_conditional[i].quantize(y_[i],'noise' if self.training else 'dequantize')
               for i in range(0,2)]
        ctx_params=[self.context_prediction[i](y_hat[i]) for i in range(0,2)]
        gaussian_params=[self.entropy_parameters[i](torch.cat((params[i],ctx_params[i]),dim=1))
                         for i in range(0,2)]
        likelihoods=[Tensor()]*2
        for i in range(0,2):
            scales_hat,means_hat=gaussian_params[i].chunk(2,1)
            _,_likelihoods=self.gaussian_conditional[i](y_[i],scales_hat,means=means_hat)
            likelihoods[i]=_likelihoods
        if flag:
            y_hat=[F.pad(y_i, (-padding, -padding, -padding, -padding)) for y_i in y_hat]
            likelihoods=[F.pad(likelihood, (-padding, -padding, -padding, -padding)) for likelihood in likelihoods]
        x_hat=self.g_s(torch.cat(y_hat,dim=1))

        #task-relevant feature maps
        feature=self.LST(y_hat[0])
        feature=self.yolo.BN_RELU(feature)
        feature_target=self.yolo.FrontEnd(x)


        return {
            "x_hat": x_hat,
            "likelihoods": {"y_0": likelihoods[0],"y_1": likelihoods[1], "z": z_likelihoods},
            "feature":feature,
            "feature_target":feature_target
        }
    
    def yolov3(self,config_path='YOLOv3/config/yolov3.cfg',weights_path='YOLOv3/yolov3.weights'):
        root=os.getcwd()
        config_path=root+'/'+config_path
        weights_path=root+'/'+weights_path
        model=load_model(config_path,weights_path)
        model.eval()
        for _,param in model.named_parameters():
            param.requires_grad=False
        return model
        
    def compress(self,x):
        #copy from JointAutoregressiveHierarchicalPriors
        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU).",
                stacklevel=2,
            )
        y=self.g_a(x)
        z=self.h_a(y)
        z_strings=self.entropy_bottleneck.compress(z)
        z_hat=self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        params = self.h_s(z_hat)
        params=torch.split(params,[i*2 for i in self.groups],dim=1)
        
        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2
        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s
        y_hat = F.pad(y, (padding, padding, padding, padding))
        
        y_hat=torch.split(y_hat,self.groups,dim=1)
        y_strings=[]
        for j in range(0,2):
            strings=[]
            for i in range(y.size(0)):
                string = self._compress_ar(
                    j,
                    y_hat[j][i : i + 1],
                    params[j][i : i + 1],
                    y_height,
                    y_width,
                    kernel_size,
                    padding,
                )
                strings.append(string)
            y_strings.append(strings)
        
        return {"strings": y_strings.append(z_strings), "shape": z.size()[-2:]}

    def _compress_ar(self, j, y_hat, params, height, width, kernel_size, padding):
        cdf = self.gaussian_conditional[j].quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional[j].cdf_length.tolist()
        offsets = self.gaussian_conditional[j].offset.tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []

        # Warning, this is slow...
        # TODO: profile the calls to the bindings...
        masked_weight = self.context_prediction[j].weight * self.context_prediction[j].mask
        for h in range(height):
            for w in range(width):
                y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size]
                ctx_p = F.conv2d(
                    y_crop,
                    masked_weight,
                    bias=self.context_prediction[j].bias,
                )

                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h : h + 1, w : w + 1]
                gaussian_params = self.entropy_parameters[j](torch.cat((p, ctx_p), dim=1))
                gaussian_params = gaussian_params.squeeze(3).squeeze(2)
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                indexes = self.gaussian_conditional[j].build_indexes(scales_hat)

                y_crop = y_crop[:, :, padding, padding]
                y_q = self.gaussian_conditional[j].quantize(y_crop, "symbols", means_hat)
                y_hat[:, :, h + padding, w + padding] = y_q + means_hat

                symbols_list.extend(y_q.squeeze().tolist())
                indexes_list.extend(indexes.squeeze().tolist())

        encoder.encode_with_indexes(
            symbols_list, indexes_list, cdf, cdf_lengths, offsets
        )

        string = encoder.flush()
        return string
    
    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 3

        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU).",
                stacklevel=2,
            )

        # FIXME: we don't respect the default entropy coder and directly call the
        # range ANS decoder

        z_hat = self.entropy_bottleneck.decompress(strings[2], shape)
        params = self.h_s(z_hat)
        params=torch.split(params,[i*2 for i in self.groups],dim=1)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        # initialize y_hat to zeros, and pad it so we can directly work with
        # sub-tensors of size (N, C, kernel size, kernel_size)
        y_=[torch.zeros((z_hat.size(0),self.groups[i],y_height+2*padding,y_width+2*padding),device=z_hat.device)
            for i in range(0,2)]
        for j in range(0,2):
            for i, y_string in enumerate(strings[j]):
                self._decompress_ar(
                    j,
                    y_string,
                    y_[j][i : i + 1],
                    params[j][i : i + 1],
                    y_height,
                    y_width,
                    kernel_size,
                    padding,
                )
        y_hat=torch.cat(y_,dim=1)
        y_hat = F.pad(y_hat, (-padding, -padding, -padding, -padding))
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}
    
    def _decompress_ar(
        j,self, y_string, y_hat, params, height, width, kernel_size, padding
    ):
        cdf = self.gaussian_conditional[j].quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional[j].cdf_length.tolist()
        offsets = self.gaussian_conditional[j].offset.tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        # Warning: this is slow due to the auto-regressive nature of the
        # decoding... See more recent publication where they use an
        # auto-regressive module on chunks of channels for faster decoding...
        for h in range(height):
            for w in range(width):
                # only perform the 5x5 convolution on a cropped tensor
                # centered in (h, w)
                y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size]
                ctx_p = F.conv2d(
                    y_crop,
                    self.context_prediction[j].weight,
                    bias=self.context_prediction[j].bias,
                )
                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h : h + 1, w : w + 1]
                gaussian_params = self.entropy_parameters[j](torch.cat((p, ctx_p), dim=1))
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                indexes = self.gaussian_conditional[j].build_indexes(scales_hat)
                rv = decoder.decode_stream(
                    indexes.squeeze().tolist(), cdf, cdf_lengths, offsets
                )
                rv = torch.Tensor(rv).reshape(1, -1, 1, 1)
                rv = self.gaussian_conditional[j].dequantize(rv, means_hat)

                hp = h + padding
                wp = w + padding
                y_hat[:, :, hp : hp + 1, wp : wp + 1] = rv

if __name__ == '__main__':
    model=Choi2022()
    for name, param in model.named_parameters():
        print(name)
    #print(model)
    #print(model.yolo)