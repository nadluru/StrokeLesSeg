#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import torch, math
from torch import nn
import torch.nn.functional as F
import numpy as np
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.generic_modular_residual_UNet import FabiansUNet, BasicResidualBlock
from nnunet.network_architecture.layers import DownSamplingNonCPE, BlockMix, InjectionMultiSum
from nnunet.network_architecture.layers import MlpChannel, MlpSplit, AttentionLocationCSwinMultiScale2SeparateConvNonShare


# def print_module_training_status(module):
#     if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d) or isinstance(module, nn.Dropout3d) or \
#             isinstance(module, nn.Dropout2d) or isinstance(module, nn.Dropout) or isinstance(module, nn.InstanceNorm3d) \
#             or isinstance(module, nn.InstanceNorm2d) or isinstance(module, nn.InstanceNorm1d) \
#             or isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d) or isinstance(module,
#                                                                                                       nn.BatchNorm1d):
#         print(str(module), module.training)


class FabiansUNet_MSCSA(FabiansUNet):
    

    def __init__(self, input_channels, base_num_features, num_blocks_per_stage_encoder, feat_map_mul_on_downscale,
                 pool_op_kernel_sizes, conv_kernel_sizes, props, num_classes, num_blocks_per_stage_decoder,
                 deep_supervision=False, upscale_logits=False, max_features=512, initializer=None,
                 block=BasicResidualBlock,
                 props_decoder=None):

        super().__init__(
            input_channels, 
            base_num_features, 
            num_blocks_per_stage_encoder, 
            feat_map_mul_on_downscale,
            pool_op_kernel_sizes, 
            conv_kernel_sizes, 
            props, 
            num_classes, 
            num_blocks_per_stage_decoder,
            deep_supervision, 
            upscale_logits, 
            max_features, 
            initializer,
            block,
            props_decoder
        )

        downsample = DownSamplingNonCPE
        mscsa_block = BlockMix
        attn = [AttentionLocationCSwinMultiScale2SeparateConvNonShare, AttentionLocationCSwinMultiScale2SeparateConvNonShare]
        mlp = [MlpSplit, MlpChannel]
        inj_module = InjectionMultiSum

        mscsa_depth = 3
        channel_ratio = 1.
        head_attn_ratio = 2.
        head_loc = 16
        head_dim_loc = 40
        mlp_ratio = 3.
        qkv_bias = False

        drop_rate = 0
        attn_drop_rate = 0


        if props['conv_op'] == nn.Conv2d:
            upsample_mode = 'bilinear'
            mscsa_norm_op = nn.BatchNorm2d
            mscsa_drop_op = nn.Dropout2d
        elif props['conv_op'] == nn.Conv3d:
            upsample_mode = 'trilinear'
            mscsa_norm_op = nn.BatchNorm3d
            mscsa_drop_op = nn.Dropout3d
        else:
            raise ValueError("unknown convolution dimensionality, conv op: %s" % str(conv_op))

        dpr = [x.item() for x in torch.linspace(0, 0.1 * mscsa_depth, mscsa_depth * 2)]

        channels = []
        for layer in self.encoder.stages:
            channels.append(layer.convs[-1].conv2.out_channels)

        mscsa_channels = [int(channel * channel_ratio) for channel in channels]

        self.mscsa_params = {}
        self.mscsa_params['conv_op'] = props['conv_op']
        self.mscsa_params['norm_op'] = mscsa_norm_op
        self.mscsa_params['dropout_op'] = mscsa_drop_op
        self.mscsa_params['upsample_mode'] = upsample_mode
        self.mscsa_params['channels'] = mscsa_channels
        self.mscsa_params['head_loc'] = head_loc
        self.mscsa_params['head_dim_loc'] = head_dim_loc
        self.mscsa_params['head_attn_ratio'] = head_attn_ratio
        print(self.mscsa_params)

        out_embed = []
        for stage_idx, channel in enumerate(channels):
            curr_dim = channel
            out_embed.append(downsample(
                curr_dim=curr_dim,
                out_dim=self.mscsa_params['channels'][stage_idx],
                mscsa_params=self.mscsa_params,
                ))
        self.out_embed = nn.ModuleList(out_embed)

        self.mscsa_block = nn.ModuleList([mscsa_block(
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i*2:i*2+2] if isinstance(dpr, list) else drop_path_rate, 
            mscsa_params=self.mscsa_params, attn=attn, mlp=mlp)
            for i in range(mscsa_depth)])

        out_dim = sum(mscsa_channels)
        self.norm = self.mscsa_params['norm_op'](out_dim)

        # SemanticInjectionModule
        self.SIM = nn.ModuleList()
        for i in range(len(channels)):
            self.SIM.append(
                inj_module(channels[i], mscsa_channels[i], self.mscsa_params))

        self.out_embed.apply(self._init_weights)
        self.mscsa_block.apply(self._init_weights)
        self.norm.apply(self._init_weights)
        self.SIM.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


    def forward_mscsa(self, x):

        for blk in self.mscsa_block:
            x = blk(x)

        x = self.norm(x)

        x = x.split(self.mscsa_params['channels'], dim=1)

        return x

    def forward_inject(self, xs, ys):

        outs = []
        for x, y, sim in zip(xs, ys, self.SIM):
            outs.append(sim(x, y))

        return outs

    def forward(self, x):

        outs = self.encoder(x)

        shape = outs[-2].shape[2:]
        feats = torch.cat([out_embed(x, shape) for x, out_embed in zip(outs, self.out_embed)], dim=1)

        feats = self.forward_mscsa(feats)

        outs = self.forward_inject(outs, feats)

        return self.decoder(outs)

    # @staticmethod
    # def compute_approx_vram_consumption(patch_size, base_num_features, max_num_features,
    #                                     num_modalities, num_classes, pool_op_kernel_sizes, num_conv_per_stage_encoder,
    #                                     num_conv_per_stage_decoder, feat_map_mul_on_downscale, batch_size):
    #     enc = ResidualUNetEncoder.compute_approx_vram_consumption(patch_size, base_num_features, max_num_features,
    #                                                               num_modalities, pool_op_kernel_sizes,
    #                                                               num_conv_per_stage_encoder,
    #                                                               feat_map_mul_on_downscale, batch_size)
    #     dec = PlainConvUNetDecoder.compute_approx_vram_consumption(patch_size, base_num_features, max_num_features,
    #                                                                num_classes, pool_op_kernel_sizes,
    #                                                                num_conv_per_stage_decoder,
    #                                                                feat_map_mul_on_downscale, batch_size)

    #     return enc + dec
