import torch
from torch import nn
import torch.nn.functional as F


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class MlpSplit(nn.Module):
    def __init__(self, mlp_ratio=2., mscsa_params=None, drop=0.):
        super().__init__()

        self.mscsa_params = mscsa_params

        pipelines = []
        for i in range(len(mscsa_params['channels'])):
            channel = mscsa_params['channels'][i]
            hidden_channel = int(channel * mlp_ratio)

            pipeline = nn.Sequential(
                mscsa_params['norm_op'](channel),
                mscsa_params['conv_op'](channel, hidden_channel, 1),
                mscsa_params['norm_op'](hidden_channel),
                mscsa_params['conv_op'](hidden_channel, hidden_channel, 3, padding=1, groups=hidden_channel),
                nn.ReLU6(),
                mscsa_params['dropout_op'](drop),
                mscsa_params['conv_op'](hidden_channel, channel, 1),
                mscsa_params['dropout_op'](drop),
            )
            
            pipelines.append(pipeline)
        
        self.pipelines = nn.ModuleList(pipelines)

    def forward(self, x):

        xs = x.split(self.mscsa_params['channels'], dim=1)
        
        out = []
        for i, x in enumerate(xs):

            x = self.pipelines[i](x)
            out.append(x)

        x = torch.cat(out, dim=1)

        return x


class MlpChannel(nn.Module):
    def __init__(self, mlp_ratio=2., mscsa_params=None, drop=0.,):
        super().__init__()

        self.mscsa_params = mscsa_params

        channel = sum(self.mscsa_params['channels'])
        hidden_channel = int(channel * mlp_ratio)

        self.pipeline = nn.Sequential(
            mscsa_params['norm_op'](channel),
            mscsa_params['conv_op'](channel, hidden_channel, 1),
            mscsa_params['norm_op'](hidden_channel),
            mscsa_params['conv_op'](hidden_channel, hidden_channel, 3, padding=1, groups=hidden_channel),
            nn.ReLU6(),
            mscsa_params['dropout_op'](drop),
            mscsa_params['conv_op'](hidden_channel, channel, 1),
            mscsa_params['dropout_op'](drop),
        )

    def forward(self, x):

        x = self.pipeline(x)

        return x


class AttentionLocationCSwinMultiScale2SeparateConvNonShare(torch.nn.Module):
    def __init__(self, qkv_bias=False, mscsa_params=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.mscsa_params = mscsa_params
       
        self.dim = sum(self.mscsa_params['channels'])
        self.attn_ratio = mscsa_params['head_attn_ratio']
        self.num_heads = mscsa_params['head_loc']
        self.head_dim = mscsa_params['head_dim_loc']
        self.head_dim_v = int(self.attn_ratio * self.head_dim)
        self.embed_dim = self.num_heads * self.head_dim
        self.embed_dim_v = int(self.attn_ratio * self.embed_dim)
        self.scale = self.head_dim ** -0.5

        self.norm = mscsa_params['norm_op'](self.dim)

        self.q = nn.Sequential(
            mscsa_params['conv_op'](self.dim, self.embed_dim, 1, bias=qkv_bias),
            mscsa_params['norm_op'](self.embed_dim),
        )
        v1 = nn.Sequential(
            mscsa_params['conv_op'](self.dim, self.embed_dim_v, 1, bias=qkv_bias),
            mscsa_params['norm_op'](self.embed_dim_v),
        )
        v2 = nn.Sequential(
            mscsa_params['conv_op'](self.dim, self.embed_dim_v, 1, bias=qkv_bias),
            mscsa_params['norm_op'](self.embed_dim_v),
        )
        v3 = nn.Sequential(
            mscsa_params['conv_op'](self.dim, self.embed_dim_v, 1, bias=qkv_bias),
            mscsa_params['norm_op'](self.embed_dim_v),
        )
        self.v = nn.ModuleList([v1, v2, v3])

        k1 = nn.Sequential(
            mscsa_params['conv_op'](self.dim, self.embed_dim, 1, bias=qkv_bias),
            mscsa_params['norm_op'](self.embed_dim),
        )
        k2 = nn.Sequential(
            mscsa_params['conv_op'](self.dim, self.embed_dim, 1, bias=qkv_bias),
            mscsa_params['norm_op'](self.embed_dim),
        )
        k3 = nn.Sequential(
            mscsa_params['conv_op'](self.dim, self.embed_dim, 1, bias=qkv_bias),
            mscsa_params['norm_op'](self.embed_dim),
        )
        self.k = nn.ModuleList([k1, k2, k3])

        self.downsample_1 = nn.Sequential(
            mscsa_params['conv_op'](self.dim, self.dim, 3, bias=qkv_bias, stride=2, padding=1, groups=self.dim),
            mscsa_params['norm_op'](self.dim),
        )
        self.downsample_2 = nn.Sequential(
            mscsa_params['conv_op'](self.dim, self.dim, 5, bias=qkv_bias, stride=3, padding=2, groups=self.dim),
            mscsa_params['norm_op'](self.dim),
        )
        self.v_conv = nn.Sequential(
            nn.Hardswish(inplace=False),
            mscsa_params['conv_op'](self.embed_dim_v, self.embed_dim_v, 3, padding=1, groups=self.embed_dim_v)
        )
        self.proj = nn.Sequential(
            mscsa_params['conv_op'](self.embed_dim_v, self.dim, 1),
        )

        self.attn_norm = mscsa_params['norm_op'](self.embed_dim_v)
        self.act = nn.Hardswish(inplace=True)

        self.attn_drop = mscsa_params['dropout_op'](attn_drop)
        self.proj_drop = mscsa_params['dropout_op'](proj_drop)


    def forward(self, x): 
        B = x.shape[0]
        feat_size = x.shape[2:]

        x = self.norm(x)

        q = self.q(x)

        x_dowmsample_1 = self.downsample_1(x)
        x_dowmsample_2 = self.downsample_2(x)
        x = [x, x_dowmsample_1, x_dowmsample_2]
        v = [v_proj_i(x_i) for v_proj_i, x_i in zip(self.v, x)]
        k = [k_proj_i(x_i) for k_proj_i, x_i in zip(self.k, x)]

        v_conv = self.v_conv(v[0])

        q = q.reshape(B, self.num_heads, self.head_dim, -1)
        k = torch.cat([k_i.reshape(B, self.num_heads, self.head_dim, -1) for k_i in k], dim=-1)
        v = torch.cat([v_i.reshape(B, self.num_heads, self.head_dim_v, -1) for v_i in v], dim=-1)

        q = q.permute(0, 1, 3, 2)
        v = v.permute(0, 1, 3, 2)

        attn = (q @ k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v

        x = x.permute(0, 1, 3, 2).reshape(B, self.embed_dim_v, *feat_size)
        x = x + v_conv
        x = self.act(self.attn_norm(x))
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class AttentionLocationCSwinMultiScale2SeparateConvShare(torch.nn.Module):
    def __init__(self, qkv_bias=False, mscsa_params=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.mscsa_params = mscsa_params
       
        self.dim = sum(self.mscsa_params['channels'])
        self.attn_ratio = mscsa_params['head_attn_ratio']
        self.num_heads = mscsa_params['head_loc']
        self.head_dim = mscsa_params['head_dim_loc']
        self.head_dim_v = int(self.attn_ratio * self.head_dim)
        self.embed_dim = self.num_heads * self.head_dim
        self.embed_dim_v = int(self.attn_ratio * self.embed_dim)
        self.scale = self.head_dim ** -0.5

        self.norm = mscsa_params['norm_op'](self.dim)

        self.q = nn.Sequential(
            mscsa_params['conv_op'](self.dim, self.embed_dim, 1, bias=qkv_bias),
            mscsa_params['norm_op'](self.embed_dim),
        )
        self.v = nn.Sequential(
            mscsa_params['conv_op'](self.dim, self.embed_dim_v, 1, bias=qkv_bias),
            mscsa_params['norm_op'](self.embed_dim_v),
        )

        self.k = nn.Sequential(
            mscsa_params['conv_op'](self.dim, self.embed_dim, 1, bias=qkv_bias),
            mscsa_params['norm_op'](self.embed_dim),
        )

        self.downsample_1 = nn.Sequential(
            mscsa_params['conv_op'](self.dim, self.dim, 3, bias=qkv_bias, stride=2, padding=1, groups=self.dim),
            mscsa_params['norm_op'](self.dim),
        )
        self.downsample_2 = nn.Sequential(
            mscsa_params['conv_op'](self.dim, self.dim, 5, bias=qkv_bias, stride=3, padding=2, groups=self.dim),
            mscsa_params['norm_op'](self.dim),
        )
        self.v_conv = nn.Sequential(
            nn.Hardswish(inplace=False),
            mscsa_params['conv_op'](self.embed_dim_v, self.embed_dim_v, 3, padding=1, groups=self.embed_dim_v)
        )
        self.proj = nn.Sequential(
            mscsa_params['conv_op'](self.embed_dim_v, self.dim, 1),
        )

        self.attn_norm = mscsa_params['norm_op'](self.embed_dim_v)
        self.act = nn.Hardswish(inplace=True)

        self.attn_drop = mscsa_params['dropout_op'](attn_drop)
        self.proj_drop = mscsa_params['dropout_op'](proj_drop)


    def forward(self, x): 
        B = x.shape[0]
        feat_size = x.shape[2:]

        x = self.norm(x)

        q = self.q(x)

        x_dowmsample_1 = self.downsample_1(x)
        x_dowmsample_2 = self.downsample_2(x)
        x = [x, x_dowmsample_1, x_dowmsample_2]
        # v = [v_proj_i(x_i) for v_proj_i, x_i in zip(self.v, x)]
        # k = [k_proj_i(x_i) for k_proj_i, x_i in zip(self.k, x)]
        v = [self.v(x_i) for x_i in x]
        k = [self.k(x_i) for x_i in x]

        v_conv = self.v_conv(v[0])

        q = q.reshape(B, self.num_heads, self.head_dim, -1)
        k = torch.cat([k_i.reshape(B, self.num_heads, self.head_dim, -1) for k_i in k], dim=-1)
        v = torch.cat([v_i.reshape(B, self.num_heads, self.head_dim_v, -1) for v_i in v], dim=-1)

        q = q.permute(0, 1, 3, 2)
        v = v.permute(0, 1, 3, 2)

        attn = (q @ k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v

        x = x.permute(0, 1, 3, 2).reshape(B, self.embed_dim_v, *feat_size)
        x = x + v_conv
        x = self.act(self.attn_norm(x))
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class BlockMix(nn.Module):

    def __init__(self, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., mscsa_params=None, 
                 attn=[AttentionLocationCSwinMultiScale2SeparateConvNonShare, AttentionLocationCSwinMultiScale2SeparateConvNonShare], 
                 mlp=[MlpChannel, MlpSplit]):
        super().__init__()
        self.attn1 = attn[0](
            qkv_bias=qkv_bias, mscsa_params=mscsa_params,
            attn_drop=attn_drop, proj_drop=drop)
        self.attn2 = attn[1](
            qkv_bias=qkv_bias, mscsa_params=mscsa_params,
            attn_drop=attn_drop, proj_drop=drop)
        drop_path1 = drop_path[0] if isinstance(drop_path, list) else drop_path
        drop_path2 = drop_path[1] if isinstance(drop_path, list) else drop_path
        self.drop_path1 = DropPath(drop_path1) if drop_path1 > 0. else nn.Identity()
        self.drop_path2 = DropPath(drop_path2) if drop_path2 > 0. else nn.Identity()
        self.mlp1 = mlp[0](mlp_ratio=mlp_ratio, mscsa_params=mscsa_params, drop=drop)
        self.mlp2 = mlp[1](mlp_ratio=mlp_ratio, mscsa_params=mscsa_params, drop=drop)

    def forward(self, x):
        x = x + self.drop_path1(self.attn1(x))
        x = x + self.drop_path1(self.mlp1(x))
        x = x + self.drop_path2(self.attn2(x))
        x = x + self.drop_path2(self.mlp2(x))
        return x


class DownSamplingNonCPE(nn.Module):
    def __init__(self, curr_dim, out_dim, mscsa_params):
        super().__init__()

        if curr_dim == out_dim:
            self.proj = nn.Identity()
        else:
            self.proj = mscsa_params['conv_op'](curr_dim, out_dim, 1)

        if mscsa_params['conv_op'] == nn.Conv2d:
            # self.pool = F.adaptive_avg_pool2d
            self.pool = lambda x, shape: F.interpolate(x, size=shape, mode=mscsa_params['upsample_mode'], align_corners=False)
        elif mscsa_params['conv_op'] == nn.Conv3d:
            # self.pool = F.adaptive_avg_pool3d
            self.pool = lambda x, shape: F.interpolate(x, size=shape, mode=mscsa_params['upsample_mode'], align_corners=False)

    def forward(self, x, shape):
        x = self.pool(x, shape)
        x = self.proj(x)
        return x


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class InjectionMultiSum(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        mscsa_params = None
    ) -> None:
        super(InjectionMultiSum, self).__init__()

        if inp != oup:
            self.local_embedding = nn.Sequential(
                mscsa_params['conv_op'](inp, oup, 1, bias=False),
                mscsa_params['norm_op'](oup),
            )
        else:
            self.local_embedding = nn.Identity()
        self.global_embedding = nn.Sequential(
            mscsa_params['conv_op'](oup, oup, 1, bias=False),
            mscsa_params['norm_op'](oup),
        )
        self.global_act = nn.Sequential(
            mscsa_params['conv_op'](oup, oup, 1, bias=False),
            mscsa_params['norm_op'](oup),
        )
        self.act = h_sigmoid()

        self.interpolate = lambda x, shape: F.interpolate(x, size=shape, mode=mscsa_params['upsample_mode'], align_corners=False)

    def forward(self, x_l, x_g):
        '''
        x_g: global features
        x_l: local features
        '''
        shape = x_l.shape[2:]
        local_feat = self.local_embedding(x_l)
        
        global_act = self.global_act(x_g)
        sig_act = self.interpolate(self.act(global_act), shape)
        
        global_feat = self.global_embedding(x_g)
        global_feat = self.interpolate(global_feat, shape)
        
        out = local_feat * sig_act + global_feat
        return out