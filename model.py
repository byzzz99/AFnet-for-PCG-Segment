import datetime
from monai.networks.layers.factories import Conv
from monai.networks.nets.basic_unet import Pool
from monai.utils import ensure_tuple_rep
from monai.networks.blocks import UpSample
from monai.inferers import sliding_window_inference
from typing import Optional, Sequence, Union
import numpy as np
import scipy
import skimage
import torch
import torch.nn as nn
import math
import torchmetrics
import pylab as plt
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim.swa_utils import AveragedModel, update_bn

from modules import BasicConv, fftRFT
from utils import DiceBCELoss, eval_metrics



class MultiScaleConv1D(nn.Module):

    def __init__(
            self,
            in_chns: int,
            out_chns: int,
            act: Union[str, tuple] = "gelu",
            norm: Union[str, tuple] = ("instance", {"affine": True}),
            bias: bool = True,
            dropout: float = 0.0
    ):
        super().__init__()
        self.in_chns = in_chns
        self.out_chns = out_chns
        self.dropout = dropout if dropout > 0 else None


        act_map = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(inplace=True),
            "leakyrelu": nn.LeakyReLU(0.2, inplace=True)
        }
        self.act = act_map[act.lower()] if isinstance(act, str) else act


        if isinstance(norm, str):
            if norm.lower() == "instance":
                self.norm = nn.InstanceNorm1d(out_chns, affine=True)
            elif norm.lower() == "batch":
                self.norm = nn.BatchNorm1d(out_chns)
            else:
                raise ValueError(f"不支持的归一化类型: {norm}")
        else:
            norm_type, norm_kwargs = norm
            if norm_type.lower() == "instance":
                self.norm = nn.InstanceNorm1d(out_chns, **norm_kwargs)
            elif norm_type.lower() == "batch":
                self.norm = nn.BatchNorm1d(out_chns, **norm_kwargs)
            else:
                raise ValueError(f"不支持的归一化类型: {norm_type}")


        branch1_out = out_chns // 3
        branch2_out = out_chns // 3
        branch3_out = out_chns - branch1_out - branch2_out

        self.scale_branch1 = nn.Conv1d(
            in_chns, branch1_out, kernel_size=3, padding=1, dilation=1, bias=bias
        )
        self.scale_branch2 = nn.Conv1d(
            in_chns, branch2_out, kernel_size=3, padding=3, dilation=3, bias=bias
        )
        self.scale_branch3 = nn.Conv1d(
            in_chns, branch3_out, kernel_size=3, padding=5, dilation=5, bias=bias
        )


        if self.dropout is not None:
            self.drop = nn.Dropout1d(dropout)

    def forward(self, x):

        x1 = self.scale_branch1(x)
        x2 = self.scale_branch2(x)
        x3 = self.scale_branch3(x)


        x = torch.cat([x1, x2, x3], dim=1)

        x = self.act(x)
        x = self.norm(x)
        if self.dropout is not None:
            x = self.drop(x)
        return x



class AttentionMLP1D(nn.Module):

    def __init__(self, dim: int, expansion_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden_dim = int(dim * expansion_ratio)
        
        self.mlp_block = nn.Sequential(

            nn.Conv1d(dim, hidden_dim, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Dropout1d(dropout) if dropout > 0 else nn.Identity(),

            nn.Conv1d(hidden_dim, dim, kernel_size=1, bias=True),
            nn.InstanceNorm1d(dim)
        )


        self.residual_conn = nn.Identity()

    def forward(self, x):

        residual = self.residual_conn(x)
        x = self.mlp_block(x)
        return x + residual



class LocalSelfAttention1D(nn.Module):

    def __init__(
            self, 
            dim, 
            num_heads=8, 
            window_size=256, 
            overlap_ratio=0.25,
            mlp_expansion=4.0,
            mlp_dropout=0.1
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = self.dim // num_heads
        self.window_size = window_size
        self.overlap_ratio = overlap_ratio
        

        self.stride = max(1, int(window_size * (1 - overlap_ratio)))


        assert self.head_dim * num_heads == self.dim, f"通道数{dim}需为{num_heads}的倍数"
        assert 0 <= overlap_ratio < 1, "重叠百分比需在[0, 1)范围内"
        assert isinstance(window_size, int) and window_size > 0, "窗口大小需为正整数"
        assert self.stride > 0, f"计算得到的步长必须为正数，当前步长: {self.stride}"


        self.pos_encoding = nn.Parameter(torch.randn(1, dim, 1))
        

        self.qkv_conv = nn.Conv1d(dim, dim * 3, kernel_size=1)
        self.out_conv = nn.Conv1d(dim, dim, kernel_size=1)
        

        self.norm = nn.InstanceNorm1d(dim)


        self.attn_post_mlp = AttentionMLP1D(
            dim=dim,
            expansion_ratio=mlp_expansion,
            dropout=mlp_dropout
        )

    def forward(self, x):

        B, C, L = x.shape
        residual = x


        x = x + self.pos_encoding


        qkv = self.qkv_conv(x).chunk(3, dim=1)
        q, k, v = qkv


        q = q.view(B, self.num_heads, self.head_dim, L).transpose(-2, -1)
        k = k.view(B, self.num_heads, self.head_dim, L).transpose(-2, -1)
        v = v.view(B, self.num_heads, self.head_dim, L).transpose(-2, -1)


        attn_outputs = []
        i = 0
        while i < L:
            end = min(i + self.window_size, L)
            q_win = q[:, :, i:end, :]
            k_win = k[:, :, i:end, :]
            v_win = v[:, :, i:end, :]


            attn_score = torch.matmul(q_win, k_win.transpose(-2, -1))  # (B, num_heads, W, W)
            attn_score = attn_score * (self.head_dim ** -0.5)  # 缩放
            attn_weight = torch.softmax(attn_score, dim=-1)


            win_out = torch.matmul(attn_weight, v_win)
            attn_outputs.append((win_out, i, end))

            i += self.stride


        output = torch.zeros_like(q)
        weight_mask = torch.zeros_like(q[..., 0])
        for win_out, start, end in attn_outputs:
            output[..., start:end, :] += win_out
            weight_mask[..., start:end] += 1

        weight_mask = weight_mask.unsqueeze(-1)
        output = output / weight_mask.clamp(min=1e-8)


        attn_out = output.transpose(-2, -1).contiguous()
        attn_out = attn_out.view(B, self.dim, L)


        x = self.out_conv(attn_out) + residual
        x = self.norm(x)


        x = self.attn_post_mlp(x)

        return x



class LocalCrossAttention1D(nn.Module):

    def __init__(
            self, 
            dim_decoder, 
            dim_encoder, 
            num_heads=8, 
            window_size=256, 
            overlap_ratio=0.25,
            mlp_expansion=4.0,
            mlp_dropout=0.1
    ):
        super().__init__()
        self.dim_decoder = dim_decoder
        self.dim_encoder = dim_encoder
        self.num_heads = num_heads
        self.head_dim = dim_decoder // num_heads
        self.window_size = window_size
        self.overlap_ratio = overlap_ratio
        

        self.stride = max(1, int(window_size * (1 - overlap_ratio)))
        

        assert dim_decoder % num_heads == 0, f"解码器通道{dim_decoder}需为{num_heads}的倍数"
        assert 0 <= overlap_ratio < 1, "重叠百分比需在[0, 1)范围内"
        assert isinstance(window_size, int) and window_size > 0, "窗口大小需为正整数"


        self.q_proj = nn.Sequential(
            nn.Conv1d(dim_decoder, dim_decoder, kernel_size=1),
            nn.InstanceNorm1d(dim_decoder)
        )
        self.k_proj = nn.Sequential(
            nn.Conv1d(dim_encoder, dim_decoder, kernel_size=1),
            nn.InstanceNorm1d(dim_decoder)
        )
        self.v_proj = nn.Sequential(
            nn.Conv1d(dim_encoder, dim_decoder, kernel_size=1),
            nn.InstanceNorm1d(dim_decoder)
        )


        self.out_proj = nn.Sequential(
            nn.Conv1d(dim_decoder, dim_decoder, kernel_size=1),
            nn.InstanceNorm1d(dim_decoder)
        )


        self.attn_post_mlp = AttentionMLP1D(
            dim=dim_decoder,
            expansion_ratio=mlp_expansion,
            dropout=mlp_dropout
        )

    def forward(self, x_decoder, x_encoder):

        B, _, L_dec = x_decoder.shape
        residual = x_decoder
        L_enc = x_encoder.shape[-1]


        q = self.q_proj(x_decoder)
        k = self.k_proj(x_encoder)
        v = self.v_proj(x_encoder)


        q = q.view(B, self.num_heads, self.head_dim, L_dec).transpose(-2, -1)
        k = k.view(B, self.num_heads, self.head_dim, L_enc).transpose(-2, -1)
        v = v.view(B, self.num_heads, self.head_dim, L_enc).transpose(-2, -1)


        attn_outputs = []
        i = 0
        while i < L_dec:

            dec_end = min(i + self.window_size, L_dec)
            q_win = q[:, :, i:dec_end, :]
            W_dec = dec_end - i


            enc_start = int(i * L_enc / L_dec)
            enc_end = int(dec_end * L_enc / L_dec)
            enc_end = min(enc_end, L_enc)
            k_win = k[:, :, enc_start:enc_end, :]
            v_win = v[:, :, enc_start:enc_end, :]


            attn_score = torch.matmul(q_win, k_win.transpose(-2, -1))
            attn_score = attn_score * (self.head_dim ** -0.5)
            attn_weight = torch.softmax(attn_score, dim=-1)


            win_out = torch.matmul(attn_weight, v_win)
            attn_outputs.append((win_out, i, dec_end))

            i += self.stride


        output = torch.zeros_like(q)
        weight_mask = torch.zeros_like(q[..., 0])
        for win_out, start, end in attn_outputs:
            output[..., start:end, :] += win_out
            weight_mask[..., start:end] += 1

        weight_mask = weight_mask.unsqueeze(-1)
        output = output / weight_mask.clamp(min=1e-8)


        attn_out = output.transpose(-2, -1).contiguous()
        attn_out = attn_out.view(B, self.dim_decoder, L_dec)


        x = self.out_proj(attn_out) + residual


        x = self.attn_post_mlp(x)

        return x



class Down(nn.Sequential):

    def __init__(
            self,
            spatial_dims: int,
            in_chns: int,
            out_chns: int,
            act: Union[str, tuple],
            norm: Union[str, tuple],
            bias: bool,
            dropout: Union[float, tuple] = 0.0,
            sa=False,
            sa_window_size=256,
            sa_overlap_ratio=0.25,
            multi=False,
            fft=False,
            mlp_expansion=4.0,
            mlp_dropout=0.1
    ):
        super().__init__()
        assert spatial_dims == 1, "仅支持1D心音数据"
        self.max_pooling = Pool["MAX", spatial_dims](kernel_size=2)
        self.fftRFT_ = fft
        self.sa = sa
        self.multi = multi

        if self.sa:
            assert in_chns % 8 == 0, f"自注意力要求输入通道数{in_chns}为8的倍数"
            self.attention = LocalSelfAttention1D(
                dim=in_chns,
                num_heads=8,
                window_size=sa_window_size,
                overlap_ratio=sa_overlap_ratio,
                mlp_expansion=mlp_expansion,
                mlp_dropout=mlp_dropout
            )
        

        self.convs_init = MultiScaleConv1D(
            in_chns=in_chns,
            out_chns=in_chns,
            act=act,
            norm=norm,
            bias=bias,
            dropout=dropout
        )
        self.convs = MultiScaleConv1D(
            in_chns=in_chns,
            out_chns=out_chns,
            act=act,
            norm=norm,
            bias=bias,
            dropout=dropout
        )
        

        self.fftRFT = fftRFT(in_chns, in_chns)

    def forward(self, x: torch.Tensor):

        if self.sa:
            x = self.attention(x)

        x = self.max_pooling(x)

        if self.fftRFT_:
            x = self.fftRFT(x)

        if self.multi:
            x = self.convs_init(x)
            if self.fftRFT_:
                x = self.fftRFT(x)

        x = self.convs(x)
        return x


class UpCat(nn.Module):

    def __init__(
            self,
            spatial_dims: int,
            in_chns: int,
            cat_chns: int,
            out_chns: int,
            act: Union[str, tuple],
            norm: Union[str, tuple],
            bias: bool,
            dropout: Union[float, tuple] = 0.0,
            upsample: str = "deconv",
            pre_conv: Optional[Union[nn.Module, str]] = "default",
            interp_mode: str = "linear",
            align_corners: Optional[bool] = None,
            halves: bool = True,
            is_pad: bool = True,
            multi=False,
            fft=False,
            ca=False,
            ca_window_size=256,
            ca_overlap_ratio=0.25,
            mlp_expansion=4.0,
            mlp_dropout=0.1
    ):
        super().__init__()
        assert spatial_dims == 1, "仅支持1D心音数据"
        self.is_pad = is_pad
        self.fftRFT_ = fft
        self.multi = multi
        self.ca = ca


        up_chns = in_chns // 2 if halves else in_chns
        self.upsample = UpSample(
            spatial_dims=1,
            in_channels=in_chns,
            out_channels=up_chns,
            scale_factor=2,
            mode=upsample,
            pre_conv=pre_conv,
            interp_mode=interp_mode,
            align_corners=align_corners,
        )


        if self.ca:
            assert up_chns % 8 == 0, f"交叉注意力要求up_chns={up_chns}为8的倍数"
            self.cross_attention = LocalCrossAttention1D(
                dim_decoder=up_chns,
                dim_encoder=cat_chns,
                num_heads=8,
                window_size=ca_window_size,
                overlap_ratio=ca_overlap_ratio,
                mlp_expansion=mlp_expansion,
                mlp_dropout=mlp_dropout
            )

            self.convs_init = MultiScaleConv1D(
                in_chns=up_chns,
                out_chns=up_chns,
                act=act,
                norm=norm,
                bias=bias,
                dropout=dropout
            )
            self.convs = MultiScaleConv1D(
                in_chns=up_chns,
                out_chns=out_chns,
                act=act,
                norm=norm,
                bias=bias,
                dropout=dropout
            )
            self.fftRFT = fftRFT(up_chns, up_chns)
        else:

            self.convs_init = MultiScaleConv1D(
                in_chns=cat_chns + up_chns,
                out_chns=cat_chns + up_chns,
                act=act,
                norm=norm,
                bias=bias,
                dropout=dropout
            )
            self.convs = MultiScaleConv1D(
                in_chns=cat_chns + up_chns,
                out_chns=out_chns,
                act=act,
                norm=norm,
                bias=bias,
                dropout=dropout
            )
            self.fftRFT = fftRFT(cat_chns + up_chns, cat_chns + up_chns)

    def forward(self, x: torch.Tensor, x_e: Optional[torch.Tensor]):

        x_0 = self.upsample(x)
        
        if x_e is not None:

            if self.is_pad and x_e.shape[-1] > x_0.shape[-1]:
                pad_len = x_e.shape[-1] - x_0.shape[-1]
                x_0 = torch.nn.functional.pad(x_0, (0, pad_len), "replicate")
            

            x_fft = self.fftRFT(x_0) if self.fftRFT_ else x_0
            x_e_fft = self.fftRFT(x_e) if (self.fftRFT_ and not self.ca) else x_e
            

            if self.ca:
                x = self.cross_attention(x_fft, x_e)
            else:
                x = torch.cat([x_e_fft, x_fft], dim=1)
            

            if self.multi:
                x = self.convs_init(x)
            x = self.convs(x)
        else:
            x = self.convs(x_0)
        
        return x



class BasicUNet(nn.Module):

    def __init__(
            self,
            spatial_dims: int = 1,
            in_channels: int = 2,
            out_channels: int = 4,
            features: Sequence[int] = (64, 64, 128, 256, 512, 512, 64),
            act: Union[str, tuple] = "gelu",
            norm: Union[str, tuple] = ("instance", {"affine": True}),
            bias: bool = True,
            dropout: Union[float, tuple] = 0.0,
            upsample: str = "deconv",
            fft: bool = False,
            multi=False,
            sa=False,
            sa_window_size=256,
            sa_overlap_ratio=0.25,
            ca=False,
            ca_window_size=256,
            ca_overlap_ratio=0.25,
            mlp_expansion=4.0,
            mlp_dropout=0.1
    ):
        super().__init__()
        fea = ensure_tuple_rep(features, 7)
        print(f"1D BasicUNet特征通道：{fea}")

        if sa or ca:
            for i, f in enumerate(fea):
                if f % 8 != 0:
                    raise ValueError(f"特征fea[{i}]={f}需为8的倍数（启用8头注意力）")


        self.conv_0 = MultiScaleConv1D(
            in_chns=in_channels,
            out_chns=fea[0],
            act=act,
            norm=norm,
            bias=bias,
            dropout=dropout
        )


        self.down_1 = Down(
            spatial_dims=1,
            in_chns=fea[0],
            out_chns=fea[1],
            act=act,
            norm=norm,
            bias=bias,
            dropout=dropout,
            sa=sa,
            sa_window_size=sa_window_size,
            sa_overlap_ratio=sa_overlap_ratio,
            fft=fft,
            multi=multi,
            mlp_expansion=mlp_expansion,
            mlp_dropout=mlp_dropout
        )
        self.down_2 = Down(
            spatial_dims=1,
            in_chns=fea[1],
            out_chns=fea[2],
            act=act,
            norm=norm,
            bias=bias,
            dropout=dropout,
            sa=sa,
            sa_window_size=sa_window_size,
            sa_overlap_ratio=sa_overlap_ratio,
            fft=fft,
            multi=multi,
            mlp_expansion=mlp_expansion,
            mlp_dropout=mlp_dropout
        )
        self.down_3 = Down(
            spatial_dims=1,
            in_chns=fea[2],
            out_chns=fea[3],
            act=act,
            norm=norm,
            bias=bias,
            dropout=dropout,
            sa=sa,
            sa_window_size=sa_window_size,
            sa_overlap_ratio=sa_overlap_ratio,
            fft=fft,
            multi=multi,
            mlp_expansion=mlp_expansion,
            mlp_dropout=mlp_dropout
        )
        self.down_4 = Down(
            spatial_dims=1,
            in_chns=fea[3],
            out_chns=fea[4],
            act=act,
            norm=norm,
            bias=bias,
            dropout=dropout,
            sa=sa,
            sa_window_size=sa_window_size,
            sa_overlap_ratio=sa_overlap_ratio,
            fft=fft,
            multi=multi,
            mlp_expansion=mlp_expansion,
            mlp_dropout=mlp_dropout
        )
        self.down_5 = Down(
            spatial_dims=1,
            in_chns=fea[4],
            out_chns=fea[5],
            act=act,
            norm=norm,
            bias=bias,
            dropout=dropout,
            sa=sa,
            sa_window_size=sa_window_size,
            sa_overlap_ratio=sa_overlap_ratio,
            fft=fft,
            multi=multi,
            mlp_expansion=mlp_expansion,
            mlp_dropout=mlp_dropout
        )


        self.upcat_5 = UpCat(
            spatial_dims=1,
            in_chns=fea[5],
            cat_chns=fea[4],
            out_chns=fea[4],
            act=act,
            norm=norm,
            bias=bias,
            dropout=dropout,
            upsample=upsample,
            fft=fft,
            multi=multi,
            ca=ca,
            ca_window_size=ca_window_size,
            ca_overlap_ratio=ca_overlap_ratio,
            mlp_expansion=mlp_expansion,
            mlp_dropout=mlp_dropout
        )
        self.upcat_4 = UpCat(
            spatial_dims=1,
            in_chns=fea[4],
            cat_chns=fea[3],
            out_chns=fea[3],
            act=act,
            norm=norm,
            bias=bias,
            dropout=dropout,
            upsample=upsample,
            fft=fft,
            multi=multi,
            ca=ca,
            ca_window_size=ca_window_size,
            ca_overlap_ratio=ca_overlap_ratio,
            mlp_expansion=mlp_expansion,
            mlp_dropout=mlp_dropout
        )
        self.upcat_3 = UpCat(
            spatial_dims=1,
            in_chns=fea[3],
            cat_chns=fea[2],
            out_chns=fea[2],
            act=act,
            norm=norm,
            bias=bias,
            dropout=dropout,
            upsample=upsample,
            fft=fft,
            multi=multi,
            ca=ca,
            ca_window_size=ca_window_size,
            ca_overlap_ratio=ca_overlap_ratio,
            mlp_expansion=mlp_expansion,
            mlp_dropout=mlp_dropout
        )
        self.upcat_2 = UpCat(
            spatial_dims=1,
            in_chns=fea[2],
            cat_chns=fea[1],
            out_chns=fea[1],
            act=act,
            norm=norm,
            bias=bias,
            dropout=dropout,
            upsample=upsample,
            fft=fft,
            multi=multi,
            ca=ca,
            ca_window_size=ca_window_size,
            ca_overlap_ratio=ca_overlap_ratio,
            mlp_expansion=mlp_expansion,
            mlp_dropout=mlp_dropout
        )
        self.upcat_1 = UpCat(
            spatial_dims=1,
            in_chns=fea[1],
            cat_chns=fea[0],
            out_chns=fea[6],
            act=act,
            norm=norm,
            bias=bias,
            dropout=dropout,
            upsample=upsample,
            fft=fft,
            multi=multi,
            halves=False,
            ca=ca,
            ca_window_size=ca_window_size,
            ca_overlap_ratio=ca_overlap_ratio,
            mlp_expansion=mlp_expansion,
            mlp_dropout=mlp_dropout
        )


        self.final_conv = Conv["conv", 1](fea[6], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):

        x0 = self.conv_0(x)
        

        x1 = self.down_1(x0)
        x2 = self.down_2(x1)
        x3 = self.down_3(x2)
        x4 = self.down_4(x3)
        x5 = self.down_5(x4)
        

        u5 = self.upcat_5(x5, x4)
        u4 = self.upcat_4(u5, x3)
        u3 = self.upcat_3(u4, x2)
        u2 = self.upcat_2(u3, x1)
        u1 = self.upcat_1(u2, x0)
        

        logits = self.final_conv(u1)
        logits = torch.sigmoid(logits / 0.2)
        return logits



class SEGNET(pl.LightningModule):

    def __init__(self,
                 net: nn.Module,
                 featureLength: int = 2560,
                 learning_rate: float = 1e-4,
                 in_channels: int = 2,
                 out_channels: int = 4,
                 minsize: int = 50,
                 thr: float = 0.5,
                 device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 infer_path: str = "/workspace/data/lightning_logs/version_{ver}/checkpoints/",
                 path: str = "./",
                 year: int = 2000,
                 toler: int = 40,
                 swa_start_epoch_ratio: float = 0.1,
                 swa_update_batch_size: int = 8,
                 ):
        super(SEGNET, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.featureLength = featureLength
        self.learning_rate = learning_rate
        self.net = net
        self.minsize = minsize
        self.thr = thr
        self.mdevice = device
        self.toler = toler
        self.infer_path = infer_path
        self.path = path
        self.year = year

        self.swa_start_epoch_ratio = swa_start_epoch_ratio
        self.swa_update_batch_size = swa_update_batch_size
        self.swa_model = None
        self.swa_start_epoch = 0

        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.lossfn = DiceBCELoss()

        self.checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            mode='min',
            filename='best',
            save_top_k=1,
            save_last=True,
            dirpath=self.infer_path
        )


        if out_channels == 5:
            task = 'multilabel'
            average = 'micro'
        else:
            task = 'multiclass'
            average = 'macro'
            
        self.validACC = torchmetrics.Accuracy(task=task, num_classes=out_channels, average=average)
        self.validSEN = torchmetrics.Recall(task=task, num_classes=out_channels, average=average)
        self.validPPV = torchmetrics.Precision(task=task, num_classes=out_channels, average=average)
        self.validF1 = torchmetrics.F1Score(task=task, num_classes=out_channels, average=average)

        self.save_hyperparameters()

    def on_train_start(self):
        total_epochs = self.trainer.max_epochs
        self.swa_start_epoch = int(total_epochs * self.swa_start_epoch_ratio)
        print(f"SWA将在第{self.swa_start_epoch}轮开始（总轮数：{total_epochs}）")
        self.swa_model = GPUSWA(self.net, device=self.mdevice)

    def compute_loss(self, yhat, y):
        if isinstance(yhat, (tuple, list)):
            return sum(self.lossfn(pred, y) for pred in yhat)
        else:
            return self.lossfn(yhat, y)

    def forward(self, x):
        return self.net(x)

    def sw_inference(self, x):

        return sliding_window_inference(
            inputs=x,
            roi_size=self.featureLength,
            sw_batch_size=16,
            predictor=self.net,
            mode='gaussian',
            overlap=0.25,
            device=self.mdevice,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }

    def pipeline(self, batch, batch_idx, save=False):

        keys = ["s1", "s2", "sys", "dia"]
        sub_keys = ["TP", "FN", "FP", "sen", "pre", "f1"]
        metrics = {key: {sub_key: 0 for sub_key in sub_keys} for key in keys}
        metrics["fname_f1_score"] = {}

        x = batch['x'].float()
        y = batch['y_onehot'].float()
        fname = batch['fname']


        yhat = self.forward(x) if x.shape[-1] == self.featureLength else self.sw_inference(x)
        loss = self.compute_loss(yhat, y)
        uncert = torch.zeros_like(yhat[0]) if isinstance(yhat, list) else torch.zeros_like(yhat)


        if save:
            for i in range(len(x)):

                y_s1 = y[i, 0].cpu().detach().numpy().round()
                idxs_y_s1 = self.get_Binaryindex(y_s1)
                y_s2 = y[i, 2].cpu().detach().numpy().round()
                idxs_y_s2 = self.get_Binaryindex(y_s2)
                y_sys = y[i, 1].cpu().detach().numpy().round()
                idxs_y_sys = self.get_Binaryindex(y_sys)
                y_dia = y[i, 3].cpu().detach().numpy().round()
                idxs_y_dia = self.get_Binaryindex(y_dia)


                output_all = yhat[i].cpu().detach().numpy().round()
                output_all = output_all.astype(bool)
                final_output_all = np.zeros_like(output_all)
                for q in range(len(output_all)):
                    output_all_ = skimage.morphology.remove_small_objects(
                        output_all[q], self.minsize, connectivity=1
                    ).astype(int)
                    final_output_all[q] = output_all_


                output_s1 = self.apply_threshold(yhat[i, 0].cpu().detach().numpy(), 0.5)
                output_s1 = self.postprocess(output_s1)
                idxs_yhat_s1 = self.get_Binaryindex(output_s1)
                output_s2 = self.apply_threshold(yhat[i, 2].cpu().detach().numpy(), self.thr)
                output_s2 = self.postprocess(output_s2)
                idxs_yhat_s2 = self.get_Binaryindex(output_s2)
                output_sys = yhat[i, 1].cpu().detach().numpy().round()
                output_sys = self.postprocess(output_sys)
                idxs_yhat_sys = self.get_Binaryindex(output_sys)
                output_dia = yhat[i, 3].cpu().detach().numpy().round()
                output_dia = self.postprocess(output_dia)
                idxs_yhat_dia = self.get_Binaryindex(output_dia)


                segment_names = ["s1", "s2", "sys", "dia"]
                idxs_y_list = [idxs_y_s1, idxs_y_s2, idxs_y_sys, idxs_y_dia]
                idxs_yhat_list = [idxs_yhat_s1, idxs_yhat_s2, idxs_yhat_sys, idxs_yhat_dia]
                for w, seg_name in enumerate(segment_names):
                    TP, FN, FP, sen, pre, f1 = eval_metrics(
                        idxs_y_list[w], idxs_yhat_list[w],
                        metrics[seg_name]["TP"], metrics[seg_name]["FN"], metrics[seg_name]["FP"],
                        metrics[seg_name]["sen"], metrics[seg_name]["pre"], metrics[seg_name]["f1"], self.toler
                    )
                    metrics[seg_name].update({
                        "TP": TP, "FN": FN, "FP": FP, "sen": sen, "pre": pre, "f1": f1
                    })


                f1_s1 = 2*(metrics["s1"]["pre"]*metrics["s1"]["sen"])/(metrics["s1"]["pre"]+metrics["s1"]["sen"]) if (metrics["s1"]["pre"]+metrics["s1"]["sen"]) else 0
                f1_s2 = 2*(metrics["s2"]["pre"]*metrics["s2"]["sen"])/(metrics["s2"]["pre"]+metrics["s2"]["sen"]) if (metrics["s2"]["pre"]+metrics["s2"]["sen"]) else 0
                f1_sys = 2*(metrics["sys"]["pre"]*metrics["sys"]["sen"])/(metrics["sys"]["pre"]+metrics["sys"]["sen"]) if (metrics["sys"]["pre"]+metrics["sys"]["sen"]) else 0
                f1_dia = 2*(metrics["dia"]["pre"]*metrics["dia"]["sen"])/(metrics["dia"]["pre"]+metrics["dia"]["sen"]) if (metrics["dia"]["pre"]+metrics["dia"]["sen"]) else 0
                metrics["fname_f1_score"][fname[i]] = {
                    "s1": round(f1_s1, 5),
                    "s2": round(f1_s2, 5),
                    "sys": round(f1_sys, 5),
                    "dia": round(f1_dia, 5),
                    "mean": round(np.mean([f1_s1, f1_s2, f1_sys, f1_dia]), 5)
                }


                plt.figure(figsize=(27, 35))
                plt.subplot(611)
                plt.title(f'{fname[i]}, ground truth')
                plt.plot(x[i, 0].cpu().detach().numpy(), label='x', color='black', alpha=.6)
                plt.scatter(idxs_y_s1, [5]*len(idxs_y_s1), color='r', label='y_s1')
                plt.scatter(idxs_y_s2, [5]*len(idxs_y_s2), color='b', label='y_s2')
                
                plt.subplot(612)
                plt.title(f'Prediction result')
                plt.plot(x[i, 0].cpu().detach().numpy(), label='x', color='black', alpha=.6)
                plt.scatter(idxs_yhat_s1, [5]*len(idxs_yhat_s1), color='r', label='yhat_s1')
                plt.scatter(idxs_yhat_s2, [5]*len(idxs_yhat_s2), color='b', label='yhat_s2')
                
                plt.subplot(613)
                plt.title(f'Ground Truth fill')
                x_values = x[i, 0].cpu().detach().numpy()
                plt.plot(x_values, label='x', color='black', alpha=.6)
                plt.fill_between(np.arange(len(x_values)), 0, np.where(y_s1!=0, x_values, np.nan), color='r', alpha=0.5)
                plt.fill_between(np.arange(len(x_values)), 0, np.where(y_s2!=0, x_values, np.nan), color='b', alpha=0.5)
                
                plt.subplot(614)
                plt.title('Ground Truth tile')
                y_s1_map = np.tile(y_s1, (50, 1))
                y_s2_map = np.tile(y_s2, (50, 1))
                plt.imshow(y_s1_map, aspect='auto', cmap='Reds', alpha=0.5, 
                           extent=[0, len(y_s1), np.min(x_values), np.max(x_values)])
                plt.imshow(y_s2_map, aspect='auto', cmap='Blues', alpha=0.5, 
                           extent=[0, len(y_s2), np.min(x_values), np.max(x_values)])
                
                plt.subplot(615)
                plt.title(f'Prediction result (s1)')
                yhat_s1 = yhat[i, 0].cpu().detach().numpy()
                plt.imshow(np.tile(yhat_s1, (50, 1)), aspect='auto', cmap='Reds', alpha=0.5,
                           extent=[0, len(yhat_s1), np.min(x_values), np.max(x_values)])
                
                plt.subplot(616)
                plt.title(f'Prediction result (s2)')
                yhat_s2 = yhat[i, 2].cpu().detach().numpy()
                plt.imshow(np.tile(yhat_s2, (50, 1)), aspect='auto', cmap='Blues', alpha=0.5,
                           extent=[0, len(yhat_s2), np.min(x_values), np.max(x_values)])
                
                plt.savefig(f'{self.path}z_{fname[i]}.png', dpi=300)
                plt.close()

        return {'loss': loss, "x": x, "y": y, "yhat": yhat, 'fname': fname[0], 'uncert': uncert, 'metrics': metrics}

    def training_step(self, batch, batch_idx):
        result = self.pipeline(batch, batch_idx)
        self.log('loss', result['loss'], on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return {"loss": result['loss']}

    def validation_step(self, batch, batch_idx):
        result = self.pipeline(batch, batch_idx)
        self.log('val_loss', result['loss'], on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.validation_step_outputs.append(result)
        return result

    def on_validation_epoch_end(self):
        self.checkpoint_callback.on_validation_epoch_end(self.current_epoch, self.trainer)
        self.evaluations(self.validation_step_outputs, plot=False, save=False)
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        result = self.pipeline(batch, batch_idx, True)
        self.log('test_loss', result['loss'], on_step=False, on_epoch=True, sync_dist=True)
        self.test_step_outputs.append(result)
        return result

    def on_test_epoch_end(self):
        print("测试阶段结束，开始计算最终指标...")
        self.evaluations(self.test_step_outputs, plot=False, save=True)
        self.test_step_outputs.clear()

    def on_train_epoch_end(self):
        """SWA模型更新（每轮结束）"""
        if self.current_epoch >= self.swa_start_epoch and self.swa_model is not None:
            self.swa_model.update_parameters(self.net)
            if self.current_epoch % 1 == 0:
                print(f"第{self.current_epoch}轮：SWA模型已更新")

    def on_train_end(self):

        if self.swa_model is not None and self.current_epoch >= self.swa_start_epoch:
            print("最终SWA模型处理：更新BN统计量...")
            train_loader = self.trainer.train_dataloader
            swa_bn_loader = torch.utils.data.DataLoader(
                train_loader.dataset,
                batch_size=self.swa_update_batch_size,
                num_workers=0,
                pin_memory=True
            )
            update_bn(
                swa_bn_loader,
                self.swa_model,
                device=self.mdevice
            )
            swa_save_path = f"{self.infer_path}/swa_best.ckpt"
            torch.save({
                'epoch': self.current_epoch,
                'state_dict': self.swa_model.state_dict(),
                'hyper_parameters': self.hparams
            }, swa_save_path)
            print(f"SWA模型已保存至：{swa_save_path}")

    def evaluations(self, outputs, plot=False, save=False):

        keyss = ["s1", "s2", "sys", "dia"]
        sub_keyss = ["TP", "FN", "FP", "sen", "pre", "f1"]
        metrics_ = {k: {sk: 0 for sk in sub_keyss} for k in keyss}
        paired_test = {}
        paired_precision = {}
        paired_sensitivity = {}
        precision_collection = {k: [] for k in keyss}
        sensitivity_collection = {k: [] for k in keyss}

        for o in outputs:
            if save:
                fname = o['fname']
                paired_test[fname] = o["metrics"]["fname_f1_score"][fname]
                paired_precision[fname] = {
                    "s1": round(o['metrics']["s1"]["pre"], 5),
                    "s2": round(o['metrics']["s2"]["pre"], 5),
                    "sys": round(o['metrics']["sys"]["pre"], 5),
                    "dia": round(o['metrics']["dia"]["pre"], 5)
                }
                paired_sensitivity[fname] = {
                    "s1": round(o['metrics']["s1"]["sen"], 5),
                    "s2": round(o['metrics']["s2"]["sen"], 5),
                    "sys": round(o['metrics']["sys"]["sen"], 5),
                    "dia": round(o['metrics']["dia"]["sen"], 5)
                }
                for segment in keyss:
                    precision_collection[segment].append(paired_precision[fname][segment])
                    sensitivity_collection[segment].append(paired_sensitivity[fname][segment])

        if save:

            for d in keyss:
                for k in ["sen", "pre", "f1"]:
                    metrics_[d][k] = np.mean([o['metrics'][d][k] for o in outputs]) if outputs else 0
                    metrics_[d][k] = round(metrics_[d][k], 4)


            np.save(f'{self.path}fname_f1_score.npy', paired_test)
            np.save(f'{self.path}f1_score_collection.npy', {k: np.mean(v) for k, v in precision_collection.items()})
            np.save(f'{self.path}fname_pre_score.npy', paired_precision)
            np.save(f'{self.path}pre_score_collection.npy', {k: np.mean(v) for k, v in precision_collection.items()})
            np.save(f'{self.path}fname_sen_score.npy', paired_sensitivity)
            np.save(f'{self.path}sen_score_collection.npy', {k: np.mean(v) for k, v in sensitivity_collection.items()})


            with open(f'{self.path}PCG_Metrics_{self.year}_toler{self.toler}_result.txt', 'w') as file:
                file.write("心音分割评估指标：\n")
                for segment in keyss:
                    file.write(f"{segment} - 敏感度: {metrics_[segment]['sen']}, "
                              f"精确率: {metrics_[segment]['pre']}, F1分数: {metrics_[segment]['f1']}\n")


            plt.rcParams['figure.figsize'] = (9, 9)
            species = ['s1', 's2', 'systolic', 'diastole']
            penguin_means = {
                'Sensitivity': [metrics_[k]['sen'] for k in keyss],
                'Precision': [metrics_[k]['pre'] for k in keyss],
                'F1_Score': [metrics_[k]['f1'] for k in keyss]
            }

            x_ = np.arange(len(species))
            width = 0.25
            multiplier = 0
            fig, ax = plt.subplots(layout='constrained')
            for attribute, measurement in penguin_means.items():
                offset = width * multiplier
                rects = ax.bar(x_ + offset, measurement, width, label=attribute)
                ax.bar_label(rects, padding=15)
                multiplier += 1

            ax.set_title('心音分割评估指标', fontsize=17)
            ax.set_xticks(x_ + width, species)
            ax.set_xticklabels(species, fontsize=16)
            ax.legend(loc='upper left', ncols=3, fontsize=12.5)
            ax.set_ylim(0, np.max(penguin_means['F1_Score']) + 0.3)
            plt.subplots_adjust(top=0.8)
            
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M")
            plt.savefig(f'{self.path}PCG_Metrics_{self.year}_toler{self.toler}_{timestamp}.png', dpi=300, bbox_inches='tight')
            plt.close()

    @staticmethod
    def apply_threshold(pred, t):

        try:
            result = pred.clone()
        except:
            result = pred.copy()
        result[result >= t] = 1
        result[result < t] = 0
        return result

    @staticmethod
    def postprocess(single_array):

        single_array = single_array.astype(bool)
        single_array = skimage.morphology.remove_small_objects(
            single_array, min_size=50, connectivity=1
        ).astype(int)
        return single_array

    @staticmethod
    def get_Binaryindex(arr):

        idxs = []
        arr_ = arr.copy().round()
        label_result, count = scipy.ndimage.label(arr_)
        for i in range(1, count + 1):
            index = np.where(label_result == i)[0]
            if len(index) == 0:
                continue
            start = index[0]
            end = index[-1]
            idxs.append(int(np.mean([start, end])))
        return idxs


class GPUSWA(AveragedModel):

    def __init__(self, model, device="cuda"):
        super().__init__(
            model,
            avg_fn=lambda avg_p, p, n: (avg_p * n + p) / (n + 1)
        )
        self.to(device)
        self.device = device

    def update_parameters(self, model):

        if next(model.parameters()).device != self.device:
            model = model.to(self.device)
        super().update_parameters(model)