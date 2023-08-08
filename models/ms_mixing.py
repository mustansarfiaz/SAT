import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_



class MixShiftBlock(nn.Module):
    r""" Mix-Shifting Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        shift_size (int): Shift size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, shift_size, shift_dist, mix_size,  layer_scale_init_value=1e-6,
                 mlp_ratio=4, drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.mlp_ratio = mlp_ratio

        self.shift_size = shift_size
        self.shift_dist = shift_dist
        self.chunk_size = [i.shape[0] for i in torch.chunk(torch.zeros(dim), self.shift_size)]
                
        self.dwDConv_lr =  nn.Conv2d(dim, dim, kernel_size=1, stride=1, groups=dim, padding=0, bias=False)
        self.dwDConv_td =  nn.Conv2d(dim, dim, kernel_size=1, stride=1, groups=dim, padding=0, bias=False)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        B_, C, H, W = x.shape

        # split groups
        xs = torch.chunk(x, self.shift_size, 1) 
        # tmp
        x_shift_lr=[]
        for x_c, shift in zip(xs, self.shift_dist):
            x_c_f = x_c.clone()
            if shift == 1:
               x_c_f = x_c_f
            else:               
                t_a=  x_c_f[:,:,:,shift]
                z_ta = torch.zeros(t_a.shape)
                x_c_f[:,:,:,shift] = z_ta
            x_shift_lr.append(x_c_f)
        
        x_shift_td=[]
        for x_c, shift in zip(xs, self.shift_dist):
            x_c_f = x_c.clone()
            if shift == 1:
               x_c_f = x_c_f
            else:               
                t_a=  x_c_f[:,:,shift,:]
                z_ta = torch.zeros(t_a.shape)
                x_c_f[:,:,shift,:] = z_ta
            x_shift_td.append(x_c_f)
            
        
        x_lr = torch.cat(x_shift_lr, 1)
        x_td = torch.cat(x_shift_td, 1)
        
        x_lr = self.dwDConv_lr(x_lr)
        x_td = self.dwDConv_td(x_td)
        
        x = x_lr + x_td 
        
        x = input + self.drop_path(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, " \
               f"shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        N = H * W
        # dwconv_1 dwconv_2
        for i in range(self.shift_size):
            flops += 2 * (N * self.chunk_size[i] * self.kernel_size[i][0])
        # x_lr + x_td
        flops += N * self.dim
        # norm
        flops += self.dim * H * W
        # pwconv
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        return flops

