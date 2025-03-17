import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('/data2/lhq/PairLIE_edit')
from pdb import set_trace as stx
import numbers
from einops import rearrange
from dct_util import dct_2d,idct_2d,low_pass,low_pass_and_shuffle,high_pass
from thop import profile
import time

operation_seed_counter = 0
def get_generator():
    global operation_seed_counter
    operation_seed_counter += 1
    g_cuda_generator = torch.Generator(device="cuda")
    g_cuda_generator.manual_seed(operation_seed_counter)
    return g_cuda_generator

class AugmentNoise(object):
    def __init__(self, style):
        #print(style)
        if style.startswith('gauss'):
            self.params = [
                float(p) / 255.0 for p in style.replace('gauss', '').split('_')
            ]
            if len(self.params) == 1:
                self.style = "gauss_fix"
            elif len(self.params) == 2:
                self.style = "gauss_range"
        elif style.startswith('poisson'):
            self.params = [
                float(p) for p in style.replace('poisson', '').split('_')
            ]
            if len(self.params) == 1:
                self.style = "poisson_fix"
            elif len(self.params) == 2:
                self.style = "poisson_range"

    def add_train_noise(self, x):
        shape = x.shape
        if self.style == "gauss_fix":
            std = self.params[0]
            std = std * torch.ones((shape[0], 1, 1, 1), device=x.device)
            noise = torch.cuda.FloatTensor(shape, device=x.device)
            torch.normal(mean=0.0,
                         std=std,
                         generator=get_generator(),
                         out=noise)
            return x + noise
        elif self.style == "gauss_range":
            min_std, max_std = self.params
            std = torch.rand(size=(shape[0], 1, 1, 1),
                             device=x.device) * (max_std - min_std) + min_std
            noise = torch.cuda.FloatTensor(shape, device=x.device)
            torch.normal(mean=0, std=std, generator=get_generator(), out=noise)
            return x + noise
        elif self.style == "poisson_fix":
            lam = self.params[0]
            lam = lam * torch.ones((shape[0], 1, 1, 1), device=x.device)
            noised = torch.poisson(lam * x, generator=get_generator()) / lam
            return noised
        elif self.style == "poisson_range":
            min_lam, max_lam = self.params
            lam = torch.rand(size=(shape[0], 1, 1, 1),
                             device=x.device) * (max_lam - min_lam) + min_lam
            noised = torch.poisson(lam * x, generator=get_generator()) / lam
            return noised
##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x



##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
    
class crossAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(crossAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x, y):
        b,c,h,w = x.shape
        #print(x.shape, y)

        kv = self.kv_dwconv(self.kv(x))
        k,v = kv.chunk(2, dim=1)
        q = self.q(y)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        #print(attn.shape)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out



##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.skip_scale= nn.Parameter(torch.ones(1,dim,1,1))
        self.skip_scale2= nn.Parameter(torch.ones(1,dim,1,1))

    def forward(self, x):
        x = x*self.skip_scale + self.attn(self.norm1(x))
        x = x*self.skip_scale2 + self.ffn(self.norm2(x))

        return x
    
class crossTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(crossTransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = crossAttention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.skip_scale= nn.Parameter(torch.ones(1,dim,1,1))
        self.skip_scale2= nn.Parameter(torch.ones(1,dim,1,1))

    def forward(self, input):
        x = input[0]
        y = input[1]
        x = x*self.skip_scale + self.attn(self.norm1(x), y)
        x = x*self.skip_scale2 + self.ffn(self.norm2(x))

        return [x,y]
##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x
    
class Illumination_Estimator(nn.Module):
    def __init__(
            self, n_fea_middle, n_fea_in=4, n_fea_out=3):  #__init__部分是内部属性，而forward的输入才是外部输入
        super(Illumination_Estimator, self).__init__()

        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)

        self.depth_conv = nn.Conv2d(
            n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=n_fea_in)

        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)

    def forward(self, img):
        # img:        b,c=3,h,w
        # mean_c:     b,c=1,h,w
        
        # illu_fea:   b,c,h,w
        # illu_map:   b,c=3,h,w
        
        mean_c = img.mean(dim=1).unsqueeze(1)
        # stx()
        input = torch.cat([img,mean_c], dim=1)

        x_1 = self.conv1(input)
        illu_fea = self.depth_conv(x_1)
        illu_map = self.conv2(illu_fea)
        return illu_fea, illu_map

class F_light_prior_estimater(nn.Module):
    def __init__(
            self, n_fea_middle, n_fea_in=16, n_fea_out=3):  #__init__部分是内部属性，而forward的输入才是外部输入
        super(F_light_prior_estimater, self).__init__()

        self.conv1 = nn.Conv2d(16, n_fea_middle, kernel_size=1, bias=True)

        self.depth_conv = nn.Conv2d(
            n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=n_fea_in)

        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)

    def forward(self, img):
        # img:        b,c=3,h,w
        # mean_c:     b,c=1,h,w
        
        # illu_fea:   b,c,h,w
        # illu_map:   b,c=3,h,w
        
        _,_,h,w=img.shape
        depth = min(h,w)//10
        mean_c = img.mean(dim=1).unsqueeze(1)
        # stx()

        f0 = idct_2d(img,norm='ortho')
        f1 = low_pass(f0, 3*depth).cuda()
        f2 = low_pass_and_shuffle(f0, depth).cuda()
        f3 = high_pass(low_pass(f0, 4*depth), 2*depth).cuda()
        f4 = high_pass(f0, 5*depth).cuda()
        ff = torch.cat([f1,f2,f3,f4], dim=1)
        ff = idct_2d(ff, norm='ortho')

        input = torch.cat([img,mean_c,ff], dim=1)

        x_1 = self.conv1(input)

        illu_fea = self.depth_conv(x_1)
        illu_map = self.conv2(illu_fea)
        return illu_fea, illu_map

class L_net(nn.Module):
    def __init__(self, num=48, num_heads = 1, num_blocks = 2,inp_channels = 3, out_channels = 1, ffn_expansion_factor = 2.66, bias = False, LayerNorm_type = 'WithBias'):
        super(L_net, self).__init__()
        self.patch_embed = OverlapPatchEmbed(inp_channels, num)
        self.encoder = nn.Sequential(*[TransformerBlock(dim=num, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks)])
        self.output = nn.Conv2d(num, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, input):
        out = self.patch_embed(input)
        out = self.encoder(out)
        out = self.output(out)
        return torch.sigmoid(out) + torch.mean(input,dim=1)
        #return torch.sigmoid(out)


class R_net(nn.Module):
    def __init__(self, num=64, num_heads = 1, num_blocks = 2,inp_channels = 3, out_channels = 3, ffn_expansion_factor = 2.66, bias = False, LayerNorm_type = 'WithBias'):
        super(R_net, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, num)
        self.encoder = nn.Sequential(*[crossTransformerBlock(dim=num, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks)])
        self.output = nn.Conv2d(num, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
    

    def forward(self, input, fea):
        x = self.patch_embed(input)
        x,_ = self.encoder([x,fea])
        #x =self.encoder(x)
        out = self.output(x)
        return torch.sigmoid(out) + input
        #return torch.sigmoid(out)

class N_net(nn.Module):
    def __init__(self, num=64, num_heads = 1, num_blocks = 2,inp_channels = 3, out_channels = 3, ffn_expansion_factor = 2.66, bias = False, LayerNorm_type = 'WithBias'):
        super(N_net, self).__init__()
        self.patch_embed = OverlapPatchEmbed(inp_channels, num)
        self.encoder = nn.Sequential(*[TransformerBlock(dim=num, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks)])
        self.output = nn.Conv2d(num, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
    

    def forward(self, input):
        x = self.patch_embed(input)
        x = self.encoder(x)
        out = self.output(x)
        return torch.sigmoid(out) + input
        #return torch.sigmoid(out)

class L_enhance_net(nn.Module):
    def __init__(self, in_channel = 1, num = 32, num_heads = 1, num_blocks = 2, ffn_expansion_factor = 2.66, bias = False, LayerNorm_type = 'WithBias'):
        super(L_enhance_net, self).__init__()
        self.patch_embed = OverlapPatchEmbed(in_channel, num)
        self.encoder = nn.Sequential(*[TransformerBlock(dim=num, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks)])
        self.head = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, num, 3, 1, 0),
            nn.ReLU(),               
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),
            nn.ReLU()
        )
        self.lin = nn.AdaptiveAvgPool2d(1)
        self.tail = nn.Sequential(
            nn.Conv2d(num, num//2, kernel_size=1, padding=0, stride=1),
            nn.ReLU(),
            nn.Conv2d(num//2,1,kernel_size=1, padding=0, stride=1),
            nn.ReLU()
        )

    def forward(self, input):
        out = self.head(input)
        #out = self.patch_embed(input)
        #out = self.encoder(out)
        #print(out)
        out = self.lin(out)
        #print(out)
        out = self.tail(out)
        #print(out)
        return out


class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()        
        self.L_net = L_net(num=48)
        self.R_net = R_net(num=64)
        self.N_net = N_net(num=64)
        self.illp = F_light_prior_estimater(n_fea_middle=64)
        #self.D_net = R_net(num=64)
        #self.D_net = D_net(in_ch=3, out_ch=3, base_ch=64, num_module=4)
        self.L_enhance_net = L_enhance_net(in_channel = 1, num = 32)
        #self.L_enhance = L_enhance_net_new(in_channel=1, num=32)

    def forward(self, input):
        x = self.N_net(input)
        y,_ = self.illp(x)
        L = self.L_net(x)
        R = self.R_net(x,y)
        #R2 = self.D_net(R1)
        alpha = self.L_enhance_net(L)
        #el = torch.pow(L,alpha)
        I = torch.pow(L,alpha) * R
        """ if Training:
            noise_adder = AugmentNoise(style='poisson10')
            DI = noise_adder.add_train_noise(I)
            DI = self.D_net(DI)
        else:
            DI = self.D_net(I) """
        return L, L, R, x, I

""" a = torch.rand(1,3,128,128).cuda()
model= net().cuda()
flops, params = profile(model, inputs=(a, ))
print(flops,params)
model.eval()
start_time = time.time()
with torch.no_grad():
    _,_,_,_ = model(a)  # 模型推理
end_time = time.time()

print(f"Inference Time: {(end_time - start_time) * 1000:.2f} ms") """