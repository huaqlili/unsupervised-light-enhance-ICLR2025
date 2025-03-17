import numpy as np
import torch
from PIL import Image
from torchvision import transforms


def dct(x, norm=None):
    '''
    Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter 'norm', see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    '''
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)
    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)
    Vc = torch.view_as_real(torch.fft.fft(v, dim=1))
    k = -torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)
    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i
    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2
    V = 2 * V.view(*x_shape)
    return V


def idct(X, norm=None):
    '''
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct(dct(x)) == x
    For the meaning of the parameter 'norm', see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    '''
    x_shape = X.shape
    N = x_shape[-1]
    X_v = X.contiguous().view(-1, x_shape[-1]) / 2
    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2
    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)
    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)
    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r
    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)
    v = torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]
    return x.view(*x_shape)


def dct_2d(x, norm=None):
    '''
    2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter 'norm', see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT_II of the signal over the last 2 dimensions
    '''
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)


def idct_2d(X, norm=None):
    '''
    The inverse to 2D DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct_2d(dct_2d(x)) == x
    For the meaning of the parameter 'norm', see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimension
    '''
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    return x2.transpose(-1, -2)


def dct_3d(x, norm=None):
    '''
    3-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter 'norm', see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT_II of the signal over the last 3 dimensions
    '''
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    X3 = dct(X2.transpose(-1, -3), norm=norm)
    return X3.transpose(-1, -3).transpose(-1, -2)


def idct_3d(X, norm=None):
    '''
    The inverse to 3D DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct_3d(dct_3d(x)) == x
    For the meaning of the parameter 'norm', see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 3 dimension
    '''
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    x3 = idct(x2.transpose(-1, -3), norm=norm)
    return x3.transpose(-1, -3).transpose(-1, -2)


def low_pass(dct, threshold):
    '''
    dct: tensor of shape [... h, w]
    threshold: integer number above which to zero out
    '''
    h, w = dct.shape[-2], dct.shape[-1]
    assert 0 <= threshold <= h + w - 2, 'invalid value of threshold'
    vertical = torch.arange(0, h)[..., None].repeat(1, w).cuda()
    horizontal = torch.arange(0, w)[None, ...].repeat(h, 1).cuda()
    mask = vertical + horizontal
    while len(mask.shape) != len(dct.shape):
        mask = mask[None, ...]
    dct = torch.where(mask > threshold, torch.zeros_like(dct), dct)
    return dct


def low_pass_and_shuffle(dct, threshold):
    '''
    dct: tensor of shape [... h, w]
    threshold: integer number above which to zero out
    '''
    h, w = dct.shape[-2], dct.shape[-1]
    assert 0 <= threshold <= h + w - 2, 'invalid value of threshold'
    vertical = torch.arange(0, h)[..., None].repeat(1, w).cuda()
    horizontal = torch.arange(0, w)[None, ...].repeat(h, 1).cuda()
    mask = vertical + horizontal
    while len(mask.shape) != len(dct.shape):
        mask = mask[None, ...]
    dct = torch.where(mask > threshold, torch.zeros_like(dct), dct)
    for i in range(0, threshold + 1):         # 0 ~ threshold
        dct = shuffle_one_frequency_level(i, dct)
    return dct


def shuffle_one_frequency_level(n, dct_tensor):
    h_num = torch.arange(n + 1)
    h_num = h_num[torch.randperm(n + 1)]
    v_num = n - h_num
    dct_tensor_copy = dct_tensor.clone()
    for i in range(n + 1):  # 0 ~ n
        dct_tensor[:, :, i, n - i] = dct_tensor_copy[:, :, v_num[i], h_num[i]]
    return dct_tensor


def high_pass(dct, threshold):
    '''
    dct: tensor of shape [... h, w]
    threshold: integer number below which to zero out
    '''
    h, w = dct.shape[-2], dct.shape[-1]
    assert 0 <= threshold <= h + w - 2, 'invalid value of threshold'
    vertical = torch.arange(0, h)[..., None].repeat(1, w).cuda()
    horizontal = torch.arange(0, w)[None, ...].repeat(h, 1).cuda()
    mask = vertical + horizontal
    while len(mask.shape) != len(dct.shape):
        mask = mask[None, ...]
    dct = torch.where(mask < threshold, torch.zeros_like(dct), dct)
    return dct

import os
from PIL import Image
from torchvision.transforms import Compose, ToTensor, RandomCrop
def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img
def transform2():
    return Compose([
        ToTensor(),
    ])
def save_image(img_data, save_path):
    # 确保输入的图像数据是 PyTorch 张量
    assert isinstance(img_data, torch.Tensor), "img_data 必须是一个 PyTorch 张量"
    img_data = img_data[0]
    
    # 转置为 (h, w, 3)
    img_data = img_data.permute(1, 2, 0)
    
    # 将数据从 PyTorch 张量转换为 NumPy 数组
    img_data = img_data.cpu().numpy()
    
    # 将数据转换为 8 位无符号整数（假设数据在 0-1 范围内）
    img_data = (img_data * 255).astype(np.uint8)
    
    # 创建 PIL 图像对象
    img = Image.fromarray(img_data)
    
    # 保存图像
    img.save(save_path)
path = './results/filter/'
a_path = '/data2/lhq/dataset/LOLv1/Test/testA/22.png'
b_path = '/data2/lhq/dataset/LOLv1/Test/target/1.png'
#os.makedirs(path)
a = load_img(a_path)
b = load_img(b_path)
a=transform2()(a).cuda().unsqueeze(0)
b=transform2()(b).cuda().unsqueeze(0)

_,_,h,w=a.shape
depth = min(h,w)//10
a0 = dct_2d(a, norm='ortho').cuda()
b0 = dct_2d(b, norm='ortho').cuda()

print(a0.shape)
a1 = low_pass(a0, 4.5*depth).cuda()
a1 = idct_2d(a1, norm='ortho')
a1=(a1-torch.min(a1))/(torch.max(a1)-torch.min(a1))
#a1 = torch.clamp(a1,0,255)
save_path = path+'lq_low_'+os.path.basename(a_path)
save_image(a1, save_path)
a2 = low_pass_and_shuffle(a0, depth).cuda()
a2 = idct_2d(a2, norm='ortho')
a2=(a2-torch.min(a2))/(torch.max(a2)-torch.min(a2))
#a2 = torch.clamp(a2,0,255)
save_path = path+'lq_mini_'+os.path.basename(a_path)
save_image(a2, save_path)
a3 = high_pass(low_pass(a0, 6*depth), 4*depth).cuda()
a3 = idct_2d(a3, norm='ortho')
#a3=(a3-torch.min(a3))/(torch.max(a3)-torch.min(a3))
#a3 = torch.clamp(a3,0,255)
save_path = path+'lq_mid_'+os.path.basename(a_path)
save_image(a3, save_path)
a4 = high_pass(a0, 9*depth).cuda()
a4 = idct_2d(a4, norm='ortho')
#a4=(a4-torch.min(a4))/(torch.max(a4)-torch.min(a4))
#a4 = torch.clamp(a4,0,255)
save_path = path+'lq_high_'+os.path.basename(a_path)
save_image(a4, save_path)


a5 = torch.mean(a, dim = 1).squeeze(0)
to_pil = transforms.ToPILImage()
img = to_pil(a5)
save_path = path+'lq_illu_'+os.path.basename(a_path)
img.save(save_path)

""" b1 = low_pass(b0, 120).cuda()
b1 = idct_2d(b1, norm='ortho')
save_path = path+'hq_low_'+os.path.basename(b_path)
save_image(b1, save_path)
b2 = low_pass_and_shuffle(b0, 40).cuda()
b2 = idct_2d(b2, norm='ortho')
save_path = path+'hq_mini_'+os.path.basename(b_path)
save_image(b2, save_path)
b3 = high_pass(low_pass(b0, 160), 80).cuda()
b3 = idct_2d(b3, norm='ortho')
save_path = path+'hq_mid_'+os.path.basename(b_path)
save_image(b3, save_path)
b4 = high_pass(b0, 200).cuda()
b4 = idct_2d(b4, norm='ortho')
save_path = path+'hq_high_'+os.path.basename(b_path)
save_image(b4, save_path) """