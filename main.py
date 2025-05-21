import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import torch
from torchvision.transforms import Resize
import argparse
import random
import shutil
#import clip
import lpips
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler as lrs
from torch.utils.data import DataLoader
from net.lformer import net
from data import get_training_set, get_eval_set
from utils import *
from torch.utils.tensorboard import SummaryWriter

#python main.py --data_train /data2/lhq/dataset/pair_lie_dataset/PairLIE-training-dataset/ --save_folder weights/prior0.4
# Training settings
parser = argparse.ArgumentParser(description='PairLIE')
parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
parser.add_argument('--nEpochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=1, help='Snapshots')
parser.add_argument('--start_iter', type=int, default=1, help='Starting Epoch')
parser.add_argument('--lr', type=float, default=1e-5, help='Learning Rate. Default=1e-4')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--decay', type=int, default='100', help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decay factor for step decay')
parser.add_argument('--seed', type=int, default=32, help='random seed to use. Default=32')
parser.add_argument('--data_train', type=str, default='../dataset/PairLIE-training-dataset/')
parser.add_argument('--data_val', type=str, default='/data2/lhq/dataset/pair_lie_dataset/PairLIE-testing-dataset/LOL-test/raw/')
parser.add_argument('--rgb_range', type=int, default=1, help='maximum value of RGB')
parser.add_argument('--save_folder', default='weights/full_loss/', help='Location to save checkpoint models')
parser.add_argument('--logroot', default='logs/tiaocan_loss0.1', help='Location to save logs')
parser.add_argument('--output_folder', default='results/', help='Location to save checkpoint models')
opt = parser.parse_args()

def seed_torch(seed=opt.seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
seed_torch()
cudnn.benchmark = True

def train():
    model.train()
    loss_print = 0
    """ CLIP, preprocess = clip.load("RN50", device=device)
    torch_resize = Resize([224,224])

    text1 = clip.tokenize(["low light image", "high light image"]).to(device)
    text2 = clip.tokenize(["noisy", "noiseless"]).to(device) """
    #text3 = clip.tokenize(["noisy", "noiseless"]).to(device)
    """ I_clip_pool = ImagePool(50)
    real_B=np.array([[0., 1.]])
    real_B=torch.Tensor(real_B).to(device) """

    l_e = L_exp(48,0.5)
    l_color = L_color()
    l_spa = L_spa()
    l_tv = L_TV()
    #criterionCycle = torch.nn.L1Loss()
    for iteration, batch in enumerate(training_data_loader, 1):

        input, file1 = batch[0], batch[1]
        input = input.cuda()
        mask1, mask2 = generate_mask_pair(input)
        im1 = generate_subimages(input, mask1)
        im2 = generate_subimages(input, mask2)

        im2 = gamma_correction(im2)
        #print(im1)
        im1 = im1.cuda()
        im2 = im2.cuda()
        L1, el1, R1, X1, I1 = model(im1)
        L2, el2, R2, X2, _ = model(im2)
        _,_,_,_,R3 = model(input)
        sub_r1 = generate_subimages(R3,mask1)
        sub_r2 = generate_subimages(R3,mask2)

        I1 = torch.clamp(I1, 0, 1)
        el1 = torch.clamp(el1, 0, 1)
        X1 = torch.clamp(X1, 0, 1)
        R1 = torch.clamp(R1, 0, 1)
        R2 = torch.clamp(R2, 0, 1)
        #DI1 = torch.clamp(DI1, 0, 1)
        
        """ I_CLIP = torch_resize(I_clip_pool.query(I1))
        logits_per_image, logits_per_text = CLIP(I_CLIP, text2)
        probs = logits_per_image.softmax(dim=-1)
        loss_I_CLIP = criterionCycle(probs, real_B) """
        #print(loss_I_CLIP)
        loss1 = C_loss(R1, R2) + C_loss(R1-R2, sub_r1-sub_r2) * 0.5
        loss2 = R_loss(L1, R1, im1, X1)
        loss3 = P_loss(im1, X1)
        #loss4 = l_e(I1) + 0.1*torch.mean(l_spa(X1, I1)) + 0.25*loss_I_CLIP
        #e_loss = enhancement_loss(X1, el1)
        #print("eloss:",e_loss)
        loss4 = l_e(I1) + 0.1*torch.mean(l_spa(X1, I1)) + 0.1*l_tv(I1) + 0.5*torch.mean(l_color(I1))
        #print("loss4:",loss4)
        loss =  loss1  + loss2  + loss3 * 500 + loss4
        #print(loss4)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_print = loss_print + loss.item()
        if iteration % 100 == 0:
            print("===> Epoch[{}]({}/{}): Loss: {:.4f} || Learning rate: lr={}.".format(epoch,
                iteration, len(training_data_loader), loss_print, optimizer.param_groups[0]['lr']))
            loss_print = 0

def checkpoint(epoch):
    model_out_path = opt.save_folder+"epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))
def ssim(prediction, target):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    img1 = prediction.astype(np.float64)
    img2 = target.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5] 
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calculate_ssim(target, ref):
    '''
    calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    img1 = np.array(target, dtype=np.float64)
    img2 = np.array(ref, dtype=np.float64)
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:, :, i], img2[:, :, i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def calculate_psnr(target, ref):
    img1 = np.array(target, dtype=np.float32)
    img2 = np.array(ref, dtype=np.float32)
    diff = img1 - img2
    psnr = 10.0 * np.log10(255.0 * 255.0 / np.mean(np.square(diff)))
    return psnr

cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")
device = "cuda" if torch.cuda.is_available() else "cpu"

print('===> Loading datasets')
train_set = get_training_set(opt.data_train)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
test_set = get_eval_set(opt.data_val)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)

print('===> Building model ')
model= net().cuda()
optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)

writer = SummaryWriter(opt.logroot)
milestones = []
for i in range(1, opt.nEpochs+1):
    if i % opt.decay == 0:
        milestones.append(i)

scheduler = lrs.MultiStepLR(optimizer, milestones, opt.gamma)
label_dir = '/data2/lhq/dataset/pair_lie_dataset/PairLIE-testing-dataset/LOL-test/reference/'
score_best = 0
# shutil.rmtree(opt.save_folder)
# os.mkdir(opt.save_folder)
if not os.path.exists(opt.save_folder):
    os.mkdir(opt.save_folder)
for epoch in range(opt.start_iter, opt.nEpochs + 1):
    train()
    scheduler.step()
    if epoch % opt.snapshots == 0:
    #if epoch == 0:
        checkpoint(epoch)
        avg_psnr = 0
        avg_ssim = 0
        avg_lpips = 0
        n = 0
        loss_fn = lpips.LPIPS(net='alex')
        loss_fn.cuda()

        for batch in testing_data_loader:
            n += 1
            with torch.no_grad():
                input, name = batch[0], batch[1]
            input = input.cuda()
            print(name)

            with torch.no_grad():
                L, el, R, X ,I= model(input)
                D = input- X
                I = torch.clamp(I, 0, 1)

                #EN_I = torch.clamp(EN_I, 0, 1)
                #print(EN_I.shape)

            im2 = Image.open(label_dir + name[0]).convert('RGB')
            (h, w) = im2.size
            im1 = I.squeeze(0)
            im1 = im1.permute(1,2,0).cpu() 

            im1 = np.array(im1)*255.
            im1 = im1.astype(np.uint8)
            im2 = np.array(im2)
            score_psnr = calculate_psnr(im1, im2)
            score_ssim = calculate_ssim(im1, im2)

            ex_p0 = I
            ex_ref = lpips.im2tensor(lpips.load_image(label_dir + name[0]))
            ex_p0 = ex_p0.cuda()
            ex_ref = ex_ref.cuda()
            score_lpips = loss_fn.forward(ex_ref, ex_p0)
        
            avg_psnr += score_psnr
            avg_ssim += score_ssim
            avg_lpips += score_lpips
        
        avg_psnr = avg_psnr / n
        avg_ssim = avg_ssim / n
        avg_lpips = avg_lpips / n
        writer.add_scalar('psnr', avg_psnr, epoch)
        writer.add_scalar('ssim', avg_ssim, epoch)
        writer.add_scalar('lpips', avg_lpips, epoch)

        print("===> Avg.PSNR: {:.4f} dB ".format(avg_psnr))
        print("===> Avg.SSIM: {:.4f} ".format(avg_ssim))
        print("===> Avg.LPIPS: {:.4f} ".format(avg_lpips.item()))

