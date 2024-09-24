import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        features = features.mean(dim=[2, 3])
        features = features.view(-1, features.size(-1))

        sim_matrix = F.cosine_similarity(
            features.unsqueeze(1), features.unsqueeze(0), dim=2
        )

        labels = labels.unsqueeze(1) == labels.unsqueeze(0)

        exp_sim = torch.exp(sim_matrix / self.temperature)
        loss = -torch.log(exp_sim[labels].sum(1) / exp_sim.sum(1))
        return loss.mean()


class Fusionloss(nn.Module):
    def __init__(self):
        super(Fusionloss, self).__init__()
        self.sobelconv = Sobelxy()

    def forward(self, image_vis, image_ir, generate_img):
        image_y = image_vis[:, :1, :, :]
        x_in_max = torch.max(image_y, image_ir)
        loss_in = F.l1_loss(x_in_max, generate_img)
        y_grad = self.sobelconv(image_y)
        ir_grad = self.sobelconv(image_ir)
        generate_img_grad = self.sobelconv(generate_img)
        x_grad_joint = torch.max(y_grad, ir_grad)
        loss_grad = F.l1_loss(x_grad_joint, generate_img_grad)
        loss_total = loss_in + 10 * loss_grad

        return loss_total, loss_in, loss_grad


def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    source = source.view(source.size(0), -1)
    target = target.view(target.size(0), -1)

    n_samples = int(source.size(0)) + int(target.size(0))
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(
        int(total.size(0)), int(total.size(0)), int(total.size(1))
    )
    total1 = total.unsqueeze(1).expand(
        int(total.size(0)), int(total.size(0)), int(total.size(1))
    )

    L2_distance = ((total0 - total1) ** 2).sum(2)

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bw) for bw in bandwidth_list]

    return sum(kernel_val)


def laplacian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    source = source.view(source.size(0), -1)
    target = target.view(target.size(0), -1)

    n_samples = int(source.size(0)) + int(target.size(0))
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(
        int(total.size(0)), int(total.size(0)), int(total.size(1))
    )
    total1 = total.unsqueeze(1).expand(
        int(total.size(0)), int(total.size(0)), int(total.size(1))
    )

    L1_distance = torch.abs(total0 - total1).sum(2)

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L1_distance.data) / (n_samples**2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L1_distance / bw) for bw in bandwidth_list]

    return sum(kernel_val)


class MKMMDLoss(nn.Module):
    def __init__(self, kernel_mul=10.0, kernel_num=5, fix_sigma=None):
        super(MKMMDLoss, self).__init__()
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
        self.fix_sigma = fix_sigma
        self.name = "MKMMD"

    def forward(self, source, target):
        batch_size = source.size(0)

        gaussian_result = gaussian_kernel(
            source, target, self.kernel_mul, self.kernel_num, self.fix_sigma
        )
        laplacian_result = laplacian_kernel(
            source, target, self.kernel_mul, self.kernel_num, self.fix_sigma
        )

        kernels = gaussian_result + laplacian_result

        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]

        loss = torch.mean(XX + YY - XY - YX)

        return loss

    def getLossName(self):
        return self.name


class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        kernely = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()

    def forward(self, x):
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx) + torch.abs(sobely)


def flatten_features(x):
    batch_size, channels, height, width = x.size()
    return x.view(batch_size, -1)


def infoNCE_loss(x, y, temperature=0.1):
    x_flat = flatten_features(x)
    y_flat = flatten_features(y)

    x_flat = F.normalize(x_flat, p=2, dim=1)
    y_flat = F.normalize(y_flat, p=2, dim=1)

    similarity_matrix = torch.matmul(x_flat, y_flat.T) / temperature

    batch_size = x.size(0)
    labels = torch.arange(batch_size).cuda()

    loss = F.cross_entropy(similarity_matrix, labels)

    return loss


def mutual_information_loss(mine, x, y):
    joint = mine(x, y).mean()
    y_shuffle = y[torch.randperm(y.size(0))]
    marginal = mine(x, y_shuffle).mean()
    mi_loss = -(joint - torch.log(marginal + 1e-6))
    return mi_loss


def cc(img1, img2):
    eps = torch.finfo(torch.float32).eps
    N, C, _, _ = img1.shape
    img1 = img1.reshape(N, C, -1)
    img2 = img2.reshape(N, C, -1)
    img1 = img1 - img1.mean(dim=-1, keepdim=True)
    img2 = img2 - img2.mean(dim=-1, keepdim=True)
    cc = torch.sum(img1 * img2, dim=-1) / (
        eps
        + torch.sqrt(torch.sum(img1**2, dim=-1))
        * torch.sqrt(torch.sum(img2**2, dim=-1))
    )
    cc = torch.clamp(cc, -1.0, 1.0)
    return cc.mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(
    img1,
    img2,
    window_size=11,
    window=None,
    size_average=True,
    full=False,
    val_range=None,
):
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * mu1_mu2 + C1
    v2 = mu1_sq + mu2_sq + C1
    v = torch.zeros_like(sigma1_sq) + 0.0001
    sigma1 = torch.where(sigma1_sq < 0.0001, v, sigma1_sq)
    mu1_sq = torch.where(mu1_sq < 0.0001, v, mu1_sq)

    ssim_map = ((2 * sigma12 + C2) * v1) / ((sigma1_sq + sigma2_sq + C2) * v2)

    ret = ssim_map
    if full:
        return ret, sigma1
    return ret


def msssim(
    img1, img2, y, window_size=11, size_average=True, val_range=None, normalize=False
):
    img1 = img1.unsqueeze(0)  # shape: [1, 1, 128, 128]
    img2 = img2.unsqueeze(0)
    y = y.unsqueeze(0)

    device = img1.device
    img3 = img1 * 0.5 + img2 * 0.5
    img3 = img3.to(device)
    Win = [11, 9, 7, 5, 3]
    loss = 0
    for s in Win:
        # [1,h,w]
        loss1, sigma1 = ssim(
            img1, y, s, size_average=size_average, full=True, val_range=val_range
        )
        loss2, sigma2 = ssim(
            img2, y, s, size_average=size_average, full=True, val_range=val_range
        )
        r = sigma1 / (sigma1 + sigma2 + 0.0000001)
        tmp = 1 - torch.mean(r * loss1) - torch.mean((1 - r) * loss2)
        loss = loss + tmp
    loss = loss / 5.0
    return loss
