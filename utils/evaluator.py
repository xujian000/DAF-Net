import numpy as np
import cv2
import sklearn.metrics as skm
from scipy.signal import convolve2d
import math
from skimage.metrics import structural_similarity as ssim
import os
from tqdm import tqdm
import numpy as np
from scipy.ndimage import convolve
import torch
import torch.nn.functional as F


def image_read_cv2(path, mode="RGB"):
    img_BGR = cv2.imread(path).astype("float32")
    assert mode == "RGB" or mode == "GRAY" or mode == "YCrCb", "mode error"
    if mode == "RGB":
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    elif mode == "GRAY":
        img = np.round(cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY))
    elif mode == "YCrCb":
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
    return img


class Evaluator:
    @classmethod
    def input_check(cls, imgF, imgA=None, imgB=None):
        if imgA is None:
            assert type(imgF) == np.ndarray, "type error"
            assert len(imgF.shape) == 2, "dimension error"
        else:
            assert type(imgF) == type(imgA) == type(imgB) == np.ndarray, "type error"
            assert imgF.shape == imgA.shape == imgB.shape, "shape error"
            assert len(imgF.shape) == 2, "dimension error"

    @classmethod
    def EN(cls, img):  # entropy
        cls.input_check(img)
        a = np.uint8(np.round(img)).flatten()
        h = np.bincount(a) / a.shape[0]
        return -sum(h * np.log2(h + (h == 0)))

    @classmethod
    def SD(cls, img):
        cls.input_check(img)
        return np.std(img)

    @classmethod
    def SF(cls, img):
        cls.input_check(img)
        return np.sqrt(
            np.mean((img[:, 1:] - img[:, :-1]) ** 2)
            + np.mean((img[1:, :] - img[:-1, :]) ** 2)
        )

    @classmethod
    def AG(cls, img):  # Average gradient
        cls.input_check(img)
        Gx, Gy = np.zeros_like(img), np.zeros_like(img)

        Gx[:, 0] = img[:, 1] - img[:, 0]
        Gx[:, -1] = img[:, -1] - img[:, -2]
        Gx[:, 1:-1] = (img[:, 2:] - img[:, :-2]) / 2

        Gy[0, :] = img[1, :] - img[0, :]
        Gy[-1, :] = img[-1, :] - img[-2, :]
        Gy[1:-1, :] = (img[2:, :] - img[:-2, :]) / 2
        return np.mean(np.sqrt((Gx**2 + Gy**2) / 2))

    @classmethod
    def MI(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        return skm.mutual_info_score(
            image_F.flatten(), image_A.flatten()
        ) + skm.mutual_info_score(image_F.flatten(), image_B.flatten())

    @classmethod
    def MSE(cls, image_F, image_A, image_B):  # MSE
        cls.input_check(image_F, image_A, image_B)
        return (
            np.mean((image_A - image_F) ** 2) + np.mean((image_B - image_F) ** 2)
        ) / 2

    @classmethod
    def CC(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        rAF = np.sum(
            (image_A - np.mean(image_A)) * (image_F - np.mean(image_F))
        ) / np.sqrt(
            (np.sum((image_A - np.mean(image_A)) ** 2))
            * (np.sum((image_F - np.mean(image_F)) ** 2))
        )
        rBF = np.sum(
            (image_B - np.mean(image_B)) * (image_F - np.mean(image_F))
        ) / np.sqrt(
            (np.sum((image_B - np.mean(image_B)) ** 2))
            * (np.sum((image_F - np.mean(image_F)) ** 2))
        )
        return (rAF + rBF) / 2

    @classmethod
    def PSNR(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        return 10 * np.log10(np.max(image_F) ** 2 / cls.MSE(image_F, image_A, image_B))

    @classmethod
    def SCD(
        cls, image_F, image_A, image_B
    ):  # The sum of the correlations of differences
        cls.input_check(image_F, image_A, image_B)
        imgF_A = image_F - image_A
        imgF_B = image_F - image_B
        corr1 = np.sum(
            (image_A - np.mean(image_A)) * (imgF_B - np.mean(imgF_B))
        ) / np.sqrt(
            (np.sum((image_A - np.mean(image_A)) ** 2))
            * (np.sum((imgF_B - np.mean(imgF_B)) ** 2))
        )
        corr2 = np.sum(
            (image_B - np.mean(image_B)) * (imgF_A - np.mean(imgF_A))
        ) / np.sqrt(
            (np.sum((image_B - np.mean(image_B)) ** 2))
            * (np.sum((imgF_A - np.mean(imgF_A)) ** 2))
        )
        return corr1 + corr2

    @classmethod
    def VIFF(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        return cls.compare_viff(image_A, image_F) + cls.compare_viff(image_B, image_F)

    @classmethod
    def compare_viff(cls, ref, dist):  # viff of a pair of pictures
        sigma_nsq = 2
        eps = 1e-10

        num = 0.0
        den = 0.0
        for scale in range(1, 5):

            N = 2 ** (4 - scale + 1) + 1
            sd = N / 5.0

            # Create a Gaussian kernel as MATLAB's
            m, n = [(ss - 1.0) / 2.0 for ss in (N, N)]
            y, x = np.ogrid[-m : m + 1, -n : n + 1]
            h = np.exp(-(x * x + y * y) / (2.0 * sd * sd))
            h[h < np.finfo(h.dtype).eps * h.max()] = 0
            sumh = h.sum()
            if sumh != 0:
                win = h / sumh

            if scale > 1:
                ref = convolve2d(ref, np.rot90(win, 2), mode="valid")
                dist = convolve2d(dist, np.rot90(win, 2), mode="valid")
                ref = ref[::2, ::2]
                dist = dist[::2, ::2]

            mu1 = convolve2d(ref, np.rot90(win, 2), mode="valid")
            mu2 = convolve2d(dist, np.rot90(win, 2), mode="valid")
            mu1_sq = mu1 * mu1
            mu2_sq = mu2 * mu2
            mu1_mu2 = mu1 * mu2
            sigma1_sq = convolve2d(ref * ref, np.rot90(win, 2), mode="valid") - mu1_sq
            sigma2_sq = convolve2d(dist * dist, np.rot90(win, 2), mode="valid") - mu2_sq
            sigma12 = convolve2d(ref * dist, np.rot90(win, 2), mode="valid") - mu1_mu2

            sigma1_sq[sigma1_sq < 0] = 0
            sigma2_sq[sigma2_sq < 0] = 0

            g = sigma12 / (sigma1_sq + eps)
            sv_sq = sigma2_sq - g * sigma12

            g[sigma1_sq < eps] = 0
            sv_sq[sigma1_sq < eps] = sigma2_sq[sigma1_sq < eps]
            sigma1_sq[sigma1_sq < eps] = 0

            g[sigma2_sq < eps] = 0
            sv_sq[sigma2_sq < eps] = 0

            sv_sq[g < 0] = sigma2_sq[g < 0]
            g[g < 0] = 0
            sv_sq[sv_sq <= eps] = eps

            num += np.sum(np.log10(1 + g * g * sigma1_sq / (sv_sq + sigma_nsq)))
            den += np.sum(np.log10(1 + sigma1_sq / sigma_nsq))

        vifp = num / den

        if np.isnan(vifp):
            return 1.0
        else:
            return vifp

    @classmethod
    def Qabf(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        gA, aA = cls.Qabf_getArray(image_A)
        gB, aB = cls.Qabf_getArray(image_B)
        gF, aF = cls.Qabf_getArray(image_F)
        QAF = cls.Qabf_getQabf(aA, gA, aF, gF)
        QBF = cls.Qabf_getQabf(aB, gB, aF, gF)

        # 计算QABF
        deno = np.sum(gA + gB)
        nume = np.sum(np.multiply(QAF, gA) + np.multiply(QBF, gB))
        return nume / deno

    @classmethod
    def Qabf_getArray(cls, img):
        # Sobel Operator Sobel
        h1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).astype(np.float32)
        h2 = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]).astype(np.float32)
        h3 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).astype(np.float32)

        SAx = convolve2d(img, h3, mode="same")
        SAy = convolve2d(img, h1, mode="same")
        gA = np.sqrt(np.multiply(SAx, SAx) + np.multiply(SAy, SAy))
        aA = np.zeros_like(img)
        aA[SAx == 0] = math.pi / 2
        aA[SAx != 0] = np.arctan(SAy[SAx != 0] / SAx[SAx != 0])
        return gA, aA

    @classmethod
    def Qabf_getQabf(cls, aA, gA, aF, gF):
        L = 1
        Tg = 0.9994
        kg = -15
        Dg = 0.5
        Ta = 0.9879
        ka = -22
        Da = 0.8
        GAF, AAF, QgAF, QaAF, QAF = (
            np.zeros_like(aA),
            np.zeros_like(aA),
            np.zeros_like(aA),
            np.zeros_like(aA),
            np.zeros_like(aA),
        )
        GAF[gA > gF] = gF[gA > gF] / gA[gA > gF]
        GAF[gA == gF] = gF[gA == gF]
        GAF[gA < gF] = gA[gA < gF] / gF[gA < gF]
        AAF = 1 - np.abs(aA - aF) / (math.pi / 2)
        QgAF = Tg / (1 + np.exp(kg * (GAF - Dg)))
        QaAF = Ta / (1 + np.exp(ka * (AAF - Da)))
        QAF = QgAF * QaAF
        return QAF

    @classmethod
    def SSIM(cls, image_F, image_A, image_B):
        data_range = 1.0 if image_F.max() <= 1.0 else 255.0
        cls.input_check(image_F, image_A, image_B)
        a = ssim(image_F, image_A, data_range=data_range)
        b = ssim(image_F, image_B, data_range=data_range)
        return (a + b) / 2

    @classmethod
    def SSIMsingle(cls, image_F, image_A):

        a = ssim(image_F, image_A)

        return a

    @classmethod
    def itensity(cls, image_F, image_A, image_B):
        # 假设 vis_img、inf_img 和 sample_y 是 NumPy 数组
        # w1 的计算
        tempture = 10
        w1 = np.exp(image_A / tempture) / (
            np.exp(image_A / tempture) + np.exp(image_B / tempture)
        )

        # 计算强度损失
        intensity_loss = np.mean(w1 * ((image_A - image_F) ** 2)) + np.mean(
            (1 - w1) * ((image_B - image_F) ** 2)
        )
        return intensity_loss

    @classmethod
    def msssimLoss(
        cls, f, a, b, window_size=11, size_average=True, val_range=None, normalize=False
    ):
        Win = [11, 9, 7, 5, 3]
        loss = 0
        for s in Win:
            loss1, sigma1 = ssim_2(a, f, s, full=True, val_range=val_range)
            loss2, sigma2 = ssim_2(b, f, s, full=True, val_range=val_range)
            r = sigma1 / (sigma1 + sigma2 + 0.0000001)
            tmp = 1 - np.mean(r * loss1) - np.mean((1 - r) * loss2)
            loss += tmp
        loss = loss / 5.0
        return loss


def gaussian_2(window_size, sigma):
    gauss = np.array(
        [
            np.exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / np.sum(gauss)


def create_window_2(window_size, channel=1):
    _1D_window = gaussian_2(window_size, 1.5).reshape(window_size, 1)
    _2D_window = np.dot(_1D_window, _1D_window.T)
    window = np.expand_dims(np.expand_dims(_2D_window, 0), 0)
    window = np.tile(window, (channel, 1, 1, 1))
    return window


def ssim_2(img1, img2, window_size=11, window=None, full=False, val_range=None):
    if val_range is None:
        L = np.max(img1) - np.min(img1)
    else:
        L = val_range

    if img1.ndim == 2:
        img1 = np.expand_dims(img1, axis=0)

    (channel, height, width) = img1.shape
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window_2(real_size, channel=channel)

    mu1 = convolve(img1, window, mode="constant", cval=0.0)
    mu2 = convolve(img2, window, mode="constant", cval=0.0)

    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = convolve(img1**2, window, mode="constant", cval=0.0) - mu1_sq
    sigma2_sq = convolve(img2**2, window, mode="constant", cval=0.0) - mu2_sq
    sigma12 = convolve(img1 * img2, window, mode="constant", cval=0.0) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * mu1_mu2 + C1
    v2 = mu1_sq + mu2_sq + C1
    v = np.full_like(sigma1_sq, 0.0001)
    sigma1_sq = np.where(sigma1_sq < 0.0001, v, sigma1_sq)
    mu1_sq = np.where(mu1_sq < 0.0001, v, mu1_sq)

    ssim_map = ((2 * sigma12 + C2) * v1) / ((sigma1_sq + sigma2_sq + C2) * v2)

    if full:
        return ssim_map, sigma1_sq
    return ssim_map


def VIFF(image_F, image_A, image_B):
    refA = image_A
    refB = image_B
    dist = image_F

    sigma_nsq = 2
    eps = 1e-10
    numA = 0.0
    denA = 0.0
    numB = 0.0
    denB = 0.0
    for scale in range(1, 5):
        N = 2 ** (4 - scale + 1) + 1
        sd = N / 5.0
        # Create a Gaussian kernel as MATLAB's
        m, n = [(ss - 1.0) / 2.0 for ss in (N, N)]
        y, x = np.ogrid[-m : m + 1, -n : n + 1]
        h = np.exp(-(x * x + y * y) / (2.0 * sd * sd))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            win = h / sumh

        if scale > 1:
            refA = convolve2d(refA, np.rot90(win, 2), mode="valid")
            refB = convolve2d(refB, np.rot90(win, 2), mode="valid")
            dist = convolve2d(dist, np.rot90(win, 2), mode="valid")
            refA = refA[::2, ::2]
            refB = refB[::2, ::2]
            dist = dist[::2, ::2]

        mu1A = convolve2d(refA, np.rot90(win, 2), mode="valid")
        mu1B = convolve2d(refB, np.rot90(win, 2), mode="valid")
        mu2 = convolve2d(dist, np.rot90(win, 2), mode="valid")
        mu1_sq_A = mu1A * mu1A
        mu1_sq_B = mu1B * mu1B
        mu2_sq = mu2 * mu2
        mu1A_mu2 = mu1A * mu2
        mu1B_mu2 = mu1B * mu2
        sigma1A_sq = convolve2d(refA * refA, np.rot90(win, 2), mode="valid") - mu1_sq_A
        sigma1B_sq = convolve2d(refB * refB, np.rot90(win, 2), mode="valid") - mu1_sq_B
        sigma2_sq = convolve2d(dist * dist, np.rot90(win, 2), mode="valid") - mu2_sq
        sigma12_A = convolve2d(refA * dist, np.rot90(win, 2), mode="valid") - mu1A_mu2
        sigma12_B = convolve2d(refB * dist, np.rot90(win, 2), mode="valid") - mu1B_mu2

        sigma1A_sq[sigma1A_sq < 0] = 0
        sigma1B_sq[sigma1B_sq < 0] = 0
        sigma2_sq[sigma2_sq < 0] = 0

        gA = sigma12_A / (sigma1A_sq + eps)
        gB = sigma12_B / (sigma1B_sq + eps)
        sv_sq_A = sigma2_sq - gA * sigma12_A
        sv_sq_B = sigma2_sq - gB * sigma12_B

        gA[sigma1A_sq < eps] = 0
        gB[sigma1B_sq < eps] = 0
        sv_sq_A[sigma1A_sq < eps] = sigma2_sq[sigma1A_sq < eps]
        sv_sq_B[sigma1B_sq < eps] = sigma2_sq[sigma1B_sq < eps]
        sigma1A_sq[sigma1A_sq < eps] = 0
        sigma1B_sq[sigma1B_sq < eps] = 0

        gA[sigma2_sq < eps] = 0
        gB[sigma2_sq < eps] = 0
        sv_sq_A[sigma2_sq < eps] = 0
        sv_sq_B[sigma2_sq < eps] = 0

        sv_sq_A[gA < 0] = sigma2_sq[gA < 0]
        sv_sq_B[gB < 0] = sigma2_sq[gB < 0]
        gA[gA < 0] = 0
        gB[gB < 0] = 0
        sv_sq_A[sv_sq_A <= eps] = eps
        sv_sq_B[sv_sq_B <= eps] = eps

        numA += np.sum(np.log10(1 + gA * gA * sigma1A_sq / (sv_sq_A + sigma_nsq)))
        numB += np.sum(np.log10(1 + gB * gB * sigma1B_sq / (sv_sq_B + sigma_nsq)))
        denA += np.sum(np.log10(1 + sigma1A_sq / sigma_nsq))
        denB += np.sum(np.log10(1 + sigma1B_sq / sigma_nsq))

    vifpA = numA / denA
    vifpB = numB / denB

    if np.isnan(vifpA):
        vifpA = 1
    if np.isnan(vifpB):
        vifpB = 1
    return vifpA + vifpB


def evaluate_single(source_a_path, source_b_path, fusion_result_path):

    metric_result = np.zeros((8))

    a = image_read_cv2(source_a_path, "GRAY")
    b = image_read_cv2(source_b_path, "GRAY")
    f = image_read_cv2(fusion_result_path, "GRAY")

    res = np.array(
        [
            Evaluator.EN(f),
            Evaluator.SD(f),
            Evaluator.SF(f),
            Evaluator.MI(f, a, b),
            Evaluator.SCD(f, a, b),
            Evaluator.VIFF(f, a, b),
            Evaluator.Qabf(f, a, b),
            Evaluator.SSIM(f, a, b),
        ]
    )
    metric_result = res
    print(f"result of {fusion_result_path}")
    print("\t EN\t SD\t SF\t MI\t SCD\t VIF\t Qabf\t SSIM")
    print(
        "\t"
        + str(np.round(metric_result[0], 2))
        + "\t"
        + str(np.round(metric_result[1], 2))
        + "\t"
        + str(np.round(metric_result[2], 2))
        + "\t"
        + str(np.round(metric_result[3], 2))
        + "\t"
        + str(np.round(metric_result[4], 2))
        + "\t"
        + str(np.round(metric_result[5], 2))
        + "\t"
        + str(np.round(metric_result[6], 2))
        + "\t"
        + str(np.round(metric_result[7], 2))
        + "\n"
    )


def evaluate(root_path):
    path_a = f"{root_path}/a"
    path_b = f"{root_path}/b"
    path_f = f"{root_path}/f"
    evaluation_results = {}
    metric_result = np.zeros((10))

    fileNmae = os.listdir(path_f)

    for img_name in tqdm(fileNmae, desc="Processing Images"):
        a = image_read_cv2(os.path.join(path_a, img_name), "GRAY")
        b = image_read_cv2(os.path.join(path_b, img_name), "GRAY")
        f = image_read_cv2(os.path.join(path_f, img_name), "GRAY")
        res = np.array(
            [
                Evaluator.EN(f),
                Evaluator.SD(f),
                Evaluator.SF(f),
                Evaluator.MI(f, a, b),
                Evaluator.SCD(f, a, b),
                Evaluator.VIFF(f, a, b),
                Evaluator.Qabf(f, a, b),
                Evaluator.SSIM(f, a, b),
                Evaluator.itensity(f, a, b),
                Evaluator.msssimLoss(f, a, b),
            ]
        )
        evaluation_results[img_name] = np.round(res, 2)
        metric_result += res
    print("=" * 80)

    metric_result /= len(fileNmae)
    print(f"result of {path_f}")
    print("\t EN\t SD\t SF\t MI\t SCD\t VIF\t Qabf\t SSIM\t itentisyLoss\t mssimLoss\t")
    print(
        "\t"
        + str(np.round(metric_result[0], 2))
        + "\t"
        + str(np.round(metric_result[1], 2))
        + "\t"
        + str(np.round(metric_result[2], 2))
        + "\t"
        + str(np.round(metric_result[3], 2))
        + "\t"
        + str(np.round(metric_result[4], 2))
        + "\t"
        + str(np.round(metric_result[5], 2))
        + "\t"
        + str(np.round(metric_result[6], 2))
        + "\t"
        + str(np.round(metric_result[7], 2))
        + "\t"
        + str(np.round(metric_result[8], 2))
        + "\t"
        + str(np.round(metric_result[9], 2))
        + "\n"
    )

    def ensure_even_dimensions(img):
        h, w = img.shape[:2]
        new_h = h if h % 2 == 0 else h - 1
        new_w = w if w % 2 == 0 else w - 1
        return img[:new_h, :new_w]


def cosine_similarity(x, y):
    """
    x: Tensor of shape (batch_size, feature_dim)
    y: Tensor of shape (batch_size, feature_dim)
    return: Tensor of shape (batch_size,)
    """
    x = F.normalize(x, p=2, dim=1)
    y = F.normalize(y, p=2, dim=1)
    return torch.sum(x * y, dim=1)


def pearson_correlation(x, y):
    """
    x: Tensor of shape (batch_size, feature_dim)
    y: Tensor of shape (batch_size, feature_dim)
    return: Tensor of shape (batch_size,)
    """
    x_centered = x - x.mean(dim=1, keepdim=True)
    y_centered = y - y.mean(dim=1, keepdim=True)

    cov = torch.sum(x_centered * y_centered, dim=1)
    std_x = torch.sqrt(torch.sum(x_centered**2, dim=1))
    std_y = torch.sqrt(torch.sum(y_centered**2, dim=1))

    return cov / (std_x * std_y + 1e-6)


def euclidean_distance(x, y):
    """
    x: Tensor of shape (batch_size, feature_dim)
    y: Tensor of shape (batch_size, feature_dim)
    return: Tensor of shape (batch_size,)
    """
    return torch.sqrt(torch.sum((x - y) ** 2, dim=1))


def average_similarity(x, y, metric):
    if metric == "cosine":
        similarity = cosine_similarity(x, y)
    elif metric == "pearson":
        similarity = pearson_correlation(x, y)
    elif metric == "euclidean":
        similarity = euclidean_distance(x, y)
    else:
        raise ValueError("Invalid metric specified.")

    return torch.mean(similarity).item()
