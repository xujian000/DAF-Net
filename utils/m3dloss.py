import torch
import torch.nn as nn


class RBF(nn.Module):
    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (
            torch.arange(n_kernels) - n_kernels // 2
        )
        self.bandwidth_multipliers = self.bandwidth_multipliers.cuda()
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples**2 - n_samples)

        return self.bandwidth

    def forward(self, X):
        # 展平输入
        X_flat = X.view(X.size(0), -1)
        L2_distances = torch.cdist(X_flat, X_flat) ** 2
        rbf_matrix = torch.exp(
            -L2_distances[None, ...]
            / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[
                :, None, None
            ]
        )
        return rbf_matrix.sum(dim=0)


class PoliKernel(nn.Module):
    def __init__(self, constant_term=1, degree=2):
        super().__init__()
        self.constant_term = constant_term
        self.degree = degree

    def forward(self, X):
        # 展平输入
        X_flat = X.view(X.size(0), -1)
        K = (torch.matmul(X_flat, X_flat.t()) + self.constant_term) ** self.degree
        return K


class LinearKernel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        # 展平输入
        X_flat = X.view(X.size(0), -1)
        K = torch.matmul(X_flat, X_flat.t())
        return K


class LaplaceKernel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gammas = torch.FloatTensor([0.1, 1, 5]).cuda()

    def forward(self, X):
        # 展平输入
        X_flat = X.view(X.size(0), -1)
        L2_distances = torch.cdist(X_flat, X_flat) ** 2
        laplace_matrix = torch.exp(
            -L2_distances[None, ...] * (self.gammas)[:, None, None]
        )
        return laplace_matrix.sum(dim=0)


class M3DLoss(nn.Module):
    def __init__(self, kernel_type):
        super().__init__()
        if kernel_type == "gaussian":
            self.kernel = RBF()
        elif kernel_type == "linear":
            self.kernel = LinearKernel()
        elif kernel_type == "polinominal":
            self.kernel = PoliKernel()
        elif kernel_type == "laplace":
            self.kernel = LaplaceKernel()

    def forward(self, X, Y):
        # 展平输入
        X_flat = X.view(X.size(0), -1)
        Y_flat = Y.view(Y.size(0), -1)

        # 计算核矩阵
        K = self.kernel(torch.cat([X_flat, Y_flat], dim=0))

        X_size = X.size(0)
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()

        return XX - 2 * XY + YY
