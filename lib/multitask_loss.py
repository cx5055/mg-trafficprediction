import torch
import torch.nn as nn
class MultiTask_AngleLoss(nn.Module):
    def __init__(self, lambda_consist: float = 0.5, eps: float = 1e-8):
        super().__init__()
        self.lambda_consist = lambda_consist
        self.eps = eps

    def forward(self, X_hat: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        X_hat_total = X_hat[0]
        X_total = X[0]
        L_total = self._angle_loss(X_hat_total, X_total)

        L_k = 0.0
        for k in range(1, len(X_hat)):
            L_k += self._angle_loss(X_hat[k], X[k])

        X_sum = X_hat[1] + X_hat[2] + X_hat[3]
        ones = torch.ones_like(X_sum)
        L_consist = self._angle_loss(ones, X_sum)

        return L_total + L_k + self.lambda_consist * L_consist

    def _angle_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_flat = x.view(x.size(0), -1)
        y_flat = y.view(y.size(0), -1)

        num = (x_flat * y_flat).sum(dim=1)
        den = torch.norm(x_flat, p=2, dim=1) * torch.norm(y_flat, p=2, dim=1) + self.eps
        cos = num / den
        return (1 - cos).mean()
