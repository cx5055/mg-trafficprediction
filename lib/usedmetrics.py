import torch
def MAE_torch(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    B = pred.size(0)
    diff = torch.abs(pred - true).view(B, -1)
    return diff.mean(dim=1).mean()
def MSE_torch(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    B = pred.size(0)
    sq = ((pred - true) ** 2).view(B, -1)
    return sq.mean(dim=1).mean()
def RMSE_torch(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(MSE_torch(pred, true))
def CORR_torch(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    p = pred.view(-1)
    t = true.view(-1)
    p_mean = p.mean()
    t_mean = t.mean()
    cov = ((p - p_mean) * (t - t_mean)).mean()
    return cov / (p.std(unbiased=False) * t.std(unbiased=False) + 1e-8)

def COS_torch(pred: torch.Tensor, true: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    B = pred.size(0)
    x_flat = pred.view(B, -1)
    y_flat = true.view(B, -1)
    num = (x_flat * y_flat).sum(dim=1)
    den = torch.norm(x_flat, dim=1) * torch.norm(y_flat, dim=1) + eps
    cos_sim = num / den
    return cos_sim.mean()


def all_metrics_torch(pred: torch.Tensor, true: torch.Tensor) -> dict:
    return {
        'MAE': MAE_torch(pred, true),
        'MSE': MSE_torch(pred, true),
        'RMSE': RMSE_torch(pred, true),
        'CORR': CORR_torch(pred, true),
        'COS': COS_torch(pred, true),
    }
