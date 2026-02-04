import numpy as np
import torch
class MinMax01Scaler:
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def transform(self, data):
        return (data - self.min) / (self.max - self.min)

    def inverse_transform(self, data):
        if isinstance(data, torch.Tensor) and isinstance(self.min, np.ndarray):
            self.min = torch.from_numpy(self.min).to(data.device).type(data.dtype)
            self.max = torch.from_numpy(self.max).to(data.device).type(data.dtype)
        return (data * (self.max - self.min) + self.min)

class Max01Scaler:
    def __init__(self, max):
        self.max = max

    def transform(self, data):
        return data / self.max

    def inverse_transform(self, data):
        if isinstance(data, torch.Tensor) and isinstance(self.max, np.ndarray):
            self.max = torch.from_numpy(self.max).to(data.device).type(data.dtype)
        return data * self.max
