import torch
import torch.nn as nn


class BaseCriterion():
    def __init__(self, _config):
        self.CE_loss = nn.CrossEntropyLoss(ignore_index=0)
        self.MSE_loss = nn.MSELoss()
        self.MAE_loss = nn.L1Loss()

    @property
    def compare(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def __call__(self):
        raise NotImplementedError