import torch
import torch.nn as nn
from .base_criterion import BaseCriterion


class ReconCriterion(BaseCriterion):
    def __init__(self, _config):
        super().__init__(_config)
        self.test_only = _config["test_only"]
        self.embed_list = _config["embed_list"].copy()
        self.structure = _config["structure"]
        self.vq = True if "VQ" in _config["exp_name"] else False
        self.dpe_ignore_index = _config['dpe_ignore_index']
        self.reset()

    def reset(self):
        self.acc = {'input': 0, 'type': 0, 'dpe': 0}
        self.loss = {'total': 0, 'input': 0, 'type': 0, 'dpe': 0}

        if self.vq:
            self.loss.update({'vq': 0})

    def __call__(self, criterion, preds, targets):
        if criterion == 'loss':
            iter_log = self.compute_loss(preds, targets)
            for loss_type in self.loss.keys():
                self.loss[loss_type] += iter_log[f'{loss_type}_loss']
        elif criterion == 'acc':
            iter_log = self.compute_acc(preds, targets)
            for acc_type in self.acc.keys():
                self.acc[acc_type] += iter_log[f'{acc_type}_acc']

        return iter_log

    def get_epoch_dict(self, total_iter):
        """
        Aggregates metrics over an epoch and resets for the next epoch.
        """
        epoch_log = {f'{metric}_loss': value / total_iter for metric, value in self.loss.items()}
        epoch_log.update({f'{metric}_acc': value / total_iter for metric, value in self.acc.items()})

        self.reset()
        return epoch_log

    def compute_loss(self, preds: torch.Tensor, targets: torch.Tensor) -> dict:
        device = preds['input_logits'].device
        loss_dict = {}
        
        if len(targets['input_ids'].shape) == 3:
            B, M, N = targets['input_ids'].shape
            input_loss = self.CE_loss(preds['input_logits'].permute(0,2,1), targets['input_ids'].view(B*M, N).to(device))
            type_loss = self.CE_loss(preds['type_logits'].permute(0,2,1), targets['type_ids'].view(B*M, N).to(device))
            dpe_loss = self.CE_loss(preds['dpe_logits'].permute(0,2,1), targets['dpe_ids'].view(B*M, N).to(device))
        else:
            input_loss = self.CE_loss(preds['input_logits'].permute(0,2,1), targets['input_ids'].to(device))
            type_loss = self.CE_loss(preds['type_logits'].permute(0,2,1), targets['type_ids'].to(device))
            dpe_loss = self.CE_loss(preds['dpe_logits'].permute(0,2,1), targets['dpe_ids'].to(device))
                
        ce_loss = input_loss + type_loss + dpe_loss
        loss_dict.update({'input_loss': input_loss, 'type_loss': type_loss, 'dpe_loss': dpe_loss})

        vq_loss = 0
        if "vq_loss" in preds:
            vq_loss = preds["vq_loss"]
            loss_dict['vq_loss'] = vq_loss

        loss = ce_loss + vq_loss
        loss_dict["total_loss"] = loss
        return loss_dict

    def compute_acc(self, preds, targets):
        device = preds['input_logits'].device

        if len(targets['input_ids'].shape) == 3:
            B, M, N = targets['input_ids'].shape
            targets['input_ids'] = targets['input_ids'].to(device).view(B*M, N)
            targets['type_ids'] = targets['type_ids'].to(device).view(B*M, N)
            targets['dpe_ids'] = targets['dpe_ids'].to(device).view(B*M, N)
        else:
            targets['input_ids'] = targets['input_ids'].to(device)
            targets['type_ids'] = targets['type_ids'].to(device)
            targets['dpe_ids'] = targets['dpe_ids'].to(device)

        embed2acc = {}
        for embed in self.embed_list:
            pred = torch.argmax(preds[f'{embed}_logits'], dim=-1)
            pad_mask = targets[f'{embed}_ids'] == 0

            correct = (targets[f'{embed}_ids'][~pad_mask] == pred[~pad_mask]).sum().detach().cpu()
            acc = int(correct)/(len(targets[f'{embed}_ids'][~pad_mask])) 

            embed2acc.update({f'{embed}_acc':acc})

        return embed2acc

    @property
    def compare(self):
        return 'decrease' if 'loss' in self.update_target else 'increase'

    @property
    def update_target(self):
        return 'input_acc'