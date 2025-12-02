import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_criterion import BaseCriterion
from itertools import groupby, count


class ARCriterion(BaseCriterion):
    def __init__(self, _config):
        super().__init__(_config)
        self.test_only = _config["test_only"]
        self.structure = _config["structure"]
        self.max_event_size = _config["max_event_size"]
        self.num_codebooks = _config["num_codebooks"]
        self.pad_token_id = _config['pad_token_id']
        self.embed_list = _config["embed_list"]

        assert self.num_codebooks == self.pad_token_id

        self.custom_CE_loss = nn.CrossEntropyLoss(ignore_index=self.pad_token_id)
        self.reset()


    def reset(self):
        self.acc = {'code': 0, 'time': 0, 'total': 0}
        self.loss = {'code': 0, 'time': 0, 'total': 0}

        
    def __call__(self, criterion, preds, targets, disc=None):
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
    
    def compute_loss(self, preds: torch.Tensor, targets: torch.Tensor, disc=None) -> dict:
        loss_dict = {}

        for embed in self.embed_list:
            loss_dict[f'{embed}_loss'] = self.custom_CE_loss(preds[f'{embed}_logits'].permute(0,2,1), targets[f'{embed}_ids'].to(preds[f'{embed}_logits'].device))            

        loss_dict["total_loss"] = sum(loss_dict.values())
        return loss_dict
    
        
    def compute_acc(self, preds, targets):
        embed2acc = {}
        total_correct = 0
        total_tokens = 0

        for embed in self.embed_list:
            targets[f'{embed}_ids'] = targets[f'{embed}_ids'].to(preds[f'{embed}_logits'].device)

            non_pad_mask = targets[f'{embed}_ids'] != self.pad_token_id

            predicted_ids = torch.argmax(preds[f'{embed}_logits'], dim=-1)
                
            # Calculate accuracy for the current embedding
            embed2acc[f'{embed}_acc'] = self._compute_accuracy(targets[f'{embed}_ids'], predicted_ids, non_pad_mask)

            # Accumulate correct predictions and total tokens for calculating total accuracy
            masked_true_vals = targets[f'{embed}_ids'][non_pad_mask]
            correct = (masked_true_vals == predicted_ids[non_pad_mask]).sum().item()
            total = non_pad_mask.sum().item()

            total_correct += correct
            total_tokens += total

        # Calculate and store total accuracy
        total_acc = total_correct / total_tokens if total_tokens > 0 else 0
        embed2acc['total_acc'] = total_acc

        return embed2acc
    
    def _compute_accuracy(self, true_vals, pred_vals, mask):
        masked_true_vals = true_vals[mask]
        masked_pred_vals = pred_vals[mask]
        correct = (masked_true_vals == masked_pred_vals).sum()
        return correct.item() / len(masked_true_vals) if len(masked_true_vals) > 0 else 0
    
    @property
    def compare(self):
        return 'decrease' if 'loss' in self.update_target else 'increase'

    @property
    def update_target(self):
        return 'total_acc'
    