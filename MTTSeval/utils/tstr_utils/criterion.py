import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
import numpy as np
from utils.train_utils import get_task

def masked_bce_with_logits_loss(preds, targets, ignore_index=-1):
    """
    Compute BCEWithLogitsLoss while ignoring targets with a specific value.
    """
    mask = (targets != ignore_index)  # Create a mask for valid targets
    masked_preds = preds[mask]
    masked_targets = targets[mask]
    
    loss_fn = nn.BCEWithLogitsLoss()
    return loss_fn(masked_preds, masked_targets)

class PredCriterion():
    def __init__(self, config):
        # self.bce = nn.BCEWithLogitsLoss()
        self.ce = nn.CrossEntropyLoss(ignore_index=-1)
        self.pred_tasks = [get_task(task) for task in config["pred_tasks"]]

        self.reset()

    def reset(self):
        self.loss = {"total": 0}
        self.metric = {task.name: {"score": [], "true": []} for task in self.pred_tasks}
    
    def __call__(self, criterion, preds, targets):
        if criterion == 'loss':
            iter_log = self.compute_loss(preds, targets)
            for loss_type in self.loss.keys():
                self.loss[loss_type] += iter_log[f'{loss_type}_loss']       
        else:
            iter_log = self.compute_auroc(preds, targets)
            for task_name in self.metric.keys():
                self.metric[task_name]["score"].append(iter_log[task_name]["score"]) 
                self.metric[task_name]["true"].append(iter_log[task_name]["true"]) 
        return iter_log

    def get_epoch_dict(self, total_iter):
        epoch_log = {}
        total_auroc = 0
        
        for loss_type in self.loss.keys():
            epoch_log[f'{loss_type}_loss'] = self.loss[loss_type] / total_iter
        for task in self.metric.keys():
            epoch_log[task] = {
                "auroc": self.auroc(task),
                "auprc": self.auprc(task),
            }
            total_auroc += self.macro_auroc(task)
        
        epoch_log["total_auroc"] = total_auroc / (len(self.pred_tasks))
                
        self.reset()
        return epoch_log


    def compute_loss(self, preds, targets):
        loss_dict = {}

        for task in self.pred_tasks:
            pred = preds[task.name]
            target = targets[task.name].to(pred.device)

            if torch.all(target == -1):
                # print(f"Skipping loss calculation for task {task.name} (All targets ignored)")
                continue
            
            if task.property == "binary":
                loss = masked_bce_with_logits_loss(pred.view(-1), target.view(-1), ignore_index=-1)

            elif task.property == "multi-class":
                values, target_labels = target.max(dim=1)
                target_labels[values == 0] = -1  # 아무 클래스도 선택되지 않은 경우 -1로 설정

                loss = self.ce(input=pred, target=target_labels)
            
            loss_dict[f"{task.name}_loss"] = loss
        
        loss_dict["total_loss"] = sum(list(loss_dict.values())) 
        return loss_dict


    def compute_auroc(self, preds, targets):
        metric = {}
        for task_name, pred in preds.items():
            metric[task_name] = {}

            target = targets[task_name].to(pred.device)
            score = torch.sigmoid(pred).detach().cpu().numpy()
            true = target.detach().cpu().numpy()

            metric[task_name]["score"] = score
            metric[task_name]["true"] = true
        return metric

    @property
    def compare(self):
        return 'decrease' if 'loss' in self.update_target else 'increase'

    @property
    def update_target(self):
        return 'total_auroc'


    def auroc(self, task_name):
        y_true = np.concatenate(self.metric[task_name]["true"])
        y_score = np.concatenate(self.metric[task_name]["score"])
        task_property = next((task.property for task in self.pred_tasks if task.name == task_name), None)
        try:
            if task_property in ["multi-class", "multi-label"]:
                missing = np.where(np.sum(y_true, axis=0) == 0)[0]
                y_true = np.delete(y_true, missing, 1)
                y_score = np.delete(y_score, missing, 1)
                return roc_auc_score(y_true=y_true, y_score=y_score, average="micro", multi_class="ovr")
            elif task_property == "binary":
                valid_indices = y_true != -1
                y_true = y_true[valid_indices]
                y_score = y_score[valid_indices]
                return roc_auc_score(y_true=y_true, y_score=y_score)

        except:
            return float("nan")

    def macro_auroc(self, task_name, average="macro"):
        y_true = np.concatenate(self.metric[task_name]["true"])
        y_score = np.concatenate(self.metric[task_name]["score"]) 
        task_property = next((task.property for task in self.pred_tasks if task.name == task_name), None)

        if task_property == "binary":
            valid_indices = y_true != -1
            y_true = y_true[valid_indices]
            y_score = y_score[valid_indices]
        
        try:
            if task_property in ["multi-class", "multi-label"]:
                missing = np.where(np.sum(y_true, axis=0) == 0)[0]
                y_true = np.delete(y_true, missing, 1)
                y_score = np.delete(y_score, missing, 1)
            return roc_auc_score(y_true=y_true, y_score=y_score, average=average) # average=None (class-wise macro auroc)
        except:
            return float("nan")

    def auprc(self, task_name):
        y_true = np.concatenate(self.metric[task_name]["true"])
        y_score = np.concatenate(self.metric[task_name]["score"])
        task_property = next((task.property for task in self.pred_tasks if task.name == task_name), None)

        if task_property == "binary":
            valid_indices = y_true != -1
            y_true = y_true[valid_indices]
            y_score = y_score[valid_indices]

        try:
            return average_precision_score(y_true=y_true, y_score=y_score, average="micro")
        except:
            return float("nan")