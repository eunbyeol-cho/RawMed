import torch
from dataclasses import dataclass

@dataclass
class Task:
    name: str
    num_classes: int
    property: str

def get_task(pred_task):
    return {
        'creatinine': Task('creatinine', 5, 'multi-class'), 
        'platelets': Task('platelets', 5, 'multi-class'),
        'wbc': Task('wbc', 3, 'multi-class'),
        'hb': Task('hb', 4, 'multi-class'),
        'bicarbonate': Task('bicarbonate', 3, 'multi-class'),
        'sodium': Task('sodium', 3, 'multi-class'),
        'magnesium sulfate': Task('magnesium sulfate', 1, 'binary'),
        'heparin': Task('heparin', 1, 'binary'),
        'potassium chloride|kcl': Task('potassium chloride|kcl', 1, 'binary'),
        'norepinephrine': Task('norepinephrine', 1, 'binary'),
        'propofol': Task('propofol', 1, 'binary'),
        'heartrate': Task('heartrate', 1, 'binary'),
        'resprate': Task('resprate', 1, 'binary'),
        'morphine': Task('morphine', 1, 'binary'),
        'ondansetron': Task('ondansetron', 1, 'binary'),
        'detect': Task('detect', 1, 'binary'),
        'age': Task('age', 4, 'multi-class'),
        'gender': Task('gender', 1, 'binary'),
        'admission_type': Task('admission_type', 4, 'multi-class'),
    }[pred_task]

class EarlyStopping:
    def __init__(self, patience=5, mode="max", delta=0.0, save_path="best_model.pth"):
        """
        Args:
        - patience (int): Number of epochs with no improvement after which training will be stopped.
        - mode (str): "max" for maximizing the metric, "min" for minimizing.
        - delta (float): Minimum change in the monitored metric to qualify as an improvement.
        - save_path (str): Path to save the best model.
        """
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.save_path = save_path
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, score, model):
        """
        Check if training should stop based on the given score.
        Save the model if the score improves.
        """
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif (self.mode == "max" and score <= self.best_score + self.delta) or \
             (self.mode == "min" and score >= self.best_score - self.delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model)

    def save_checkpoint(self, model):
        """
        Save the current best model to the specified path.
        """
        print(f"Saving model with score {self.best_score} to {self.save_path}")
        torch.save(model.state_dict(), self.save_path)
