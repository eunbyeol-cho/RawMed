import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
import logging
import pickle
from torch.nn.functional import one_hot
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class LatentEHRDataset(Dataset):
    def __init__(self, data, config):
        super().__init__()
        
        self.code_ids = data["code_ids"]
        self.time_ids = data["time_ids"]

        self.num_codebooks = config["num_codebooks"]
        self.max_event_size = config["max_event_size"]
        self.time_data_type = config["time_data_type"]
        self.time_len = config["time_len"]

    def __len__(self):
        return len(self.code_ids)

    def __getitem__(self, idx):
        code_ids = self.code_ids[idx]
        time_ids = self.time_ids[idx]
        
        data = {
            "code_ids": torch.LongTensor(code_ids).view(-1),
            "time_ids": torch.LongTensor(time_ids).view(-1),
            }
        return data


    def collate_fn(self, samples):
        collated_input = dict()
        out = dict()

        for k in samples[0].keys():
            if k == "time_ids":
                pad_token_value = self.num_codebooks
                end_token_value = self.num_codebooks + 1 
                
                time_max_length = self.max_event_size * self.time_len
            
                time_padded = [F.pad(torch.cat((s[k]+self.num_codebooks+3, torch.tensor([end_token_value]))), 
                                    (0, time_max_length - len(s[k]) - 1), 
                                    mode='constant', value=pad_token_value) for s in samples]
                collated_input[k] = torch.stack(time_padded)

            else:
                collated_input[k] = torch.stack([s[k] for s in samples])

        out['net_input'] = collated_input
        return out



def ehrtimelatent_data_loader(config):
    """
    Main function to load EHR data and create data loaders.
    """
    code_ids = np.load(os.path.join(config['input_path'], f"{config['ehr']}_{config['structure']}_code.npy"), allow_pickle=True)
    time_ids = np.load(os.path.join(config['real_input_path'], f"{config['ehr']}_{config['structure']}_time.npy"), allow_pickle=True)

    logger.info(f"Loaded code from {config['input_path']}")
    logger.info(f"Loaded time from {config['real_input_path']}")
    logger.info(f"code shape: {code_ids.shape}, time len: {len(time_ids)}")

    df_path = os.path.join(config['real_input_path'], f"{config['ehr']}_split.csv")
    df = pd.read_csv(df_path)

    data_loaders = {}
    seed_column = f"seed{config['seed']}"

    for split in ["train", "valid", "test", "total"]:
        if len(df) > len(code_ids):
            indices = np.arange(len(code_ids))
        else:
            indices = np.arange(len(df)) if split == "total" else np.where(df[seed_column] == split)[0]

        data = {
            "code_ids": code_ids[indices],
            "time_ids": time_ids[indices],
            }

        dataset = LatentEHRDataset(data, config)

        shuffle = split == "train"
        data_loaders[split] = DataLoader(dataset, collate_fn=dataset.collate_fn, batch_size=config['batch_size'], num_workers=0, shuffle=shuffle)

        logger.info(f"Loaded {len(data_loaders[split].dataset)} {split} samples")

    return data_loaders
