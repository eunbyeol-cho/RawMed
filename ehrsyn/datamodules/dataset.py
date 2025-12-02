import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import ConcatDataset, RandomSampler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os
import logging
import pickle
from torch.nn.functional import one_hot
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class BaseEHRDataset(Dataset):
    def __init__(self, data, config):
        super(BaseEHRDataset, self).__init__()
        
        self.embed_list = config["embed_list"]
        self.max_event_size = config["max_event_size"]
        self.time_len = config["time_len"]
        self.use_time = ("time" in self.embed_list) or config["require_gt_time"]
    
        # Dynamically assign attributes based on embed_list
        for embed in self.embed_list:
            embed_key = embed + "_ids"
            setattr(self, embed_key, data.get(embed_key, None))

        if self.use_time:
            self.time_ids = data["time_ids"]

        if "indep" in config["exp_name"] and not config["test_only"]:
            self._reshape_for_event(config["max_event_token_len"])

    def _reshape_for_event(self, event_len):
        for embed in self.embed_list:
            embed_key = embed + "_ids"
            attr = getattr(self, embed_key, None)
            if attr is not None:
                reshaped_attr = attr.reshape(-1, event_len)
                setattr(self, embed_key, reshaped_attr)

        pad_event_mask = self.input_ids[:, 0] == 0 if hasattr(self, 'input_ids') else None

        if pad_event_mask is not None:
            for embed in self.embed_list:
                embed_key = embed + "_ids"
                attr = getattr(self, embed_key, None)
                if attr is not None:
                    setattr(self, embed_key, attr[~pad_event_mask])

        reduced_size = len(self.input_ids) if hasattr(self, 'input_ids') else 0
        original_size = len(pad_event_mask) if pad_event_mask is not None else 0
        print(f"Reduction in size: {original_size} - {reduced_size} = {original_size - reduced_size} events")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        raise NotImplementedError()
    
    def collate_fn(self, samples):
        if not samples:
            raise ValueError("No samples to collate")

        collated_input = {}
        out = {}

        for k in samples[0].keys():
            if k == "time_ids":
                time_padded = [F.pad(s[k], (0, self.max_event_size - len(s[k])), mode='constant', value=-100) for s in samples]
                collated_input[k] = torch.stack(time_padded)
            else:
                collated_input[k] = torch.stack([s[k] for s in samples])

        out['net_input'] = collated_input
        return out

    
class HiEHRDataset(BaseEHRDataset):
    def __init__(self, data, config):
        super().__init__(data, config)

    def __getitem__(self, idx):
        data = {}
        for embed in self.embed_list:
            embed_key = embed + '_ids'
            if hasattr(self, embed_key):
                data[embed_key] = torch.LongTensor(getattr(self, embed_key)[idx])

        if self.use_time:
            data["time_ids"] = torch.LongTensor(self.time_ids[idx])

        return data


def load_data_arrays(config):
    # Determine the input path based on the configuration
    input_path = config["input_path"]
    
    # Helper function to load numpy arrays
    def load_array(path, file_suffix):
        try:
            return np.load(os.path.join(path, f"{config['ehr']}_{config['structure']}_{file_suffix}.npy"), allow_pickle=True)
        except FileNotFoundError as e:
            logging.error(f"File not found: {e.filename}. Attempting to load from fallback path.")
            raise
    
    # Load arrays based on the configuration's embed_list
    data = {}
    
    embed_keys = {
        "input": "input_reduced",
        "type": "type",
        "dpe": "dpe",
        "time": "num_time",
    }
        
    for embed in config['embed_list']:
        if embed in embed_keys:
            key = embed_keys[embed]
            data[embed + '_ids'] = load_array(input_path, key)
    
    if config["require_gt_time"]:
        embed = "time"
        key = embed_keys[embed]
        data[embed + '_ids'] = load_array(input_path, key)

    logging.info(f"Loaded data arrays from {embed_keys['input']}-{input_path}")
    
    return data

def ehrsyn_data_loader(config):
    data = load_data_arrays(config)

    file_name = f"{config['ehr']}_split"
    if not config["test_only"] and config['input_path'] != config["real_input_path"]:
        file_name += config['file_suffix']
    
    path_prefix = config['real_input_path'] if config["test_only"] or config['input_path'] == config["real_input_path"] else config['input_path']
    df_path = os.path.join(path_prefix, f"{file_name}.csv")
    
    # Load and process the DataFrame
    df = pd.read_csv(df_path)
    logger.info(f"Loaded dataframe from {df_path}")
            
    # Create and return data loaders
    return create_data_loaders(df, data, config)


def create_data_loaders(df, data, config):
    data_loaders = {}
    seed_column = f"seed{config['seed']}"

    possible_splits = ["train", "valid", "test", "total"]
    
    for split in possible_splits:
        indices = np.arange(len(df)) if split == "total" else np.where(df[seed_column] == split)[0]

        split_data = {embed: data[embed][indices] for embed in data.keys()}

        dataset = HiEHRDataset(split_data, config)
        shuffle = split == "train"
        data_loaders[split] = DataLoader(dataset, collate_fn=dataset.collate_fn, batch_size=config['batch_size'], num_workers=0, shuffle=shuffle)

        logger.info(f"Loaded {len(data_loaders[split].dataset)} {split} samples")

    return data_loaders
