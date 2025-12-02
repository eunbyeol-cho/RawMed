import os
import argparse
import pandas as pd
import numpy as np
import torch
import random
import json
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
from collections import defaultdict
from utils.configs import get_config

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_item(table_names, root, train_ids, test_ids, mode=None, config=None):
    item_df_list = []
    pid_col = config['pid_column']
    time_col = config['time_column']

    for table_name in table_names:
        item_col = config['item_column'][table_name]    
        df = pd.read_csv(os.path.join(root, f"{table_name}.csv"))
        df = df.rename(columns={item_col: 'item'})
        df = df[[pid_col, time_col, 'item']]
        # print(table_name, df.head())
        if mode == 'train':
            df = df[df[pid_col].isin(train_ids)]
        elif mode == 'test':
            df = df[df[pid_col].isin(test_ids)]
        item_df_list.append(df)

    item_df = pd.concat(item_df_list).dropna()
    item_df = item_df.sort_values(by=[pid_col, time_col])
    item_grouped = item_df.groupby([pid_col, time_col])['item'].apply(set).reset_index()
    return item_grouped


class RandomTEventDataset(Dataset):
    def __init__(self, seq_data_list, t_indices=None):
        self.data = [events for events in seq_data_list if len(events) >= 2]
        if t_indices is not None:
            self.t_indices = t_indices
        else:
            self.t_indices = [None] * len(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        events = self.data[idx]
        if self.t_indices[idx] is not None:
            t = self.t_indices[idx]
        else:
            t = np.random.randint(1, len(events))
        input_seq = np.array([vec for _, vec in events[:t]])
        target = np.array(events[t][1])
        return input_seq, target, len(input_seq)

    def collate_fn(self, batch):
        batch.sort(key=lambda x: x[2], reverse=True)
        sequences, targets, lengths = zip(*batch)
        sequences = [torch.tensor(seq, dtype=torch.float32) for seq in sequences]
        padded_sequences = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
        targets = torch.tensor(np.stack(targets), dtype=torch.float32)
        lengths = torch.tensor(lengths, dtype=torch.long)
        return padded_sequences, targets, lengths


class EventPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True)
        _, (hn, _) = self.lstm(packed)
        return self.sigmoid(self.fc(hn[-1]))
 
 
def train_model(config, train_loader, valid_loader, criterion, device, model, ckpt_name):
    optimizer = torch.optim.Adam(model.parameters(), lr=config["nep_lr"])
    best_valid_f1 = 0
    patience_counter = 0
        
    for epoch in range(config["nep_epochs"]):
        print(f"\n Epoch {epoch+1}/{config['nep_epochs']}")
        model.train()
        total_loss = 0
        for batch_x, batch_y, lengths in train_loader:
            batch_x, batch_y, lengths = batch_x.to(device), batch_y.to(device), lengths.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_x, lengths), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Train Loss: {total_loss / len(train_loader):.4f}")

        # Validation
        model.eval()
        preds_all, targets_all = [], []
        with torch.no_grad():
            for batch_x, batch_y, lengths in valid_loader:
                batch_x, batch_y, lengths = batch_x.to(device), batch_y.to(device), lengths.to(device)
                outputs = model(batch_x, lengths)
                preds_all.append((outputs > config["nep_threshold"]).int().cpu().numpy())
                targets_all.append(batch_y.int().cpu().numpy())

        preds_all = np.vstack(preds_all)
        targets_all = np.vstack(targets_all)
        f1 = f1_score(targets_all, preds_all, average='micro', zero_division=0)
        print(f"Valid F1: {f1:.4f}")

        if f1 > best_valid_f1:
            best_valid_f1 = f1
            torch.save(model.state_dict(), os.path.join(config["output_data_root"], ckpt_name))
            print(f"Best model saved → {ckpt_name}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement: {patience_counter}/{config['nep_patience']} epochs")
            if patience_counter >= config['nep_patience']:
                print("Early stopping triggered.")
                break
    return best_valid_f1

def test_model(config, test_loader, model, device, ckpt_name):
    print("\n Test start")
    
    model.load_state_dict(torch.load(os.path.join(config["output_data_root"], ckpt_name)))
    model.eval()

    preds_all, targets_all = [], []
    with torch.no_grad():
        for batch_x, batch_y, lengths in test_loader:
            batch_x, batch_y, lengths = batch_x.to(device), batch_y.to(device), lengths.to(device)
            outputs = model(batch_x, lengths)
            preds_all.append((outputs > config["nep_threshold"]).int().cpu().numpy())
            targets_all.append(batch_y.int().cpu().numpy())

    preds_all = np.vstack(preds_all)
    targets_all = np.vstack(targets_all)
    test_f1 = f1_score(targets_all, preds_all, average='micro', zero_division=0)
    print(f" micro F1-score: {test_f1:.4f}")
    return test_f1
   

def get_data_loaders(config, mlb, train_ids, test_ids, seed):
    """
    Returns train_loader, valid_loader, test_loader for the given config and ids.
    """
    pid_col = config["pid_column"]
    time_col = config["time_column"]
    real_data_root = config["real_data_root"]
    syn_data_root = config["syn_data_root"]
    table_names = config["table_names"]

    # Load Syn data
    grouped = load_item(table_names, root=syn_data_root, train_ids=train_ids, test_ids=test_ids, mode='train', config=config)
    item_matrix = mlb.transform(grouped['item'])

    # Get class weights
    item_counts = np.sum(item_matrix, axis=0)
    weights = 1.0 / np.log(item_counts + 1.1 + 1e-5)
    weights /= np.mean(weights)

    # Get seq data
    grouped['item_vector'] = list(item_matrix)
    seq_data = defaultdict(list)
    for _, row in grouped.iterrows():
        seq_data[row[pid_col]].append((row[time_col], row['item_vector']))
    for sid in seq_data:
        seq_data[sid] = sorted(seq_data[sid], key=lambda x: x[0])

    # Split train/valid
    all_seq_data = list(seq_data.values())
    train_size = int(len(all_seq_data) * 0.8)
    train_data, valid_data = random_split(all_seq_data, [train_size, len(all_seq_data) - train_size], generator=torch.Generator().manual_seed(seed))
    train_dataset = RandomTEventDataset(train_data)
    valid_dataset = RandomTEventDataset(valid_data)
    
    train_loader = DataLoader(
        train_dataset, batch_size=config["nep_batch_size"], shuffle=True,
        collate_fn=train_dataset.collate_fn, num_workers=0
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=config["nep_batch_size"], shuffle=False,
        collate_fn=valid_dataset.collate_fn, num_workers=0
    )

    # Load Real for test
    test_grouped = load_item(table_names, root=real_data_root, train_ids=train_ids, test_ids=test_ids, mode='test', config=config)
    test_item_matrix = mlb.transform(test_grouped['item'])
    test_grouped['item_vector'] = list(test_item_matrix)
    test_seq_data = defaultdict(list)
    for _, row in test_grouped.iterrows():
        test_seq_data[row[pid_col]].append((row[time_col], row['item_vector']))
    for sid in test_seq_data:
        test_seq_data[sid] = sorted(test_seq_data[sid], key=lambda x: x[0])
    test_seq_list = list(test_seq_data.values())
    test_seq_list = [events for events in test_seq_list if len(events) >= 2]

    # Get t_indices
    t_indices_file = os.path.join(config["output_data_root"], f"{config['ehr']}_test_t_indices.npy")
    if os.path.exists(t_indices_file):
        t_indices = np.load(t_indices_file, allow_pickle=True)
        print(f"{t_indices_file} loaded.")
    else:
        t_indices = [np.random.randint(1, len(events)) for events in test_seq_list]
        np.save(t_indices_file, t_indices)
        print(f"t indices saved to {t_indices_file}.")

    test_dataset = RandomTEventDataset(test_seq_list, t_indices=t_indices)
    test_loader = DataLoader(
        test_dataset, batch_size=config["nep_batch_size"], shuffle=False,
        collate_fn=test_dataset.collate_fn, num_workers=0
    )
    return train_loader, valid_loader, test_loader, weights

def run_next_event_predict(config):
    pid_col = config["pid_column"]
    time_col = config["time_column"]
    real_data_root = config["real_data_root"]
    syn_data_root = config["syn_data_root"]
    table_names = config["table_names"]

    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load splits
    splits = pd.read_csv(os.path.join(config["real_data_root"], config["split_file_name"])).reset_index()
    train_ids = splits[splits[config["split_column"]] == "train"].index.unique()
    test_ids = splits[splits[config["split_column"]] == "test"].index.unique()

    # Load Real data
    real_grouped = load_item(table_names, root=real_data_root, train_ids=train_ids, test_ids=test_ids, mode='train', config=config)
    mlb = MultiLabelBinarizer()
    mlb.fit(real_grouped['item'])  

    # Get data loaders
    train_loader, valid_loader, test_loader, weights = get_data_loaders(config, mlb, train_ids, test_ids, config["seed"])

    # Get class weights   
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    criterion = nn.BCELoss(weight=class_weights)

    model = EventPredictor(len(mlb.classes_), 128, len(mlb.classes_))
    model.to(device)

    ckpt_name = f'best_model_lstm_{config["ehr"]}_seed{config["seed"]}_lr{config["nep_lr"]}_th{config["nep_threshold"]}.pt'
    
    _ = train_model(config, train_loader, valid_loader, criterion, device, model, ckpt_name)    
    test_f1 = test_model(config, test_loader, model, device, ckpt_name)
    json.dump(
        {"test_f1": test_f1}, 
        open(os.path.join(config["output_data_root"], f"nep_f1_{config['seed']}.json"), "w")
    )
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ehr", type=str, default='mimiciv')
    parser.add_argument("--obs_size", type=int, default=12)
    parser.add_argument('--real_data_root', type=str, default='data/real_data/mimiciv/')
    parser.add_argument('--syn_data_root', type=str, default='data/syn_data/mimiciv/sdv/')
    parser.add_argument('--output_data_root', type=str, default='results/mimiciv')
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
   
    # Load config using new loader
    config = get_config(
        ehr=args.ehr,
        obs_size=args.obs_size,
        real_data_root=args.real_data_root,
        syn_data_root=args.syn_data_root,
        output_data_root=args.output_data_root,
        seed=args.seed
    )

    run_next_event_predict(config)