import math
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from utils.train_utils import get_task

class GenHPF(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Load the configuration dynamically
        self.config = config
        
        # Model parameters from the configuration
        self.input_index_size = self.config["input_index_size"]
        self.embed_dim = self.config["embed_dim"]
        self.pred_dim = self.config["pred_dim"]
        self.max_len = self.config["max_event_token_len"]
        self.embed_list = self.config["embed_list"]
        self.dropout = self.config["dropout"]
        self.n_heads = self.config["n_heads"]
        self.n_layers = self.config["n_layers"]
        self.max_event_size = self.config["max_event_size"]
        self.pred_tasks = [get_task(task) for task in self.config["pred_tasks"]]

        # Embedding layers
        self.input_ids_embedding = nn.Embedding(
            self.input_index_size, self.embed_dim, padding_idx=0
        )
        self.pos_encoder = PositionalEncoding(self.embed_dim, self.dropout, self.max_len)
        self.layer_norm = nn.LayerNorm(self.embed_dim, eps=1e-12)

        # Text Encoder
        encoder_layers = TransformerEncoderLayer(
            self.embed_dim, self.n_heads, self.embed_dim * 4, self.dropout, batch_first=True
        )
        self.text_encoder = TransformerEncoder(encoder_layers, self.n_layers)
        self.post_encode_proj = nn.Linear(self.embed_dim, self.pred_dim)

        # Event Encoder
        self.event_pos_encoder = PositionalEncoding(
            self.pred_dim, self.dropout, self.max_event_size
        )
        self.event_layer_norm = nn.LayerNorm(self.pred_dim, eps=1e-12)

        event_encoder_layers = TransformerEncoderLayer(
            self.pred_dim, self.n_heads, self.pred_dim * 4, self.dropout, batch_first=True
        )
        self.event_encoder = TransformerEncoder(event_encoder_layers, self.n_layers)

        # Prediction Heads
        self.final_proj = nn.ModuleDict({
            task.name: nn.Linear(self.pred_dim, task.num_classes) for task in self.pred_tasks
        })

    def forward(self, input_ids):
        B, S = input_ids.shape[0], input_ids.shape[1] # time: hi - (B, S, 1), fl - (B, S, 1).
        
        # Handle rows with all zeros (padding)
        rows_all_zero = (input_ids == 0).all(dim=2)
        true_indices = torch.nonzero(rows_all_zero, as_tuple=True)
        input_ids[true_indices[0], true_indices[1], 0] = 1
        
        # Embedding
        x = self.input_ids_embedding(input_ids)        
        x = x.view(B*S, -1, self.embed_dim)
        x = self.pos_encoder(x) # (B, S, W, E) -> (B*S, W, E)
        x = self.layer_norm(x)

        # Text Encoding
        src_pad_mask = (input_ids.view(B * S, -1).eq(0).to(x.device)) 
        encoder_output = self.text_encoder(x, src_key_padding_mask=src_pad_mask)
        
        encoder_output[src_pad_mask] = 0
        encoder_output = torch.div(
            encoder_output.sum(dim=1),
            (encoder_output != 0).sum(dim=1)
        )
        events = self.post_encode_proj(encoder_output).view(B, -1, self.pred_dim) # (B, S, e)
        
        # Event Encoding
        src_pad_mask = input_ids[:, :, 1].eq(0).to(events.device)
        events = self.event_layer_norm(self.event_pos_encoder(events))
        x = self.event_encoder(events, mask=None, src_key_padding_mask=src_pad_mask)

        # Prediciton Head
        mask = ~input_ids[:, :, 1].eq(0)
        mask = mask.unsqueeze(dim=2).to(x.device).expand(B, S, self.pred_dim)
        x = (x * mask).sum(dim=1) / mask.sum(dim=1) # (B, W)

        preds = {k: layer(x) for k, layer in self.final_proj.items()}
        return preds

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)
