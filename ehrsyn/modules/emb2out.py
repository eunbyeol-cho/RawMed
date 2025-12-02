import math
import torch.nn as nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from torch.nn import TransformerDecoderLayer, TransformerDecoder
import torch
import torch.nn.functional as F
from ..modules import register_model


@register_model("event_autoencoder_emb2out")
class EventAutoEncoderEmb2Out(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config 
        self.input_index_size = config["input_index_size"]
        self.type_index_size = config["type_index_size"]
        self.dpe_index_size = config["dpe_index_size"]
        self.embed_list = config['embed_list']
        self.embed_dim = config['embed_dim']

        self.input_ids_out = nn.Linear(self.embed_dim, self.input_index_size)
        self.type_ids_out = nn.Linear(self.embed_dim, self.type_index_size) if 'type' in config["embed_list"] else None
        self.dpe_ids_out = nn.Linear(self.embed_dim, self.dpe_index_size) if 'dpe' in config["embed_list"] else None

    @classmethod
    def build_model(cls, args):
        """Build a new model instance."""
        return cls(args)

    def forward(self, y, **kwargs):
        input_logits = self.input_ids_out(y)
        type_logits = self.type_ids_out(y) if self.type_ids_out else None
        dpe_logits = self.dpe_ids_out(y) if self.dpe_ids_out else None
        return {
            "input_logits": input_logits,
            "type_logits": type_logits,
            "dpe_logits": dpe_logits,
        }
        

@register_model("classifier_emb2out")
class ClassifierEmb2Out(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config 
        self.input_index_size = config["input_index_size"]
        self.type_index_size = config["type_index_size"]
        self.dpe_index_size = config["dpe_index_size"]
        self.embed_list = config['embed_list']
        self.embed_dim = config['embed_dim']
        self.max_len = config["max_event_token_len"]

        self.input_ids_out = nn.Linear(self.embed_dim, self.input_index_size)
        self.type_ids_out = nn.Linear(self.embed_dim, self.type_index_size) if 'type' in config["embed_list"] else None
        self.dpe_ids_out = nn.Linear(self.embed_dim, self.dpe_index_size) if 'dpe' in config["embed_list"] else None

    @classmethod
    def build_model(cls, args):
        """Build a new model instance."""
        return cls(args)

    def forward(self, input_ids, type_ids, dpe_ids, **kwargs):
        return
    

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