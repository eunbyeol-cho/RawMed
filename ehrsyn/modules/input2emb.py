import math
import torch.nn as nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from torch.nn import TransformerDecoderLayer, TransformerDecoder
import torch
import torch.nn.functional as F
from ..modules import register_model


@register_model("event_autoencoder_input2emb")
class EventAutoEncoderInput2Emb(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config 
        self.input_index_size = config["input_index_size"]
        self.type_index_size = config["type_index_size"]
        self.dpe_index_size = config["dpe_index_size"]
        self.embed_list = config['embed_list']
        self.embed_dim = config['embed_dim']
        self.max_event_token_len = config["max_event_token_len"]

        self.input_ids_embedding = nn.Embedding(self.input_index_size, self.embed_dim, padding_idx=0)
        self.type_ids_embedding = nn.Embedding(self.type_index_size, self.embed_dim, padding_idx=0) if "type" in self.embed_list else None
        self.dpe_ids_embedding = nn.Embedding(self.dpe_index_size, self.embed_dim, padding_idx=0) if "dpe" in self.embed_list else None

    @classmethod
    def build_model(cls, args):
        """Build a new model instance."""
        return cls(args)

    def forward(self, input_ids, type_ids, dpe_ids, **kwargs):
        if ("indep" in self.config["exp_name"]) and not self.training:
            input_ids = input_ids.view(-1, self.max_event_token_len)
            type_ids = type_ids.view(-1, self.max_event_token_len)
            dpe_ids = dpe_ids.view(-1, self.max_event_token_len)
        
        x = self.input_ids_embedding(input_ids)
        if self.type_ids_embedding:
            x += self.type_ids_embedding(type_ids) 
        if self.dpe_ids_embedding:
            x += self.dpe_ids_embedding(dpe_ids)
        return x
        

@register_model("classifier_input2emb")
class ClassifierInput2Emb(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config 
        self.input_index_size = config["input_index_size"]
        self.type_index_size = config["type_index_size"]
        self.dpe_index_size = config["dpe_index_size"]
        self.embed_list = config['embed_list']
        self.embed_dim = config['embed_dim']
        self.max_len = config["max_event_token_len"]
        self.max_event_token_len = config["max_event_token_len"]

        self.input_ids_embedding = nn.Embedding(self.input_index_size, self.embed_dim, padding_idx=0)
        self.type_ids_embedding = nn.Embedding(self.type_index_size, self.embed_dim, padding_idx=0) if "type" in self.embed_list else None
        self.dpe_ids_embedding = nn.Embedding(self.dpe_index_size, self.embed_dim, padding_idx=0) if "dpe" in self.embed_list else None
        self.pos_encoder = PositionalEncoding(self.embed_dim, self.config["dropout"], self.max_len)
        self.layer_norm = nn.LayerNorm(self.embed_dim, eps=1e-12)

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