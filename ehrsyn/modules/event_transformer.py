import os
import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from ..modules import register_model, MODEL_REGISTRY
from ..modules.event_autoencoder import EventAutoEncoder
from ..modules.input2emb import PositionalEncoding

logger = logging.getLogger(__name__)


@register_model("event_transformer")
class EventTransformer(nn.Module):
    def __init__(self, config, pretrained_path=None):
        super().__init__()

        self.config = config
        self.d_model = config["embed_dim"]
        self.dropout = config["dropout"]
        self.max_event_size = config["max_event_size"]
        self.num_codebooks = config["num_codebooks"]

        self.time_data_type = config["time_data_type"]
        self.time_len = config["time_len"]
        if self.time_data_type == "text":
            self.vocab = config["num_codebooks"] + 13 # [0,C-1]: codes, [C+3, C+12]: times
        else:
            if config["time_scale"] == 10:
                # self.vocab = config["num_codebooks"] + 76 # [0,C-1]: codes, [C+3, C+75]: times
                self.vocab = config["num_codebooks"] + 4 + config["obs_size"] * 6 # [0,C-1]: codes, [C+3, C+75]: times
            if config["time_scale"] == 30:
                self.vocab = config["num_codebooks"] + 28 # [0,C-1]: codes, [C+3, C+27]: times

        code_len = config["spatial_dim"] * config["num_quantizers"]
        self.max_len = config["max_event_size"] * (code_len+self.time_len) + 1
        # self.max_len = config["max_event_size"] * (4+self.time_len) + 1
        self.start_token = config["num_codebooks"] + 2
        self.end_token = config["num_codebooks"] + 1
        self.sample = config["sample"]

        self.lut = nn.Embedding(self.vocab, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model, self.dropout, self.max_len)
        self.transformer = self._build_transformer()
        self.proj = nn.Linear(self.d_model, self.vocab)
        
        initrange = 0.1
        self.lut.weight.data.uniform_(-initrange, initrange)
                
    def _build_transformer(self):
        encoder_layers = TransformerEncoderLayer(self.d_model, self.config["n_heads"], self.d_model*4, self.dropout, batch_first=True)
        return TransformerEncoder(encoder_layers, self.config["n_layers"])

    @classmethod
    def build_model(cls, args):
        """Build a new model instance."""
        return cls(args)

    def __call__(self, **kwargs):
        if self.sample:
            assert self.config["test_only"] == True, "generate can only be called in test-only mode"
            return self.generate(top_k=self.config["topk"], temperature=self.config["temperature"], sample=True, **kwargs)
        else:
            return self.forward(**kwargs)
        
    def get_targets(self, **kwargs):
        targets = {victim + '_ids': kwargs[victim + '_ids'] for victim in self.config["embed_list"]}
        return targets

    def forward(self, **kwargs):
        """Perform a forward pass of the model."""
        code_ids = kwargs["code_ids"]
        time_ids = kwargs["time_ids"]
        
        # Calculate batch size and event length
        batch_size = code_ids.shape[0]
        event_len = (time_ids.size(1) + code_ids.size(1)) // self.max_event_size
        
        # Reshape and concatenate time_ids and code_ids for processing
        reshaped_time_ids = time_ids.reshape(batch_size, self.max_event_size, -1)
        reshaped_code_ids = code_ids.reshape(batch_size, self.max_event_size, -1)
        input_ids = torch.cat((reshaped_time_ids, reshaped_code_ids), dim=2).reshape(batch_size, -1)[:, :-1]
        
        # Prepend the start token to input IDs
        start_tokens = torch.full((batch_size, 1), self.start_token, dtype=input_ids.dtype, device=input_ids.device)
        input_ids = torch.cat([start_tokens, input_ids], dim=1)
        
        # Embedding and positional encoding
        x = self.lut(input_ids) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        
        # Create padding mask for input IDs
        pad_mask = input_ids.eq(self.config["pad_token_id"])
        
        subsequent_mask = self._generate_square_subsequent_mask(x.size(1)).to(x.device)
        
        transformer_output = self.transformer(x, mask=subsequent_mask, src_key_padding_mask=pad_mask)
        
        # Project transformer output to the desired output size
        transformer_output = self.proj(transformer_output)
        
        # Reshape output for code and time logits
        reshaped_output = transformer_output.reshape(batch_size, self.max_event_size, event_len, -1)
        
        output_dict = {
            "code_logits": reshaped_output[:, :, self.time_len:, :].reshape(batch_size, -1, self.vocab),
            "time_logits": reshaped_output[:, :, :self.time_len, :].reshape(batch_size, -1, self.vocab)
        }
        return output_dict, self.get_targets(**kwargs)
    
        
    @torch.no_grad()
    def generate(self, top_k: int = None, temperature: float = 1.0, sample: bool = True, **kwargs) -> torch.Tensor:
        batch_size = kwargs["code_ids"].shape[0]
        event_len = (kwargs["time_ids"].size(1) +  kwargs["code_ids"].size(1)) // self.max_event_size

        device = kwargs["code_ids"].device
        sequence = torch.full((batch_size, 1), self.start_token, dtype=torch.long, device=device)
        
        for i in range(self.max_len - 1):
            x = self.lut(sequence) * math.sqrt(self.d_model)
            x = self.pos_encoder(x)

            pad_mask = sequence.eq(self.config["pad_token_id"])
            subsequent_mask = self._generate_square_subsequent_mask(x.size(1)).to(x.device)

            transformer_output = self.transformer(x, mask=subsequent_mask, src_key_padding_mask=pad_mask)
            
            logits = self.proj(transformer_output)[:, -1, :] / temperature

            logits = self._handle_time_location(logits, i, event_len)

            if top_k is not None:
                logits = self._top_k_logits(logits, top_k)

            probs = F.softmax(logits, dim=-1)
            next_word = self._get_next_word(probs, sample)
            sequence = torch.cat([sequence, next_word], dim=1)
        
        # Reshape output for code and time logits
        sequence = sequence[:, 1:] # Remove start token
        reshaped_output = sequence.reshape(batch_size, self.max_event_size, event_len)

        output_dict = {
            "code_logits": F.one_hot(reshaped_output[:, :, self.time_len:].reshape(batch_size, -1),  num_classes=self.vocab).float(),
            "time_logits": F.one_hot(reshaped_output[:, :, :self.time_len].reshape(batch_size, -1), num_classes=self.vocab).float()
        }
        return output_dict, self.get_targets(**kwargs)
    
    def _handle_time_location(self, logits, iter, event_len):
        """
        Handle time location based logits adjustments.
        [PAD]:self.num_codebooks
        [EOS]:self.num_codebooks + 1
        [SOS]:self.num_codebooks + 2
        """
        if self.time_data_type == "text":
            if iter % event_len == 0: # pad, end, time
                logits[:, :self.num_codebooks] = -float('Inf')
            elif iter % event_len == 1: # pad, time
                logits[:, :self.num_codebooks] = -float('Inf') 
                logits[:, self.num_codebooks+1:self.num_codebooks+3] = -float('Inf')
            elif (self.time_len == 3) and (iter % event_len == 2):
                logits[:, :self.num_codebooks] = -float('Inf') 
                logits[:, self.num_codebooks+1:self.num_codebooks+3] = -float('Inf')
            else: # code
                logits[:, self.num_codebooks+1:] = -float('Inf')
        else:
            if iter % event_len == 0: # pad, end, time
                logits[:, :self.num_codebooks] = -float('Inf')
            else: # code
                logits[:, self.num_codebooks+1:] = -float('Inf')
        return logits

    def _generate_square_subsequent_mask(self, sz: int):
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
    
    def _top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float('Inf')
        return out
    

    def _get_next_word(self, probs, sample):
        """Sample or get the top probability word."""
        if sample:
            next_word = torch.multinomial(probs.squeeze(0), num_samples=1)
        else:
            _, next_word = torch.topk(probs, k=1, dim=-1)
            next_word = next_word.squeeze(0)

        if next_word.dim() == 1:
            next_word = next_word.unsqueeze(1)
    
        return next_word
