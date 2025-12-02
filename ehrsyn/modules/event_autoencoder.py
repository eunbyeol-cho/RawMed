import os
import logging
import torch
import torch.nn as nn
from ..modules import register_model, MODEL_REGISTRY

logger = logging.getLogger(__name__)


@register_model("event_autoencoder")
class EventAutoEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Model Components
        self.input2emb_model = self._build_input2emb_model()
        self.encode_model = self._build_encode_model()
        self.quantize_model = self._build_quantize_model() if self.config["quantizer"] else None
        self.decode_model = self._build_decode_model()
        self.emb2out_model = self._build_emb2out_model()

    def _build_input2emb_model(self):
        return MODEL_REGISTRY[f'{self.config["model"]}_input2emb'].build_model(self.config)

    def _build_encode_model(self):
        return MODEL_REGISTRY[self.config["encode_model"]].build_model(self.config)

    def _build_quantize_model(self):
        if self.config["quantizer"] == "residual_vector_quantizer":
            from .residual_vq import ResidualVQ
            return ResidualVQ(
                dim = self.config["emb_codebooks"],
                num_quantizers = self.config["num_quantizers"],
                codebook_size = self.config["num_codebooks"],
                decay = self.config["decay"],
                commitment_weight = self.config["commitment_cost"],
                stochastic_sample_codes = self.config["stochastic_sample_codes"],
                sample_codebook_temp = self.config["sample_codebook_temp"],
                shared_codebook = self.config["shared_codebook"],
            )     
        else:
            return MODEL_REGISTRY[self.config["quantizer"]].build_model(self.config)

    def _build_decode_model(self):
        return MODEL_REGISTRY[self.config["decode_model"]].build_model(self.config)

    def _build_emb2out_model(self):
        return MODEL_REGISTRY[f'{self.config["model"]}_emb2out'].build_model(self.config)

    @classmethod
    def build_model(cls, args):
        """Build a new model instance."""
        return cls(args)

    def get_targets(self, **kwargs):
        targets = {victim + '_ids': kwargs[victim + '_ids'] for victim in self.config["embed_list"]}
        if self.config["require_gt_time"]:
            targets["time_ids"] = kwargs['time_ids']
        return targets

    def forward(self, **kwargs):
        output_dict = {}

        # Encoding step 
        encoded = self.encode_model(self.input2emb_model(**kwargs)) 
        if self.quantize_model: # Vector quantization
            vq_output = self.quantize_model(encoded)
            output_dict.update(vq_output)
            encoded = vq_output['quantized']
        else:
            output_dict.update({"cont_latents": encoded})
        
        # Decoding step
        decoded = self.decode_model(encoded)

        # Converting embeddings to output logits
        logits = self.emb2out_model(decoded)
        output_dict.update(logits)

        return output_dict, self.get_targets(**kwargs)

    
    def encode(self, **kwargs):
        """Encode the input using the event autoencoder."""
        with torch.no_grad():
            encoded = self.encode_model(
                self.input2emb_model(**kwargs))
            vq_output = self.quantize_model(encoded)
        return vq_output['enc_indices']
        
    def decode(self, net_output):
        """Decode the encoded input."""
        self.max_event_size = self.config["max_event_size"]
        self.eoe = self.config["num_codebooks"]
        self.soe = self.config["num_codebooks"] + 1
    
        with torch.no_grad():
            
            encoded = torch.argmax(net_output["code_logits"], axis=-1).detach()
            batch_size = len(encoded)

            # Decoding step
            if self.config["quantizer"] == "residual_vector_quantizer":
                encoded = encoded.reshape(batch_size*self.max_event_size, -1, self.config["num_quantizers"])

                # Mask for non-existing vocabulary indices (e.g., SOE, EOE)
                non_existing_vocab_mask = (encoded == self.soe) | (encoded == self.eoe)
                encoded[non_existing_vocab_mask] = 0

                decoded = self.decode_model(self.quantize_model.get_codes_from_indices(encoded).sum(axis=0))

            else:
                encoded = encoded.reshape(batch_size*self.max_event_size, -1)

                # Mask for non-existing vocabulary indices (e.g., SOE, EOE)
                non_existing_vocab_mask = (encoded == self.soe) | (encoded == self.eoe)
                encoded[non_existing_vocab_mask] = 0
                
                decoded = self.decode_model(self.quantize_model._embedding(encoded))
            logits = self.emb2out_model(decoded)  
        
        net_output.update(logits)
        net_output.update({"enc_indices":encoded})
        return net_output
    