
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ..modules import register_model

@register_model("vector_quantizer")
class VectorQuantizerEMA(nn.Module):
    def __init__(self, _config):
        super(VectorQuantizerEMA, self).__init__()
        
        self._embedding_dim = _config["emb_codebooks"]
        self._num_embeddings = _config["num_codebooks"]
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = _config["commitment_cost"]
        
        self.register_buffer('_ema_cluster_size', torch.zeros(self._num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(self._num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()
        
        self._decay = _config["decay"]
        self._epsilon = 1e-5

    @classmethod
    def build_model(cls, args):
        """Build a new model instance."""
        return cls(args)
    
    def forward(self, inputs):
        # convert inputs from BCH -> BHC
        # inputs = inputs.permute(0, 2, 1).contiguous()
        inputs = inputs.contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
        
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)
            
            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n)
            
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
            
            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))
            
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # Historgram
        enc_ind = encoding_indices.squeeze().view(inputs.shape[0], inputs.shape[1]).cpu().detach()
        
        return {
            'embedding':self._embedding.weight,
            'vq_loss':loss,
            'quantized':quantized.contiguous(),
            'perplexity':perplexity,
            'enc_indices':enc_ind
        }
