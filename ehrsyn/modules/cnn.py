import math
import torch.nn as nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from torch.nn import TransformerDecoderLayer, TransformerDecoder
import torch
import torch.nn.functional as F
from ..modules import register_model


class CNNModule(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.max_event_size = args["max_event_size"]
        self.max_seq_len = args["max_event_token_len"]
        self.embed_dim = args["embed_dim"]
        self.spatial_dim = args["spatial_dim"]
        latent_dim = args["latent_dim"]
        self.drop_last_activation = args["drop_last_activation"]
        self.channel_dim = latent_dim // self.spatial_dim
        
        self.diff_spatial = int(math.log(self.max_seq_len,2) - math.log(self.spatial_dim,2)) 
        self.diff_channel = int(math.log(self.embed_dim,2) - math.log(self.channel_dim,2))

        self.n_layers = max(self.diff_spatial, self.diff_channel)
        self.conv1x1_index = []


@register_model("cnn_encoder")
class CNNEncoder(CNNModule):
    def __init__(self, args):
        super().__init__(args)

        if self.diff_spatial < self.diff_channel:
            rep = self.diff_channel - self.diff_spatial
            self.channel = [self.embed_dim // (2**i) for i in range(self.diff_channel+1)]

            # Compute conv1x1 index
            self.conv1x1_index = [len(self.channel) - 2*i - 2 for i in range(rep)]
            if any(c < 0 for c in self.conv1x1_index):
                unused_indices = set(range(self.n_layers)) - set(self.conv1x1_index)
                self.conv1x1_index = [c if c >= 0 else unused_indices.pop() for c in self.conv1x1_index]

        else:
            all_rep, more_rep = divmod(self.n_layers + 1, self.diff_channel + 1)
            raw_channel = [self.embed_dim // (2**i) for i in range(self.diff_channel+1)]
            more_rep_channel = raw_channel[-more_rep:] if more_rep != 0 else []
            rep_channel = sorted(raw_channel * all_rep, reverse=True)
            self.channel = sorted(rep_channel + more_rep_channel, reverse=True)

        # Create convolutional layers
        downsample_layers = []
        for i in range(self.n_layers):
            if i in self.conv1x1_index:
                if (i == (self.n_layers-1)) and (self.drop_last_activation):
                    downsample_layers += [nn.Sequential(
                        nn.Conv1d(self.channel[i], self.channel[i+1], 1, stride=1, bias = False),
                        nn.BatchNorm1d(self.channel[i+1]),
                        )
                    ]
                else:
                    downsample_layers += [nn.Sequential(
                        nn.Conv1d(self.channel[i], self.channel[i+1], 1, stride=1, bias = False),
                        nn.BatchNorm1d(self.channel[i+1]),
                        nn.ReLU()
                        )
                    ]
            else: 
                if (i == (self.n_layers-1)) and (self.drop_last_activation):
                    downsample_layers += [nn.Sequential(
                        nn.Conv1d(self.channel[i], self.channel[i+1], 5, stride=2, padding=2, bias = False),
                        nn.BatchNorm1d(self.channel[i+1]),
                        )
                    ]
                else:
                    downsample_layers += [nn.Sequential(
                        nn.Conv1d(self.channel[i], self.channel[i+1], 5, stride=2, padding=2, bias = False),
                        nn.BatchNorm1d(self.channel[i+1]),
                        nn.ReLU()
                        )
                    ]
        self.encoder = nn.Sequential(*downsample_layers)

        # Confirm desired output size
        input = torch.randn(1, self.max_seq_len, self.embed_dim).permute(0, 2, 1)
        output_size = tuple(self.encoder(input).permute(0, 2, 1).size())
        assert output_size == (1, self.spatial_dim, self.channel_dim)

    @classmethod
    def build_model(cls, args):
        """Build a new model instance."""
        return cls(args)

    def forward(self, x):   
        return self.encoder(x.permute(0,2,1)).permute(0,2,1)


@register_model("cnn_decoder")
class CNNDecoder(CNNModule):
    def __init__(self, args):
        super().__init__(args)

        if self.diff_spatial < self.diff_channel:
            rep = self.diff_channel - self.diff_spatial
            self.channel = [self.embed_dim // (2**i) for i in range(self.diff_channel+1)]

            # Compute conv1x1 index
            self.conv1x1_index = [2*i for i in range(rep)]
            if any(c >= self.n_layers for c in self.conv1x1_index):
                unused_indices = set(range(self.n_layers)) - set(self.conv1x1_index)
                self.conv1x1_index = [c if c < self.n_layers else unused_indices.pop() for c in self.conv1x1_index]

        else:
            all_rep, more_rep = divmod(self.n_layers + 1, self.diff_channel + 1)
            raw_channel = [self.embed_dim // (2**i) for i in range(self.diff_channel+1)]
            more_rep_channel = raw_channel[-more_rep:] if more_rep != 0 else []
            rep_channel = sorted(raw_channel * all_rep, reverse=True)
            self.channel = sorted(rep_channel + more_rep_channel, reverse=True)

        # Create deconvolutional layers
        upsample_layers = []
        for i in range(self.n_layers-1, -1, -1):
            if i in self.conv1x1_index:
                upsample_layers += [nn.Sequential(
                    nn.ConvTranspose1d(self.channel[i+1], self.channel[i], 1, stride=1, bias=False),
                    nn.BatchNorm1d(self.channel[i]),
                    nn.ReLU()
                    )
                ]
            else: 
                upsample_layers += [nn.Sequential(
                    nn.ConvTranspose1d(self.channel[i+1], self.channel[i], 6, stride=2, padding=2, bias=False),
                    nn.BatchNorm1d(self.channel[i]),
                    nn.ReLU()
                    )
                ]

        self.decoder = nn.Sequential(*upsample_layers)

        # Confirm desired output size
        input = torch.randn(1, self.spatial_dim, self.channel_dim).permute(0, 2, 1)
        output_size = tuple(self.decoder(input).permute(0, 2, 1).size())
        assert output_size == (1, self.max_seq_len, self.embed_dim)

    @classmethod
    def build_model(cls, args):
        """Build a new model instance."""
        return cls(args)

    def forward(self, x):
        return self.decoder(x.permute(0,2,1)).permute(0,2,1)


@register_model("aggregate_cnn_encoder")
class AggregateCNNEncoder(CNNModule):
    def __init__(self, args):
        super().__init__(args)
        self.encoder = CNNEncoder(args)

    @classmethod
    def build_model(cls, args):
        """Build a new model instance."""
        return cls(args)

    def forward(self, x):
        B, M, N, E = x.shape
        z = self.encoder(x.view(B, M*N, E))
        return z.reshape(B, -1, E)


@register_model("aggregate_cnn_decoder")
class AggregateCNNDecoder(CNNModule):
    def __init__(self, args):
        super().__init__(args)
        self.decoder = CNNDecoder(args)

    @classmethod
    def build_model(cls, args):
        """Build a new model instance."""
        return cls(args)

    def forward(self, z):
        B, _, E = z.shape 
        x = self.decoder(z)
        return x.reshape(B*self.max_event_size, -1, E)