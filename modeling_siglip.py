from typing import Optional, Tuple
import torch
import torch.nn as nn

class SiglipVisionConfig:
    """ Class for vision encoder"""

    def __init__(
        self,
        hidden_size           = 768,  #Embedding Size of vision encode.
        intermediate_size     = 3072, #Size of the linear layer in FFNN.
        num_hidden_layers     = 12,   #No. of layers in vision transformer.
        num_attention_heads   = 12,   #No. of attention in multi head attention layer of vision transformer.
        num_channels          = 3,    #No. of channels in image.
        image_size            = 224,  #Image size.it can vary.
        patch_size            = 16,   #Patch of after spllint the image
        layer_norm_eps        = 1e-6, #Hyperparameter to use in layer normalization.
        attention_dropout     = 0.0,  #Hyperparameter to use in attention.
        num_image_tokens: int = None, #Length of output embedding of image transformer.
        **kwargs
    ):
        super().__init__()

        self.hidden_size         = hidden_size
        self.intermediate_size   = intermediate_size
        self.num_hidden_layers   = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels        = num_channels
        self.patch_size          = patch_size
        self.image_size          = image_size
        self.attention_dropout   = attention_dropout
        self.layer_norm_eps      = layer_norm_eps
        self.num_image_tokens    = num_image_tokens

class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels  = config.num_channels,
            out_channels = self.embed_dim,
            kernal_size  = self.patch_size,
            stride       = self.patch_size,
            padding      = "valid", # This indicates no padding is added. 
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.positional_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).extend((1,-1)),
            persitant=False
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        _, _, height, width = pixel_values.shape
        patch_embeds        = self.patch_embedding(pixel_values)
        embeddings          = patch_embeds.flatten(2)
        embeddings          = embeddings.transpose(1,2)
        embeddings          = embeddings + self.positional_embedding(self.position_ids)
        return embeddings

class SiglipVisionTransformer(nn.Module):
    def __init__(self, config:SiglipVisionConfig):
        super().__init__()
        self.config         = config
        embed_dim           = config.hidden_size
        
        self.embeddings     = SiglipVisionEmbeddings(config)
        self.encoder        = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:

        hidden_states     = self.embeddings(pixel_values)
        last_hidden_state = self.encoder(inputs_embeds=hidden_states)
        last_hidden_state = self.post_layernorm(last_hidden_state)
        return last_hidden_state

class SiglipVisionModel(nn.Module):

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values) -> Tuple:
        """
        inputs : [Batch_Size, Channels, Height, Width]
        outputs: [Batch_Size, Num_Patches, Embed_Dim]
        """
        return self.vision_model(pixel_values=pixel_values)