import torch
from torch import nn
from typing import Optional, Tuple, List
from torch.nn import CrossEntropyLoss
import math
from modeling_siglip import SiglipVisionConfig, SiglipVisionModel


class PaliGemmaForConditionalGeneration(nn.Module):

    def __init__(self, config: PaliGemmaConfig):
        self.config                = config
        self.vision_tower          = SiglipVisionModel(config.vision_config)
        self.multi_model_projector = PaliGemmaMultiModelProjector(config)
        self.vocab_size            = config.vocab_size

        self.language_model        = GemmaForCasualLM(config.text_config)
        self.pad_token_id          = self.config.pad_token_id if self.config.pad_token_id is not None else -1


    def tie_weights(self):
        return self.language_model.tie_weights()

    def forward(
        self,
        input_ids          : torch.LongTensor       = None,
        pixel_values       : torch.FloatTensor      = None,
        attention_mask     : Optional[torch.Tensor] = None,
        kv_cache           : Optional[KVCache]      = None 
    ) -> Tuple:
        pass
        