import torch.nn as nn
from . import register_model
from .base_model import BaseModel
from transformers import BertModel, BertTokenizer

@register_model("BertEncoder")
class BertEncoder(BaseModel):
    """
    A wrapper around a pre-trained BERT model from the Hugging Face
    library to make it compatible with our factory system.
    """
    def __init__(self, config: dict):
        super().__init__(config)
        model_name = self.info['bert_model_name']
        
        print(f"[BertEncoder] Loading pre-trained model: {model_name}")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name)
        
        # The "architecture" for this model is just its output dimension,
        # which is required by our generic CrossAttention module.
        self.info['architecture'] = [self.bert.config.hidden_size]

    def forward(self, text_inputs):
        """
        Takes a list of raw text strings and returns the last hidden state
        from the BERT model.
        """
        device = next(self.parameters()).device
        # Tokenize the text input strings
        tokens = self.tokenizer(
            text_inputs, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        ).to(device)
        
        # Get the embeddings
        outputs = self.bert(**tokens)
        return outputs.last_hidden_state