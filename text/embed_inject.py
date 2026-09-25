import torch
from transformers.masking_utils import create_bidirectional_mask


def embed(model, input_ids, position_ids=None):
    """Output of model.roberta.embeddings: LayerNorm(word + position + token_type[0]) followed by dropout.

    Position ids follow RoBERTa (padding_idx + 1 onwards, pad tokens keep padding_idx); token types are all 0.
    Shape (batch, seq_len, hidden). Dropout is inactive in eval mode.
    """
    return model.roberta.embeddings(input_ids=input_ids, position_ids=position_ids)


def forward_from_embeddings(model, embeds, attention_mask=None):
    """Run embedding-space activations through the encoder and classification head; returns logits.

    attention_mask None means every position is attended. The head reads position 0 only.
    """
    mask = create_bidirectional_mask(config=model.config, inputs_embeds=embeds, attention_mask=attention_mask)
    hidden = model.roberta.encoder(embeds, attention_mask=mask).last_hidden_state
    return model.classifier(hidden)
