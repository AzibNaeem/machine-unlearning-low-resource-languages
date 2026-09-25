# Step 4: embedding-layer injection plumbing

Code: `text/embed_inject.py`. Tests: `text/tests/test_embed_inject.py` (plain python; output in `outputs/step4/A/test_output.txt`).

## API
- `embed(model, input_ids, position_ids=None)` -> `(batch, seq, 768)`: `model.roberta.embeddings` output.
- `forward_from_embeddings(model, embeds, attention_mask)` -> logits `(batch, 7)`: encoder then `model.classifier`. `embeds` may be a leaf tensor with `requires_grad`.

## What the embedding output contains
`LayerNorm(word_emb + position_emb + token_type_emb[0])`, then dropout (identity in eval). Position ids are RoBERTa style (start at `padding_idx + 1 = 2`, pad tokens keep id 1); token types are all 0. Noise injected here is therefore already post-LayerNorm, i.e. on the scale of real encoder inputs, and replaces all three embedding components at once.

## transformers 5.17 findings
- `XLMRobertaEncoder.forward(hidden_states, attention_mask, ...)` takes a mask prepared by the model, not the raw 0/1 mask. The old `get_extended_attention_mask` is gone.
- `XLMRobertaModel` builds it with `masking_utils.create_bidirectional_mask(config=, inputs_embeds=, attention_mask=)`; `forward_from_embeddings` calls the same helper, so the mask format (bool or additive, depending on the attention backend) always matches the installed backend.
- Classification head reads token 0 (`<s>`) only; there is no pooler (`add_pooling_layer=False`).
- `model(inputs_embeds=...)` is NOT the bypass: it still adds position and token-type embeddings and applies LayerNorm to the input.

## Results (seed-0 baseline checkpoint, eval mode, fp32, GPU)
| Test | Result |
|---|---|
| 1 bypass vs normal logits, 16 val sentences (padded to 164) | PASS, max abs diff 0.0 |
| 2 32 val sentences, argmax identical | PASS, max abs diff 0.0 |
| 3 grad wrt random noise, params frozen | PASS, finite, non-zero (mean abs 6.5e-4) |
| 4 random values in padded positions | PASS, logits unchanged (diff 0.0) |
| 5 full validation (n=99) accuracy | PASS, 87.88 both paths |

## Gotchas
- The checkpoint is a state_dict; the model must be built from `data/xlmr` with `num_labels=7` first (HF prints harmless UNEXPECTED/MISSING load notes for the untrained-head init, then the state_dict fills the classifier).
- Diff 0.0 exactly is expected: both paths run the identical ops.
- Padded positions are masked as attention keys, but the encoder still computes hidden states there; they are just never read (only token 0 feeds the head). Noise at padded positions is harmless; noise shaped to full 164 length is fine as long as the mask is passed.
- Noise has no `<s>` guarantee: position 0 is the only one the head reads, so injected noise must include position 0 unmasked.
- Keep the model in eval for injection (dropout in train mode makes the bypass stochastic, and `embed` output would be dropped-out too).
