import torch
import sentencepiece as spm


sp_model = spm.SentencePieceProcessor(
    model_file="configs/llama_tokenizer_extended.model"
)
merged_vocab_size = sp_model.vocab_size()
ckpt = torch.load("data/llama_raw_ckpt/7B/consolidated.00.pth")

raw_vocab_size, hidden_size = ckpt["tok_embeddings.weight"].shape
extended_tok_embeddings = torch.randn(merged_vocab_size - raw_vocab_size, hidden_size)
extended_tok_embeddings = extended_tok_embeddings * 0.001
ckpt["tok_embeddings.weight"] = torch.cat(
    [ckpt["tok_embeddings.weight"], extended_tok_embeddings], dim=0
)

extended_out_embeddings = torch.randn(merged_vocab_size - raw_vocab_size, hidden_size)
extended_out_embeddings = extended_out_embeddings * 0.001
ckpt["output.weight"] = torch.cat(
    [ckpt["output.weight"], extended_out_embeddings], dim=0
)

rename_map = {
    "tok_embeddings.weight": "model.embed_tokens.weight",
    "norm.weight": "model.norm.weight",
    "output.weight": "lm_head.weight",
}

for f, t in rename_map.items():
    v = ckpt.pop(f)
    ckpt[t] = v

from_names = [
    "layers.{}.attention.wq.weight",
    "layers.{}.attention.wk.weight",
    "layers.{}.attention.wv.weight",
    "layers.{}.attention.wo.weight",
    "layers.{}.feed_forward.w1.weight",
    "layers.{}.feed_forward.w2.weight",
    "layers.{}.feed_forward.w3.weight",
    "layers.{}.attention_norm.weight",
    "layers.{}.ffn_norm.weight",
    "layers.{}.attention.inner_attention.rope.freqs",
]

to_names = [
    "model.layers.{}.self_attn.q_proj.weight",
    "model.layers.{}.self_attn.k_proj.weight",
    "model.layers.{}.self_attn.v_proj.weight",
    "model.layers.{}.self_attn.o_proj.weight",
    "model.layers.{}.mlp.gate_proj.weight",
    "model.layers.{}.mlp.down_proj.weight",
    "model.layers.{}.mlp.up_proj.weight",
    "model.layers.{}.input_layernorm.weight",
    "model.layers.{}.post_attention_layernorm.weight",
    "model.layers.{}.self_attn.rotary_emb.inv_freq",
]

for layer in range(32):
    for f, t in zip(from_names, to_names):
        f = f.format(layer)
        t = t.format(layer)
        v = ckpt.pop(f)
        ckpt[t] = v
torch.save(ckpt, "data/llama_raw_ckpt/7B/extended.pth")
