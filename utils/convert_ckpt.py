import torch
import sentencepiece as spm


sp_model = spm.SentencePieceProcessor(
    model_file="configs/llama_tokenizer_extended.model"
)
merged_vocab_size = sp_model.vocab_size()
ckpt = torch.load('data/llama_raw_ckpt/7B/consolidated.00.pth')

raw_vocab_size, hidden_size = ckpt['tok_embeddings.weight'].shape
extended_tok_embeddings = torch.randn(merged_vocab_size - raw_vocab_size, hidden_size)
extended_tok_embeddings = extended_tok_embeddings * 0.001
ckpt['tok_embeddings.weight'] = torch.cat([ckpt['tok_embeddings.weight'], extended_tok_embeddings], dim=0)

extended_out_embeddings = torch.randn(merged_vocab_size - raw_vocab_size, hidden_size)
extended_out_embeddings = extended_out_embeddings * 0.001
ckpt['output.weight'] = torch.cat([ckpt['output.weight'], extended_out_embeddings], dim=0)

torch.save(ckpt, 'data/llama_raw_ckpt/7B/extended.pth')