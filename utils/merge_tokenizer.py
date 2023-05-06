from tqdm import tqdm
import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as model

raw_model = model.ModelProto()
raw_model.ParseFromString(
    open("configs/tokenizer_models/llama_tokenizer.model", "rb").read()
)

exist_pieces = set([p.piece for p in raw_model.pieces])
cn_model = model.ModelProto()
cn_model.ParseFromString(
    open("configs/tokenizer_models/4w_cn_vocab_wudao15.model", "rb").read()
)

for p in tqdm(cn_model.pieces, total=len(cn_model.pieces)):
    if p.piece not in exist_pieces:
        raw_model.pieces.append(p)

with open("configs/tokenizer_models/llama_tokenizer_extended.model", "wb") as f:
    f.write(raw_model.SerializeToString())

sp_model = spm.SentencePieceProcessor(
    model_file="configs/tokenizer_models/llama_tokenizer_extended.model"
)

print("merged vocab size: {}".format(sp_model.vocab_size()))
