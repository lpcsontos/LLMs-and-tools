from torch.nn import functional as F
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()
trainer = trainers.BpeTrainer(
    vocab_size=8000, min_frequency=2,
    special_tokens=["<pad>", "<unk>", "<sos>", "<eos>"]
)
tokenizer.train(files=["../data/out.txt"], trainer=trainer)
tokenizer.save("../tokenizers/tokenizer_big.json")