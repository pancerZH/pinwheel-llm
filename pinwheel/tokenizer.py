"""Byte-level BPE tokenizer (HF `tokenizers`), plus the byte counts that BPB needs."""

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

# Reserved from the start so the vocabulary never has to change for chat / SFT later.
SPECIAL_TOKENS = [
    "<|bos|>",
    "<|user|>",
    "<|assistant|>",
    "<|end|>",
    "<|pad|>",
    "<|reserved_0|>",
    "<|reserved_1|>",
    "<|reserved_2|>",
]
BOS = "<|bos|>"


def train_tokenizer(texts, vocab_size: int) -> Tokenizer:
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [
            # One token per digit, so every number is spelled the same way (helps arithmetic).
            pre_tokenizers.Digits(individual_digits=True),
            pre_tokenizers.ByteLevel(add_prefix_space=False),
        ]
    )
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        min_frequency=2,
    )
    tok.train_from_iterator(texts, trainer=trainer)
    return tok


def load_tokenizer(path) -> Tokenizer:
    return Tokenizer.from_file(str(path))


def token_byte_lengths(tok: Tokenizer) -> torch.Tensor:
    """UTF-8 bytes each token stands for; 0 for special tokens, which BPB leaves out.

    Byte-level BPE maps every byte to exactly one character, so the length of a
    token's string is its byte count.
    """
    special = set(SPECIAL_TOKENS)
    lengths = [
        0 if (s := tok.id_to_token(i)) in special else len(s) for i in range(tok.get_vocab_size())
    ]
    return torch.tensor(lengths, dtype=torch.int64)
