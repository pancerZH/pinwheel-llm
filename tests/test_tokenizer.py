from pinwheel.tokenizer import SPECIAL_TOKENS, token_byte_lengths, train_tokenizer


def test_token_bytes_add_up_to_utf8_length():
    corpus = ["The quick brown fox jumps over the lazy dog. 12345 naïve café, ok!"] * 50
    tok = train_tokenizer(corpus, vocab_size=300)
    lengths = token_byte_lengths(tok)
    text = "A naïve fox paid 42 € for café."
    ids = tok.encode(text).ids
    assert tok.decode(ids) == text
    assert lengths[ids].sum().item() == len(text.encode("utf-8"))
    assert all(lengths[tok.token_to_id(s)] == 0 for s in SPECIAL_TOKENS)
