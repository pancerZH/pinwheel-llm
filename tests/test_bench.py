import torch
import torch.nn.functional as F

from pinwheel.bench import answer_logprobs
from pinwheel.model import GPT, GPTConfig
from pinwheel.tokenizer import BOS, train_tokenizer

TINY = GPTConfig(vocab_size=300, n_layer=2, n_head=4, n_kv_head=2, d_model=64, d_ff=128, max_seq_len=64)


def test_batched_answer_logprobs_match_one_by_one():
    torch.manual_seed(0)
    tok = train_tokenizer(["The cat sat on the mat. A dog ran in the park."] * 20, vocab_size=TINY.vocab_size)
    model = GPT(TINY).eval()
    pairs = [("The cat", " sat on the mat."), ("A dog ran", " in the park."), ("", "The dog sat.")]
    batched = answer_logprobs(model, tok, pairs, batch_size=3)  # padded to a common length

    for (ctx, ans), got in zip(pairs, batched):
        c, a = [tok.token_to_id(BOS)] + tok.encode(ctx).ids, tok.encode(ans).ids
        x = torch.tensor([c + a])
        logp = F.log_softmax(model(x).float(), dim=-1)[0]
        want = sum(logp[len(c) - 1 + j, t] for j, t in enumerate(a))
        assert abs(got.item() - want.item()) < 1e-4
