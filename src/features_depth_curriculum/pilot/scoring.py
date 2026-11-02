import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from tokenizers import Tokenizer

from ..callbacks import END_OF_TEXT_ID
from ..model import GPT
from .messages import Probe, ProbeScore


def encode_prompts(
    tokenizer: Tokenizer, prompts: list[str], device: str
) -> tuple[Int[torch.Tensor, "batch seq"], Int[torch.Tensor, " batch"]]:
    """Tokens of each prompt after an end-of-text, right-padded with 0 (safe: attention is
    causal), and the position of each prompt's last token."""
    ids = [[END_OF_TEXT_ID] + tokenizer.encode(prompt).ids for prompt in prompts]
    longest = max(map(len, ids))
    padded = [prompt_ids + [0] * (longest - len(prompt_ids)) for prompt_ids in ids]
    last_index = [len(prompt_ids) - 1 for prompt_ids in ids]
    return torch.tensor(padded, device=device), torch.tensor(last_index, device=device)


@torch.no_grad()
def residuals(
    model: GPT, token_ids: Int[torch.Tensor, "batch seq"], last_index: Int[torch.Tensor, " batch"]
) -> Float[torch.Tensor, "layers_plus_one batch width"]:
    """Residual stream at each prompt's last token: the embedding, then the output of each block."""
    rows = torch.arange(len(token_ids), device=token_ids.device)
    x = model.embedding(token_ids)  # (batch, seq, width)
    states = [x[rows, last_index]]
    for block in model.blocks:
        x = block(x)
        states.append(x[rows, last_index])
    return torch.stack(states)


def token_id(tokenizer: Tokenizer, word: str) -> int:
    """The id of a word that is a single token."""
    [word_id] = tokenizer.encode(word).ids
    return word_id


@torch.no_grad()
def score(model: GPT, tokenizer: Tokenizer, probes: list[Probe], device: str) -> list[ProbeScore]:
    """Log-prob of the target and each foil as the next token after every probe's prompt."""
    model.eval()
    token_ids, last_index = encode_prompts(tokenizer, [probe.prompt for probe in probes], device)
    logits = model(token_ids)[
        torch.arange(len(probes), device=device), last_index
    ]  # (batch, vocab)
    logprobs = F.log_softmax(logits.float(), dim=-1).cpu()
    return [
        ProbeScore(
            probe_id=probe.id,
            target_logprob=logprobs[i, token_id(tokenizer, probe.target)].item(),
            foil_logprobs={
                foil: logprobs[i, token_id(tokenizer, foil)].item() for foil in probe.foils
            },
        )
        for i, probe in enumerate(probes)
    ]


@torch.no_grad()
def val_loss(model: GPT, windows: Int[torch.Tensor, "n seq_plus_one"], device: str) -> float:
    """Mean next-token cross-entropy over the windows, 32 at a time."""
    model.eval()
    total = 0.0
    for batch in windows.split(32):
        logits = model(batch[:, :-1].to(device))  # (batch, seq, vocab)
        total += F.cross_entropy(
            logits.float().flatten(0, 1), batch[:, 1:].flatten().to(device), reduction="sum"
        ).item()
    return total / windows[:, 1:].numel()
