"""DoLa: Decoding by Contrasting Layers (Chuang et al., ICLR 2024).

Contrasts logits from a mature (late) layer with a premature (early)
layer to amplify factual knowledge encoded in later layers while
suppressing surface-level patterns from earlier layers.

Core algorithm per token step:
  1. Compute residual stream at mature layer and premature candidates
  2. Unembed both through ln_final + W_U
  3. Select premature layer with highest Jensen-Shannon divergence
  4. contrasted = log_softmax(mature) - log_softmax(premature)
"""

import logging
from typing import Any, Optional

import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer

logger = logging.getLogger(__name__)


def _unembed(model: HookedTransformer, resid: torch.Tensor) -> torch.Tensor:
    """Project residual stream to vocabulary logits via ln_final + W_U.

    Args:
        model: HookedTransformer model
        resid: Residual stream tensor [..., d_model]

    Returns:
        Logits tensor [..., vocab_size]
    """
    normed = model.ln_final(resid)
    logits = normed @ model.W_U
    if model.b_U is not None:
        logits = logits + model.b_U
    return logits


def _jensen_shannon_divergence(
    log_p: torch.Tensor, log_q: torch.Tensor
) -> torch.Tensor:
    """Compute Jensen-Shannon divergence between two log-probability distributions.

    JSD(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M), where M = 0.5*(P+Q)

    Args:
        log_p: Log-probabilities of distribution P [..., vocab]
        log_q: Log-probabilities of distribution Q [..., vocab]

    Returns:
        JSD value (scalar or per-batch)
    """
    p = log_p.exp()
    q = log_q.exp()
    m = 0.5 * (p + q)
    log_m = m.log()

    kl_pm = F.kl_div(log_m, p, reduction="none", log_target=False).sum(dim=-1)
    kl_qm = F.kl_div(log_m, q, reduction="none", log_target=False).sum(dim=-1)
    return 0.5 * (kl_pm + kl_qm)


def select_premature_layer(
    model: HookedTransformer,
    cache: dict,
    mature_layer: int,
    premature_candidates: list[int],
    position: int = -1,
) -> int:
    """Select the premature layer with highest JSD from the mature layer.

    Higher JSD means the premature layer distribution diverges most from
    the mature layer, indicating the mature layer has added the most
    factual information relative to that premature layer.

    Args:
        model: HookedTransformer model
        cache: Activation cache from model.run_with_cache()
        mature_layer: Index of the mature (late) layer
        premature_candidates: List of candidate premature layer indices
        position: Token position to evaluate (-1 for last)

    Returns:
        Index of the selected premature layer
    """
    mature_resid = cache[f"blocks.{mature_layer}.hook_resid_post"][0, position, :]
    mature_logits = _unembed(model, mature_resid)
    mature_log_probs = F.log_softmax(mature_logits.float(), dim=-1)

    best_layer = premature_candidates[0]
    best_jsd = -1.0

    for layer in premature_candidates:
        premature_resid = cache[f"blocks.{layer}.hook_resid_post"][0, position, :]
        premature_logits = _unembed(model, premature_resid)
        premature_log_probs = F.log_softmax(premature_logits.float(), dim=-1)

        jsd = _jensen_shannon_divergence(mature_log_probs, premature_log_probs).item()
        if jsd > best_jsd:
            best_jsd = jsd
            best_layer = layer

    return best_layer


def compute_dola_logits(
    model: HookedTransformer,
    tokens: torch.Tensor,
    mature_layer: Optional[int] = None,
    premature_candidates: Optional[list[int]] = None,
    dynamic_premature: bool = True,
    static_premature_layer: Optional[int] = None,
) -> torch.Tensor:
    """Compute DoLa contrasted logits for a full token sequence.

    For each token position, contrasts mature layer logits with the
    selected premature layer logits.

    Args:
        model: HookedTransformer model
        tokens: Input token IDs [1, seq_len]
        mature_layer: Mature layer index (default: last layer)
        premature_candidates: Premature layer candidates (default: 0 to 60% of layers)
        dynamic_premature: If True, select premature layer per-position via JSD
        static_premature_layer: Fixed premature layer (used when dynamic=False)

    Returns:
        Contrasted logits [1, seq_len, vocab_size]
    """
    n_layers = model.cfg.n_layers
    if mature_layer is None:
        mature_layer = n_layers - 1

    if premature_candidates is None:
        cutoff = int(n_layers * 0.6)
        premature_candidates = list(range(0, max(1, cutoff)))

    # Cache residual streams at mature and all premature candidate layers
    all_layers = set(premature_candidates) | {mature_layer}
    hook_names = [f"blocks.{l}.hook_resid_post" for l in all_layers]

    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=hook_names)

    seq_len = tokens.shape[1]
    mature_resid = cache[f"blocks.{mature_layer}.hook_resid_post"][0]  # [seq, d_model]
    mature_logits = _unembed(model, mature_resid)  # [seq, vocab]
    mature_log_probs = F.log_softmax(mature_logits.float(), dim=-1)

    if not dynamic_premature:
        # Static: use a fixed premature layer for all positions
        p_layer = static_premature_layer if static_premature_layer is not None else premature_candidates[0]
        premature_resid = cache[f"blocks.{p_layer}.hook_resid_post"][0]
        premature_logits = _unembed(model, premature_resid)
        premature_log_probs = F.log_softmax(premature_logits.float(), dim=-1)

        contrasted = mature_log_probs - premature_log_probs
        return contrasted.unsqueeze(0)  # [1, seq, vocab]

    # Dynamic: select premature layer per position via JSD
    contrasted_list = []
    for pos in range(seq_len):
        # Find premature layer with highest JSD from mature at this position
        best_layer = premature_candidates[0]
        best_jsd = -1.0

        mature_lp_pos = mature_log_probs[pos]

        for layer in premature_candidates:
            p_resid = cache[f"blocks.{layer}.hook_resid_post"][0, pos, :]
            p_logits = _unembed(model, p_resid)
            p_lp = F.log_softmax(p_logits.float(), dim=-1)

            jsd = _jensen_shannon_divergence(mature_lp_pos.unsqueeze(0), p_lp.unsqueeze(0)).item()
            if jsd > best_jsd:
                best_jsd = jsd
                best_layer = layer

        # Compute contrasted logits for this position
        p_resid = cache[f"blocks.{best_layer}.hook_resid_post"][0, pos, :]
        p_logits = _unembed(model, p_resid)
        p_lp = F.log_softmax(p_logits.float(), dim=-1)

        contrasted_list.append(mature_lp_pos - p_lp)

    contrasted = torch.stack(contrasted_list, dim=0)  # [seq, vocab]
    return contrasted.unsqueeze(0)  # [1, seq, vocab]


def generate_with_dola(
    model: HookedTransformer,
    input_text: str,
    mature_layer: Optional[int] = None,
    premature_candidates: Optional[list[int]] = None,
    dynamic_premature: bool = True,
    max_new_tokens: int = 100,
) -> tuple[str, str]:
    """Generate text using DoLa decoding.

    Args:
        model: HookedTransformer model
        input_text: Input prompt
        mature_layer: Mature layer index (default: last layer)
        premature_candidates: Premature candidates (default: 0 to 60%)
        dynamic_premature: Dynamic premature layer selection per token
        max_new_tokens: Maximum tokens to generate

    Returns:
        Tuple of (original_output, dola_output)
    """
    n_layers = model.cfg.n_layers
    if mature_layer is None:
        mature_layer = n_layers - 1
    if premature_candidates is None:
        cutoff = int(n_layers * 0.6)
        premature_candidates = list(range(0, max(1, cutoff)))

    tokens = model.to_tokens(input_text)

    # Generate original (greedy) output
    with torch.no_grad():
        orig_ids = model.generate(tokens, max_new_tokens=max_new_tokens, do_sample=False)
    original_output = model.to_string(orig_ids[0, tokens.shape[1]:]).strip()

    # Generate with DoLa (token-by-token)
    all_layers = set(premature_candidates) | {mature_layer}
    hook_names = [f"blocks.{l}.hook_resid_post" for l in all_layers]

    generated = tokens.clone()

    for _ in range(max_new_tokens):
        with torch.no_grad():
            _, cache = model.run_with_cache(generated, names_filter=hook_names)

        # Get mature layer logits at last position
        mature_resid = cache[f"blocks.{mature_layer}.hook_resid_post"][0, -1, :]
        mature_logits = _unembed(model, mature_resid)
        mature_lp = F.log_softmax(mature_logits.float(), dim=-1)

        if dynamic_premature:
            # Dynamic selection: pick premature layer with max JSD
            best_layer = premature_candidates[0]
            best_jsd = -1.0
            for layer in premature_candidates:
                p_resid = cache[f"blocks.{layer}.hook_resid_post"][0, -1, :]
                p_logits = _unembed(model, p_resid)
                p_lp = F.log_softmax(p_logits.float(), dim=-1)
                jsd = _jensen_shannon_divergence(
                    mature_lp.unsqueeze(0), p_lp.unsqueeze(0)
                ).item()
                if jsd > best_jsd:
                    best_jsd = jsd
                    best_layer = layer
            selected_premature = best_layer
        else:
            selected_premature = premature_candidates[0]

        # Contrasted logits
        p_resid = cache[f"blocks.{selected_premature}.hook_resid_post"][0, -1, :]
        p_logits = _unembed(model, p_resid)
        p_lp = F.log_softmax(p_logits.float(), dim=-1)

        contrasted = mature_lp - p_lp

        # Greedy select next token from contrasted distribution
        next_token = contrasted.argmax(dim=-1, keepdim=True)

        if model.tokenizer.eos_token_id is not None:
            if next_token.item() == model.tokenizer.eos_token_id:
                break

        generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)

    dola_output = model.to_string(generated[0, tokens.shape[1]:]).strip()
    return original_output, dola_output


def compute_answer_log_prob_dola(
    model: HookedTransformer,
    question: str,
    answer: str,
    mature_layer: Optional[int] = None,
    premature_candidates: Optional[list[int]] = None,
    dynamic_premature: bool = True,
) -> float:
    """Compute answer log-probability using DoLa contrasted logits.

    Drop-in replacement for compute_answer_log_prob that uses DoLa's
    layer-contrasted logits instead of standard model logits.
    Compatible with evaluate_truthfulqa_mc1/mc2 via score_fn parameter.

    Args:
        model: HookedTransformer model
        question: Question text
        answer: Answer text
        mature_layer: Mature layer index
        premature_candidates: Premature layer candidates
        dynamic_premature: Dynamic premature selection

    Returns:
        Sum of contrasted log-probabilities for answer tokens
    """
    prompt = f"Q: {question}\nA:"
    full_text = f"Q: {question}\nA: {answer}"

    prompt_tokens = model.to_tokens(prompt, prepend_bos=True)
    full_tokens = model.to_tokens(full_text, prepend_bos=True)

    q_len = prompt_tokens.shape[1]

    if full_tokens.shape[1] <= q_len:
        return float("-inf")

    # Get DoLa contrasted logits for full sequence
    contrasted_logits = compute_dola_logits(
        model, full_tokens,
        mature_layer=mature_layer,
        premature_candidates=premature_candidates,
        dynamic_premature=dynamic_premature,
    )

    # contrasted_logits: [1, seq_len, vocab_size]
    # These are already in log-probability-like space (log_softmax difference)
    # Normalize to valid log-probs for scoring
    log_probs = F.log_softmax(contrasted_logits[0].float(), dim=-1)

    total_log_prob = 0.0
    for i in range(q_len, full_tokens.shape[1]):
        token_id = full_tokens[0, i].item()
        total_log_prob += log_probs[i - 1, token_id].item()

    return total_log_prob


def create_dola_score_fn(
    mature_layer: Optional[int] = None,
    premature_candidates: Optional[list[int]] = None,
    dynamic_premature: bool = True,
):
    """Create a DoLa scoring function compatible with MC1/MC2 evaluation.

    Returns a function with signature (model, question, answer) -> float
    suitable for passing as score_fn to evaluate_truthfulqa_mc1/mc2.

    Args:
        mature_layer: Mature layer index
        premature_candidates: Premature layer candidates
        dynamic_premature: Dynamic premature selection

    Returns:
        Scoring function (model, question, answer) -> float
    """
    def score_fn(model, question, answer):
        return compute_answer_log_prob_dola(
            model, question, answer,
            mature_layer=mature_layer,
            premature_candidates=premature_candidates,
            dynamic_premature=dynamic_premature,
        )
    return score_fn
