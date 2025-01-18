import torch
import torch.nn.functional as F
from collections import Counter

def top_k_top_p_filtering(logits, top_k=50, top_p=0.9, min_tokens_to_keep=1):
    """
    Filter logits using top-k and/or top-p sampling.

    Args:
        logits (torch.Tensor): Logits of size [batch_size, vocab_size].
        top_k (int): Keep only top_k tokens with highest probabilities.
        top_p (float): Keep tokens with cumulative probabilities >= top_p.
        min_tokens_to_keep (int): Ensure at least this many tokens are kept.

    Returns:
        torch.Tensor: Filtered logits.
    """
    top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))

    # Top-k filtering
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = -float("inf")

    # Top-p filtering (nucleus sampling)
    # TODO: Top-p have bugs, probs become negative
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above top_p
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0

        # Shift indices to match the original logits
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits[indices_to_remove] = -float("inf")

    return logits

def sampling(logits, top_k, top_p):
    # Step 1: Apply top-k and/or top-p sampling
    logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

    # Step 2: Sample next token
    probabilities = F.softmax(logits, dim=-1)

    # TODO: fix the bug of negative probabilities
    probabilities = torch.clamp(probabilities, min=1e-9)
    next_token = torch.multinomial(probabilities, num_samples=1).squeeze(1)
    return next_token

def calculate_entropy(logits: torch.Tensor):
    # Ensure the probabilities are normalized
    probabilities = F.softmax(logits, dim=-1)
    probabilities = probabilities / torch.sum(probabilities, dim=-1, keepdim=True)

    # Compute the entropy using log base e (natural log)
    entropy_value = -torch.sum(probabilities * torch.log(probabilities + 1e-9), dim=-1)  # Add small value to avoid log(0)

    return entropy_value

def most_common_from_list(list_for_choose):
    most_common_element = Counter(list_for_choose).most_common()[0][0]
    return most_common_element