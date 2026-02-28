import torch
import torch.nn.functional as F
from cs336_basics.solutions.tokenizer import Tokenizer
from cs336_basics.solutions.transformer_lm import TransformerLM

@torch.no_grad()
def decode(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 32,
    temperature: float = 0.7,
    top_p: float = 0.9,
):
    device = next(model.parameters()).device
    end_id = tokenizer.encode("<|endoftext|>")[0]
    input_ids = torch.tensor(tokenizer.encode(prompt), device=device, dtype=torch.long)
    
    context_length = getattr(model, "context_length", 1024)

    for _ in range(max_new_tokens):
        idx_cond = input_ids[-context_length:].unsqueeze(0) # (1, T)

        logits = model(idx_cond) # (1, T, vocab_size)
        next_token_logits = logits[0, -1, :] / max(temperature, 1e-5)

        filtered_logits = top_p_filter(next_token_logits, top_p)

        probs = F.softmax(filtered_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        if next_token.item() == end_id:
            break
        
        input_ids = torch.cat((input_ids, next_token)) # batch decode later

    return tokenizer.decode(input_ids.tolist())

def top_p_filter(logits: torch.Tensor, top_p: float):
    if top_p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[indices_to_remove] = float('-inf')
    return logits