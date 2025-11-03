import torch, torch.nn.functional as F
from pathlib import Path
from tokenizers import Tokenizer
from bpe import MiniGPT, GPTConfig, build_or_load_tokenizer
import argparse

@torch.no_grad()
def beam_search(model, start, beam_width, max_new_tokens, cfg):
    """Deterministic beam‑search decoding."""
    model.eval()
    device = next(model.parameters()).device
    beams = [(start, 0.0)]  # (sequence tensor, log‑prob)
    for _ in range(max_new_tokens):
        candidates = []
        for seq, score in beams:
            logits, _ = model(seq[:, -cfg.context_size:])
            logits = logits[:, -1, :]
            log_probs = F.log_softmax(logits, dim=-1)
            top_log_probs, top_indices = torch.topk(log_probs, beam_width, dim=-1)
            for lp, idx_next in zip(top_log_probs[0], top_indices[0]):
                new_seq = torch.cat([seq, idx_next.view(1, 1)], dim=1)
                candidates.append((new_seq, score + float(lp)))
        # keep the best few
        ordered = sorted(candidates, key=lambda x: x[1], reverse=True)
        beams = ordered[:beam_width]
    return beams[0][0]

@torch.no_grad()
def generate(model, idx, cfg, max_new_tokens=200, temperature=0.8, top_k=50):
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -cfg.context_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            v, _ix = torch.topk(logits, top_k)
            logits[logits < v[..., -1, None]] = -float("inf")

        probs = F.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_tok), dim=1)
    return idx

def main(
    checkpoint="checkpoints/best.pt",
    text_path="goethe.txt",
    prompt_text=None,
    steps=200,
    temperature=0.8,
    top_k=50,
    use_beam=False,
    beam_width=4,
):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using", device)

    ckpt = torch.load(checkpoint, map_location=device)
    cfg = GPTConfig(**ckpt["cfg"])
    model = MiniGPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded model at step {ckpt.get('step', '?')}")

    tokenizer = Tokenizer.from_file("goethe_bpe.json")

    # pick or supply a prompt
    if prompt_text is None:
        text = Path(text_path).read_text(encoding="utf-8")
        import random
        pos = random.randint(0, len(text) - 100)
        prompt_text = text[pos : pos + 80]
        print("\n[Random prompt:]", repr(prompt_text))

    enc = tokenizer.encode(prompt_text)
    idx = torch.tensor([enc.ids], dtype=torch.long, device=device)

    import random
    torch.manual_seed(random.randint(0, 10_000_000))

    # ---- choose decoding method ----
    if use_beam:
        print(f"\nUsing beam search (width={beam_width}) ...\n")
        out = beam_search(model, idx, beam_width=beam_width,
                          max_new_tokens=steps, cfg=cfg)
    else:
        print("\nUsing temperature/top‑k sampling ...\n")
        out = generate(model, idx, cfg, max_new_tokens=steps,
                       temperature=temperature, top_k=top_k)

    decoded = tokenizer.decode(out[0].cpu().tolist())
    print("\n=== OUTPUT ===\n")
    print(decoded)
    print("\n==============\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate text with the trained GoethePT model."
    )
    parser.add_argument(
        "--checkpoint", default="checkpoints/best.pt",
        help="Path to model checkpoint (.pt)"
    )
    parser.add_argument(
        "--text_path", default="goethe.txt",
        help="Path to the source text (for random prompts)"
    )
    parser.add_argument(
        "--prompt", type=str, default=None,
        help="Prompt text to start generation."
    )
    parser.add_argument(
        "--steps", type=int, default=200,
        help="How many new tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8,
        help="Sampling temperature (ignored for beam search)"
    )
    parser.add_argument(
        "--top_k", type=int, default=50,
        help="Top‑k cutoff for sampling (ignored for beam search)"
    )
    parser.add_argument(
        "--beam", action="store_true",
        help="Use beam search instead of sampling"
    )
    parser.add_argument(
        "--beam_width", type=int, default=4,
        help="Beam width if --beam is set"
    )

    args = parser.parse_args()

    main(
        checkpoint=args.checkpoint,
        text_path=args.text_path,
        prompt_text=args.prompt,
        steps=args.steps,
        temperature=args.temperature,
        top_k=args.top_k,
        use_beam=args.beam,
        beam_width=args.beam_width,
    )