import argparse
import torch, torch.nn as nn, torch.nn.functional as F
import os, glob
from dataclasses import dataclass
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from pathlib import Path
from time import time

# ---------------------------------------------------------------------
#  CONFIGURATION
# ---------------------------------------------------------------------
@dataclass
class GPTConfig:
    vocab_size: int = 0
    context_size: int = 256
    n_layers: int = 4
    n_heads: int = 4
    n_embd: int = 256
    dropout: float = 0.2

# ---------------------------------------------------------------------
#  CORE MODEL
# ---------------------------------------------------------------------
class SelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.n_embd // cfg.n_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=2)
        q = q.view(B, T, self.n_heads, -1).transpose(1, 2)
        k = k.view(B, T, self.n_heads, -1).transpose(1, 2)
        v = v.view(B, T, self.n_heads, -1).transpose(1, 2)

        mask = torch.tril(torch.ones(T, T, device=x.device)) == 0
        att = (q @ k.transpose(-2, -1)) * self.scale
        att = att.masked_fill(mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.dropout(self.proj(out))

class TransformerBlock(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = SelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class MiniGPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_embed   = nn.Embedding(cfg.context_size, cfg.n_embd)
        self.blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.token_embed(idx) + self.pos_embed(pos)
        x = self.blocks(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss

# ---------------------------------------------------------------------
#  TOKENIZER UTILITIES
# ---------------------------------------------------------------------
def build_or_load_tokenizer(text_path: str, vocab_size: int = 5000):
    tok_path = Path("goethe_bpe.json")
    if tok_path.exists():
        print("Loading existing tokenizer...")
        tok = Tokenizer.from_file(str(tok_path))
    else:
        print("Training new ByteLevel BPE tokenizer...")
        tok = Tokenizer(models.BPE())
        # ByteLevel preserves spaces explicitly as part of token text; otherwise no spaces are ever trained on -> generates no spaces
        tok.pre_tokenizer = pre_tokenizers.ByteLevel()
        trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["<unk>"])
        tok.train([text_path], trainer)
        tok.decoder = decoders.ByteLevel()      # crucial for decoding with spaces
        tok.save(str(tok_path))
    return tok

# ---------------------------------------------------------------------
#  TRAINING LOOP
# ---------------------------------------------------------------------

def train(cfg: GPTConfig,
          text_path="goethe.txt",
          steps=50000,
          batch_size=16,
          save_every=2000,
          eval_every=500,
          patience=10,
          save_dir="checkpoints"):
    """
    Trains MiniGPT with validation, checkpointing, and auto‑resume.
    """

    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    # ---- Load data and tokenizer ----
    text = Path(text_path).read_text(encoding="utf-8")
    tokenizer = build_or_load_tokenizer(text_path)
    ids = torch.tensor(tokenizer.encode(text).ids, dtype=torch.long)
    cfg.vocab_size = tokenizer.get_vocab_size()

    n = int(0.9 * len(ids))
    train_data, val_data = ids[:n], ids[n:]

    model = MiniGPT(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    # add scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=steps, eta_min=1e-5)

    # ---- Auto‑resume if checkpoint exists ----
    ckpts = sorted(glob.glob(f"{save_dir}/step_*.pt"))
    start_step = 0
    best_val = float("inf")
    bad_epochs = 0

    if ckpts:
        last_ckpt = ckpts[-1]
        print(f"Resuming from {last_ckpt}")
        data = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(data["model"])
        if "opt" in data:
            opt.load_state_dict(data["opt"])
        start_step = data.get("step", 0)
        best_val = data.get("best_val", best_val)
        bad_epochs = data.get("bad_epochs", bad_epochs)

    def get_batch(split):
        data = train_data if split == "train" else val_data
        ix = torch.randint(len(data) - cfg.context_size, (batch_size,))
        x = torch.stack([data[i:i+cfg.context_size] for i in ix])
        y = torch.stack([data[i+1:i+1+cfg.context_size] for i in ix])
        return x.to(device), y.to(device)

    t0 = time()
    for step in range(start_step + 1, steps + 1):
        model.train()
        xb, yb = get_batch("train")
        _, loss = model(xb, yb)
        opt.zero_grad(); loss.backward(); opt.step(); scheduler.step()

        if step % eval_every == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                vx, vy = get_batch("val")
                _, vloss = model(vx, vy)

            elapsed = time() - t0
            print(f"step {step:6d} | train {loss.item():.3f} | val {vloss.item():.3f} | {elapsed/60:.1f} min")
            t0 = time()

            # improvement tracking
            if vloss.item() < best_val:
                best_val = vloss.item()
                bad_epochs = 0
                torch.save({
                    "model": model.state_dict(),
                    "cfg": cfg.__dict__,
                    "step": step,
                    "val_loss": best_val
                }, f"{save_dir}/best.pt")
                print(f"  ✅ new best model saved (val {best_val:.3f})")
            else:
                bad_epochs += 1

            evaluate(model, val_data, cfg, tokenizer)

            if bad_epochs >= patience:
                print(f"No val. improvement in {patience} intervals → early stop.")
                break

        if step % save_every == 0:
            torch.save({
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "step": step,
                "best_val": best_val,
                "bad_epochs": bad_epochs
            }, f"{save_dir}/step_{step}.pt")
            print(f"  💾 checkpoint saved at step {step}")

    print("Training complete.")
    return model, tokenizer

# ---------------------------------------------------------------------
#  EVALUATION / GENERATION
# ---------------------------------------------------------------------
@torch.no_grad()
def generate(model, idx, cfg, max_new_tokens=100, temperature=0.8, top_k=50):
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -cfg.context_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, ix = torch.topk(logits, top_k)
            logits[logits < v[..., -1, None]] = -float('inf')
        probs = F.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, 1)
        idx = torch.cat((idx, next_tok), dim=1)
    return idx

def evaluate(model, data, cfg, tok):
    device = next(model.parameters()).device
    prompt = data[:cfg.context_size].unsqueeze(0).to(device)
    sample = generate(model, prompt, cfg, 150)
    out_text = tok.decode(sample[0].cpu().numpy().tolist())
    print("=== SAMPLE ===")
    print(out_text[:500])
    print("===============\n")

# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MiniGPT on custom text")
    parser.add_argument(
        "--text",
        "-t",
        dest="text_path",
        default="goethe.txt",
        help="Path to the training text file (default: goethe.txt)",
    )
    args = parser.parse_args()

    cfg = GPTConfig()
    model, tokenizer = train(cfg, text_path=args.text_path, steps=100_000)
