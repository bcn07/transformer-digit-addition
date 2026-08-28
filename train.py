"""Train a small transformer on decimal addition.

Extracted from notebooks/digit_addition.ipynb so runs can be launched from the
command line and swept on a cluster. The notebook remains the place for analysis.
"""

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import einops
import torch as t
from jaxtyping import Int
from torch import Tensor
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer, HookedTransformerConfig


def pick_device(requested: str) -> t.device:
    if requested != "auto":
        return t.device(requested)
    if t.cuda.is_available():
        return t.device("cuda")
    if t.backends.mps.is_available():
        return t.device("mps")
    return t.device("cpu")


DEVICE = pick_device(os.environ.get("DIGIT_ADDITION_DEVICE", "auto"))


VOCABULARY = {f"{i:d}": i for i in range(10)}
VOCABULARY["+"] = 10
VOCABULARY["="] = 11

# convenience aliases
TOKEN_TO_ID = VOCABULARY
ID_TO_TOKEN = {idx: token for token, idx in TOKEN_TO_ID.items()}
D_VOCAB = len(VOCABULARY)

# 1D (seq,) or 2D (batch, seq) ids, as a tensor or plain lists
TokenIds = Tensor | list[int] | list[list[int]]


def to_tokens(strings: str | list[str]) -> Int[Tensor, "batch seq"]:
    """'49+97=641' -> tensor([[4, 9, 10, 9, 7, 11, 6, 4, 1]]). Always returns (batch, seq)."""
    if isinstance(strings, str):
        strings = [strings]
    return t.tensor([[TOKEN_TO_ID[c] for c in s] for s in strings], device=DEVICE)


def to_str_tokens(tokens: TokenIds) -> list[str] | list[list[str]]:
    """Token ids -> the characters they stand for, one per token. Nested if the input is 2D."""
    if isinstance(tokens, Tensor):
        tokens = tokens.tolist()
    if tokens and isinstance(tokens[0], list):
        return [[ID_TO_TOKEN[i] for i in row] for row in tokens]
    return [ID_TO_TOKEN[i] for i in tokens]


def to_string(tokens: TokenIds) -> str | list[str]:
    """Token ids -> '49+97=641'. A list of strings if the input is 2D."""
    strs = to_str_tokens(tokens)
    if strs and isinstance(strs[0], list):
        return ["".join(row) for row in strs]
    return "".join(strs)


# sequences will have the result of the computation reversed


def sum_dataset(num_digits: int = 2, seed: int = 42, frac_train: float = 0.3) -> tuple[Tensor, Tensor]:
    
    pairs = []

    for i in range(10**num_digits):
        for j in range(10**num_digits):
            pairs.append(f"{i:0{num_digits}d}+{j:0{num_digits}d}={f'{(i+j):0{num_digits+1}d}'[::-1]}")

    data = t.tensor([[TOKEN_TO_ID[token] for token in example] for example in pairs], device=DEVICE)
    g = t.Generator().manual_seed(seed)

    _n = data.shape[0]
    split = t.randperm(_n, generator=g)
    train_idx = split[:int(frac_train * _n)].to(device=DEVICE)
    test_idx = split[int(frac_train * _n):].to(device=DEVICE)


    assert t.cat([train_idx, test_idx], dim=0).sort().values.equal(t.arange(len(data), device=data.device))

    train = data[train_idx]
    test = data[test_idx]
    
    return train, test

@dataclass
class TrainArgs:
    num_digits: int
    n_layers: int
    # trainset_size: int   -- derivable from train.shape[0]
    # valset_size: int     -- derivable from test.shape[0]
    frac_train: float
    epochs: int
    # batch_size: int      -- full batch, so there is nothing to vary
    lr: float
    # lr_start: float
    # lr_end: float
    weight_decay: float
    seed: int
    eval_every: int
    # d_model: int
    # d_head: int
    # n_heads: int
    # d_mlp: int
    # normalization_type: str | None
    use_wandb: bool
    device: str
    wandb_project: str = "transformer-digit-addition"
    wandb_group: str | None = None

    def __post_init__(self):
        self.seq_len = self.num_digits * 3 + 2 + 1  # We have [{a}, +, {b}, =, {a+b}]
        self.run_name = (
            f"L{self.n_layers}_wd{self.weight_decay}"
            f"_ft{self.frac_train}_s{self.seed}"
        )

def create_model(args: TrainArgs) -> HookedTransformer:
    """A fresh model from init, sized from `args`.

    Re-seeded on every call, so a sweep over `frac_train` varies only the thing
    being swept -- every run starts from identical weights.
    """
    t.manual_seed(args.seed)

    d_model = 128
    n_heads = 4

    cfg = HookedTransformerConfig(
        n_layers=args.n_layers,
        n_ctx=args.seq_len,  # We have [{a}, +, {b}, =, {a+b}]
        d_model=d_model,
        d_head=d_model // n_heads,
        n_heads=n_heads,
        d_mlp=4 * d_model,
        attn_only=False,
        act_fn="relu",
        # We have all digits, and "+="
        d_vocab=len(VOCABULARY),
        # off for training: these materialise per-head tensors and cost ~140x
        # throughput. Switch on before the interpretability section.
        use_attn_result=False,
        use_split_qkv_input=False,
        use_hook_tokens=False,
        normalization_type=None,
        device=args.device,
    )

    return HookedTransformer(cfg)

class Trainer:
    def __init__(self, model: HookedTransformer, args: TrainArgs):
        self.args = args
        self.model = model
        
    def training_step(self, toks: Tensor) -> Tensor:
        logits, target = self._shared_train_validation_step(toks)
        return self._loss(logits, target)
    
    def validation_step(self, toks: Tensor) -> tuple[float, float, float]:
        """Loss, per-digit accuracy, per-sequence accuracy.

        Per-sequence is the metric that matters: a sum with one wrong digit is a
        wrong sum. It also separates "learned the seen table entries" from "full
        generalisation" far more sharply than the per-digit average does.
        """
        logits, target = self._shared_train_validation_step(toks)
        loss = self._loss(logits, target).item()
        correct = logits.argmax(dim=-1) == target
        return loss, correct.float().mean().item(), correct.all(dim=-1).float().mean().item()
    
    def _loss(self, logits: Tensor, target: Tensor) -> Tensor:
        return t.nn.functional.cross_entropy(
            einops.rearrange(logits, "batch seq vocab_out -> (batch seq) vocab_out"),
            einops.rearrange(target, "batch seq -> (batch seq)")
        )
    
    def _shared_train_validation_step(self, toks: Tensor) -> tuple[Tensor, Tensor]:
        toks = toks.to(self.args.device)
        logits = self.model(toks)[:, -(self.args.num_digits + 2): -1]
        target = toks[:, -(self.args.num_digits + 1):]
        return logits, target
    
    def configure_optimizers(self):
        optimizer = t.optim.AdamW(
            self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay
        )
        return optimizer

def _wandb():
    import wandb
    return wandb


def train_model(
    model: HookedTransformer,
    args: TrainArgs,
    ckpt_dir: Path | None = None,
    ckpt_every: int = 0,
):
    """Train and return (model, history). Checkpoints only if `ckpt_dir` is given."""
    best = -1.0
    if args.use_wandb:
        configs = asdict(args)
        configs["device"] = str(args.device)
        _wandb().init(
            project=args.wandb_project,
            group=args.wandb_group,
            name=args.run_name,
            config=configs,
        )
    trainer = Trainer(model=model, args=args)
    optimizer = trainer.configure_optimizers()
    
    train, test = sum_dataset(
        num_digits=args.num_digits, seed=args.seed, frac_train=args.frac_train
    )

    # the two curves that make the grokking plot, plus test accuracy
    history = {
        "step": [],
        "train_loss": [],
        "test_loss": [],
        "test_accuracy": [],
        "test_seq_accuracy": [],
        # grokking is defined by the gap between train and test, so log both
        "train_accuracy": [],
        "train_seq_accuracy": [],
    }

    # training
    optimizer.zero_grad()
    progress_bar = tqdm(range(args.epochs))
    try:
      for epoch in progress_bar:
        loss = trainer.training_step(train)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
        # validation
        if epoch % args.eval_every == 0 or epoch == args.epochs - 1:
            with t.inference_mode():
                test_loss, test_accuracy, test_seq = trainer.validation_step(test)
                _, train_accuracy, train_seq = trainer.validation_step(train)
            if args.use_wandb:
                _wandb().log(
                    {
                        "train_loss": loss.item(),
                        "test_loss": test_loss,
                        "test_accuracy": test_accuracy,
                        "test_seq_accuracy": test_seq,
                        "train_accuracy": train_accuracy,
                        "train_seq_accuracy": train_seq,
                    },
                    step=epoch,
                )
            history["step"].append(epoch)
            history["train_loss"].append(loss.item())
            history["test_loss"].append(test_loss)
            history["test_accuracy"].append(test_accuracy)
            history["test_seq_accuracy"].append(test_seq)
            history["train_accuracy"].append(train_accuracy)
            history["train_seq_accuracy"].append(train_seq)

            if ckpt_dir is not None:
                # save on improvement: these runs oscillate, so the final step is
                # not reliably the best model
                if test_seq > best:
                    best = test_seq
                    t.save({"state_dict": model.state_dict(), "step": epoch,
                            "test_seq_accuracy": test_seq, "config": asdict(args)},
                           ckpt_dir / f"{args.run_name}_best.pt")
                if ckpt_every and epoch % ckpt_every == 0:
                    t.save({"state_dict": model.state_dict(), "step": epoch,
                            "test_seq_accuracy": test_seq, "config": asdict(args)},
                           ckpt_dir / f"{args.run_name}_step{epoch:06d}.pt")
            progress_bar.set_description(
                f"train {loss.item():.4f} | test {test_loss:.4f} | "
                f"seq {test_seq:.3f} | digit {test_accuracy:.3f}"
            )
    except KeyboardInterrupt:
        print(f"interrupted at step {epoch}")
    finally:
        if args.use_wandb:
            _wandb().finish()

    return trainer.model, history


def parse_args() -> tuple[TrainArgs, Path, Path | None, int]:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frac-train", type=float, required=True)
    p.add_argument("--weight-decay", type=float, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--steps", type=int, default=100_000,
                   help="every run in results/sweep used the default")
    p.add_argument("--num-digits", type=int, default=2)
    p.add_argument("--n-layers", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--out-dir", type=Path, default=Path("results"))
    p.add_argument("--ckpt-dir", type=Path, default=None,
                   help="save weights here; omit to skip checkpointing entirely")
    p.add_argument("--ckpt-every", type=int, default=0,
                   help="also snapshot every N steps (0 = only save on improvement)")
    p.add_argument("--wandb", action="store_true", help="off by default: exec nodes often have no outbound network")
    a = p.parse_args()
    args = TrainArgs(
        num_digits=a.num_digits,
        n_layers=a.n_layers,
        frac_train=a.frac_train,
        epochs=a.steps,
        lr=a.lr,
        weight_decay=a.weight_decay,
        seed=a.seed,
        eval_every=a.eval_every,
        use_wandb=a.wandb,
        device=str(DEVICE),
    )
    return args, a.out_dir, a.ckpt_dir, a.ckpt_every


if __name__ == "__main__":
    args, out_dir, ckpt_dir, ckpt_every = parse_args()
    out_dir.mkdir(parents=True, exist_ok=True)
    if ckpt_dir is not None:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"history_{args.run_name}.json"
    print(f"device={DEVICE}  run={args.run_name}  -> {out}", flush=True)

    model = create_model(args)
    model, history = train_model(model, args, ckpt_dir=ckpt_dir, ckpt_every=ckpt_every)

    json.dump({"config": asdict(args), "history": history}, open(out, "w"))
    seq = history["test_seq_accuracy"]
    print(
        f"done  steps={history['step'][-1]}  "
        f"test_seq={seq[-1]:.4f}  max_test_seq={max(seq):.4f}  "
        f"train_seq={history['train_seq_accuracy'][-1]:.4f}",
        flush=True,
    )
