# Copyright Pathway Technology, Inc.

import os
from contextlib import nullcontext
from pathlib import Path

import jdh
import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import math

from config import (
    BLOCK_SIZE,
    TARGET_SIZE,
    BATCH_SIZE,
    MAX_ITERS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    LOG_FREQ,
    EMA_DECAY,
    L_RELEVANCE
)
from decoder_model import Decoder

'''
TODO: still model collapse

1. ?opt? use different loss?
2. 
'''


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# On a Mac you can also try
# device=torch.device('mps')


PROJECT_ROOT = Path(__file__).resolve().parents[3]
# here ~/Dev/pybullet/src/rl/JDH/train.py so PROJECT_ROOT is ~/Dev/pybullet

### adds environment variables from .env.development file to os.environ
def _load_env_file() -> None:
    env_path = PROJECT_ROOT / ".env.development"
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'") 
        if key and key not in os.environ:
            os.environ[key] = value

# funcion resolves the path
def _resolve_path(env_name: str, default_path: Path) -> Path:
    raw_value = os.getenv(env_name)
    if not raw_value:
        return default_path

    candidate = Path(raw_value).expanduser()
    if not candidate.is_absolute():
        candidate = (PROJECT_ROOT / candidate).resolve()
    return candidate


_load_env_file()

#dtype is the data type of the model parameters
dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler


#ptdtype is the data type of the model parameters in PyTorch
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype] # map of dtypes

# mixed operwation autocast in nvidia devices, enables faster computation dynamically switching between float16 and 32
ctx = (
    torch.amp.autocast(device_type=device.type, dtype=ptdtype)
    if "cuda" in device.type
    else nullcontext()
) 
# "automatic mixed precision", uses scaling to prevent underflow
scaler = torch.amp.GradScaler(device=device.type, enabled=(dtype == "float16"))
torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
print(f"Using device: {device} with dtype {dtype}")

JDH_CONFIG = jdh.JDHConfig()
# Training/model constants live in config.py (single source of truth).
# Imported above: BLOCK_SIZE, TARGET_SIZE, BATCH_SIZE, MAX_ITERS,
# LEARNING_RATE, WEIGHT_DECAY, LOG_FREQ, EMA_DECAY.

INPUT_FILE_PATH = _resolve_path("JDH_INPUT_FILE", Path(__file__).resolve().parent / "input.txt")
MODEL_PATH = _resolve_path("JDH_MODEL_PATH", Path(__file__).resolve().parent / "parameters" / "JDH_model.pt")


# Fetch the tiny Shakespeare dataset
def fetch_data():
    if not INPUT_FILE_PATH.exists():
        data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        INPUT_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(INPUT_FILE_PATH, "w", encoding="utf-8") as f:
            f.write(requests.get(data_url).text)


def get_batch(split):
    # x is the context chunk, y is the DISJOINT target chunk that follows it
    # (no overlap -> the JEPA task is not trivial)
    data = np.memmap(INPUT_FILE_PATH, dtype=np.uint8, mode="r")
    if split == "train":
        data = data[: int(0.9 * len(data))]
    else:
        data = data[int(0.9 * len(data)) :]
    ix = torch.randint(len(data) - BLOCK_SIZE - TARGET_SIZE, (BATCH_SIZE,))
    x = torch.stack(
        [torch.from_numpy((data[i : i + BLOCK_SIZE]).astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy(
                (data[i + BLOCK_SIZE : i + BLOCK_SIZE + TARGET_SIZE]).astype(np.int64)
            )
            for i in ix
        ]
    )
    if torch.cuda.is_available():
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
            device, non_blocking=True
        )
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def eval(model):
    model.eval()

def iLoss(pred, target):
    return F.smooth_l1_loss(pred, target)

# def vLoss(pred, target_std=1.0):
#     std = torch.std(pred, dim=(-1), unbiased=True)
#     if std < target_std:

def vLoss(pred, target_var=1.0): #VicReg style by deepseek
    pred = pred.reshape(-1, pred.shape[-1]) #D
    std = torch.sqrt(torch.var(pred, dim=0, unbiased=True) + 1e-06) #+1e-04 to avoid div by 0 at sqrt
    return F.relu(target_var - std).pow(2).mean()

def cLoss(pred, target):
    std_pred = torch.std(pred, dim=(-1), unbiased=True)
    std_target = torch.std(target, dim=(-1), unbiased=True)
    loss = (std_pred*std_target).mean()
    return loss**2

if __name__ == "__main__":
    fetch_data()

    #decoder for encoding
    decoder = Decoder(JDH_CONFIG).to(device)

    # target encoder: EMA copy of the online model, created BEFORE torch.compile
    # so we deepcopy a plain JDH, not an OptimizedModule
    model = jdh.JDH(JDH_CONFIG).to(device)
    target_encoder = copy.deepcopy(model).to(device)
    target_encoder.eval()
    for p in target_encoder.parameters():
        p.requires_grad_(False)

    model = torch.compile(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    # separate optimizer for the decoder
    decoder_optimizer = torch.optim.AdamW(
        decoder.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = torch.nn.L1Loss()

    @torch.no_grad()
    def ema_update():
        for p_online, p_target in zip(model.parameters(), target_encoder.parameters()):
            p_target.mul_(EMA_DECAY).add_(p_online.detach(), alpha=1 - EMA_DECAY)

    x, y = get_batch("train")

    loss_decoder = 0
    loss_acc = 0
    loss_steps = 0
    dec_correct = 0 # number of correct predictions by the decoder
    dec_tokens = 0
    for step in range(MAX_ITERS):
        optimizer.zero_grad(set_to_none=True)
        decoder_optimizer.zero_grad(set_to_none=True)

        with ctx, torch.no_grad(): #freeze encoder
            # target embeddings: B, 1, T_tgt, D -> B, T_tgt, D (no grad!)
            target = target_encoder(y).squeeze(1) #used to be y

        with ctx:
            pred = model.predict(model(x))  # B, T_tgt, D

        pred = pred.float()

        # Training on both predicted and target. 
        tgt_logits, tgt_dec_loss = decoder(target, y)
        pred_logits, pred_dec_loss = decoder(pred.detach(), y)
        dec_loss = 0.5 * (tgt_dec_loss + pred_dec_loss)
        loss_decoder += dec_loss.detach()
        # decoder telemetry: token accuracy over the target chunk
        dec_correct += (pred_logits.detach().argmax(dim=-1) == y).sum().item()
        dec_tokens += y.numel()

        #pred = pred
        target = target.float()

        """
        TODO: implement vic loss here
        """
        loss = (
            L_RELEVANCE[0] * iLoss(pred, target)
            + L_RELEVANCE[1] * vLoss(pred, target)
        )


        loss_acc += loss
        loss_steps += 1

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # decoder trains on its own loss (tangent to the JEPA objective)
        scaler.scale(dec_loss).backward()
        scaler.step(decoder_optimizer)
        ema_update()

        x, y = get_batch("train")

        if step % LOG_FREQ == 0:
            dec_ce = loss_decoder.item() / loss_steps #ce stands for cross entropy
            dec_acc = dec_correct / max(dec_tokens, 1)
            print(
                f"Step: {step}/{MAX_ITERS} "
                f"jepa {loss_acc.item() / loss_steps:.3} | "
                f"decoder ce {dec_ce:.3} acc {dec_acc:.3} ppl {math.exp(min(dec_ce, 20)):.1}"
            )
            loss_acc = 0
            loss_steps = 0
            loss_decoder = 0
            dec_correct = 0
            dec_tokens = 0

    print("Training done, saving model")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    # torch.save(model.state_dict(), MODEL_PATH)
    # torch.save(decoder.state_dict(), MODEL_PATH.with_name(MODEL_PATH.stem + "_decoder.pt"))
    torch.save(model.state_dict(), MODEL_PATH)
    torch.save(decoder.state_dict(), MODEL_PATH.with_name(MODEL_PATH.stem + "_decoder.pt"))
    print(f"Model saved to {MODEL_PATH}, decoder saved to {MODEL_PATH.with_name(MODEL_PATH.stem + '_decoder.pt')}")


    # JEPA models have no output head, so qualitative eval = compare
    # predicted vs. target embeddings for a held-out chunk.
    model.eval()
    decoder.eval()
    x, y = get_batch("val")
    with torch.no_grad(), ctx:
        pred = model.predict(model(x))
        target = target_encoder(y).squeeze(1)
        # decoder telemetry on held-out data: can the probing head read the
        # target embeddings, and can it read the *predicted* embeddings?
        tgt_logits, _ = decoder(target, y)
        pred_logits, _ = decoder(pred, y)
    dist = (pred - target).abs().mean()
    cos = F.cosine_similarity(pred, target, dim=-1).mean()
    tgt_ce = F.cross_entropy(tgt_logits.view(-1, tgt_logits.size(-1)), y.view(-1))
    pred_ce = F.cross_entropy(pred_logits.view(-1, pred_logits.size(-1)), y.view(-1))
    tgt_acc = (tgt_logits.argmax(dim=-1) == y).float().mean()
    pred_acc = (pred_logits.argmax(dim=-1) == y).float().mean()
    print(f"val L1 distance: {dist.item():.4f}, cosine similarity: {cos.item():.4f}")
    print(
        f"decoder on target emb:  ce {tgt_ce.item():.3} acc {tgt_acc.item():.3} "
        f"ppl {math.exp(min(tgt_ce.item(), 20)):.1}"
    )
    print(
        f"decoder on predicted emb: ce {pred_ce.item():.3} acc {pred_acc.item():.3} "
        f"ppl {math.exp(min(pred_ce.item(), 20)):.1}"
    )