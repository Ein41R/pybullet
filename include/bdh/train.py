# Copyright Pathway Technology, Inc.

import os
from contextlib import nullcontext
from pathlib import Path

import bdh
import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# On a Mac you can also try
# device=torch.device('mps')


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_env_file() -> None:
    env_path = PROJECT_ROOT / ".env"
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
) # does nothing if no nvidia device, otherwise enables mixed precision training for faster computation and lower memory usage.

# scaler does scalar scaling of gradients to prevent underflow in flaot16 used in combination with amp.autocast
scaler = torch.amp.GradScaler(device=device.type, enabled=(dtype == "float16"))
torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
print(f"Using device: {device} with dtype {dtype}")


# Configuration
BDH_CONFIG = bdh.BDHConfig()
BLOCK_SIZE = 512
BATCH_SIZE = 32
MAX_ITERS = 2300
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.1
LOG_FREQ = 100

INPUT_FILE_PATH = _resolve_path("BDH_INPUT_FILE", Path(__file__).resolve().parent / "input.txt")
MODEL_PATH = _resolve_path("BDH_MODEL_PATH", Path(__file__).resolve().parent / "parameters" / "bdh_model.pt")


# Fetch the tiny Shakespeare dataset
def fetch_data():
    if not INPUT_FILE_PATH.exists(): #if no trainng text given.
        data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        INPUT_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(INPUT_FILE_PATH, "w", encoding="utf-8") as f:
            f.write(requests.get(data_url).text)


def get_batch(split):
    # treat the file as bytes
    data = np.memmap(INPUT_FILE_PATH, dtype=np.uint8, mode="r")
    if split == "train":
        data = data[: int(0.9 * len(data))] #rabdomly select 90% of the data for training
    else:
        data = data[int(0.9 * len(data)) :] #use the remaining 10% for validation
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack(
        [torch.from_numpy((data[i : i + BLOCK_SIZE]).astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy((data[i + 1 : i + 1 + BLOCK_SIZE]).astype(np.int64))
            for i in ix
        ]
    )
    if torch.cuda.is_available():
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
            device, non_blocking=True
        )
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def eval(model):
    model.eval()


if __name__ == "__main__":
    fetch_data()

    model = bdh.BDH(BDH_CONFIG).to(device) #.to(device) converts from torch tensor to device/cuda tensor
    model = torch.compile(model) # make torch faster
    optimizer = torch.optim.AdamW( # Adam optimizer, finds location of minimum loss faster.
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    x, y = get_batch("train")

    loss_acc = 0
    loss_steps = 0
    for step in range(MAX_ITERS):
        optimizer.zero_grad(set_to_none=True)
        with ctx: # when context is nullcontext skips. otherwise loads the model in amp mode.
            logits, loss = model(x, y)
        x, y = get_batch("train") # if splitting enabled
        loss_acc += loss #add loss to loss accumulator for logging
        loss_steps += 1 # increment step counter
        scaler.scale(loss).backward() # scale the loss and compute gradients
        scaler.step(optimizer) # update params with adam ykyk
        scaler.update() #since scalar isnt constant, update it
        if step % LOG_FREQ == 0: # log every LOG_FREQ steps aka every 100 of 2300 i think
            print(f"Step: {step}/{MAX_ITERS} loss {loss_acc.item() / loss_steps:.3}")
            loss_acc = 0 #reset accumulated loss
            loss_steps = 0 #reset step counter
    print("Training done, now generating a sample ")
    model.eval() # set model to evaluation mode, disables dropout and other training specific layers
    prompt = torch.tensor(
        bytearray("To be or ", "utf-8"), dtype=torch.long, device=device # correct would be "To be or not to be,", let the model predict that
    ).unsqueeze(0)
    ret = model.generate(prompt, max_new_tokens=100, top_k=3)
    ret_decoded = bytes(ret.to(torch.uint8).to("cpu").squeeze(0)).decode(
        errors="backslashreplace"
    )
    print(ret_decoded)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH) # save the model to disk