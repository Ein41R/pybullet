# Copyright 2025 Pathway Technology, Inc.

import dataclasses
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import mode, nn


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
MODEL_PATH = _resolve_path("BDH_MODEL_PATH", Path(__file__).resolve().parent / "parameters" / "bdh_model.pt")


@dataclasses.dataclass
class BDHConfig:
    n_layer: int = 4#6
    n_embd: int = 128#256
    dropout: float = 0.1
    n_head: int = 4
    mlp_internal_dim_multiplier: int = 32#128
    vocab_size: int = 256

#--> N = n_embd × mlp_internal_dim_multiplier
#--> params ≈ k × n_embd × N
#--> param_bytes ≈ params × bytes_per_element
#--> total_training_param_memory ≈ param_bytes × 4   (roughly, for Adam)
#--> activation_bytes_per_tensor ≈ batch_size × block_size × N × bytes_per_element
#--> total_activation_memory ≈ batch_size × block_size × N × bytes_per_element × tensors_per_layer × n_layer
#--> total_activation_memory_with_checkpointing ≈ batch_size × block_size × N × bytes_per_element × tensors_per_layer
#--> attention_bytes ≈ batch_size × n_head × block_size² × bytes_per_element
#--> total_memory ≈ total_training_param_memory + total_activation_memory + attention_bytes + framework_overhead
#--> 32 × 512 × 32,768 × 4 = 2,147,483,648 bytes

def get_freqs(n, theta, dtype):
    def quantize(t, q=2):
        return (t / q).floor() * q

    return (
        1.0
        / (theta ** (quantize(torch.arange(0, n, 1, dtype=dtype)) / n))
        / (2 * math.pi)
    )


class Attention(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        nh = config.n_head #define number of attention heads
        D = config.n_embd #define embedding dimension
        N = config.mlp_internal_dim_multiplier * D // nh #define internal dimension of the MLP --> look at grah on github
        self.freqs = torch.nn.Buffer(
            get_freqs(N, theta=2**16, dtype=torch.float32).view(1, 1, 1, N)
        )

    @staticmethod
    def phases_cos_sin(phases):
        phases = (phases % 1) * (2 * math.pi)
        phases_cos = torch.cos(phases)
        phases_sin = torch.sin(phases)
        return phases_cos, phases_sin

    @staticmethod
    def rope(phases, v):
        v_rot = torch.stack((-v[..., 1::2], v[..., ::2]), dim=-1).view(*v.size())
        phases_cos, phases_sin = Attention.phases_cos_sin(phases)
        return (v * phases_cos).to(v.dtype) + (v_rot * phases_sin).to(v.dtype)

    def forward(self, Q, K, V):
        assert self.freqs.dtype == torch.float32
        assert K is Q #throws error when K is not equal to Q
        _, _, T, _ = Q.size() #Q contains the query vectors, T is the sequence length

        r_phases = (
            torch.arange(
                0,
                T,
                device=self.freqs.device,
                dtype=self.freqs.dtype,
            ).view(1, 1, -1, 1)
        ) * self.freqs
        QR = self.rope(r_phases, Q)
        KR = QR

        # Current attention
        scores = (QR @ KR.mT).tril(diagonal=-1)
        return scores @ V


class BDH(nn.Module):
    def __init__(self, config: BDHConfig):
        super().__init__()
        assert config.vocab_size is not None
        self.config = config
        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh
        self.decoder = nn.Parameter(torch.zeros((nh * N, D)).normal_(std=0.02))
        self.encoder = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))

        self.attn = Attention(config)

        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.embed = nn.Embedding(config.vocab_size, D)
        self.drop = nn.Dropout(config.dropout)
        self.encoder_v = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))

        self.lm_head = nn.Parameter(
            torch.zeros((D, config.vocab_size)).normal_(std=0.02)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        C = self.config

        B, T = idx.size()
        D = C.n_embd
        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh

        x = self.embed(idx).unsqueeze(1)

        # actually helps with training
        x = self.ln(x)  # B, 1, T, D

        for level in range(C.n_layer):
            x_latent = x @ self.encoder

            x_sparse = F.relu(x_latent)  # B, nh, T, N

            yKV = self.attn(
                Q=x_sparse,
                K=x_sparse,
                V=x,
            )
            yKV = self.ln(yKV)

            y_latent = yKV @ self.encoder_v
            y_sparse = F.relu(y_latent)
            xy_sparse = x_sparse * y_sparse  # B, nh, T, N

            xy_sparse = self.drop(xy_sparse)

            yMLP = (
                xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ self.decoder
            )  # B, 1, T, D
            y = self.ln(yMLP)
            x = self.ln(x + y)

        logits = x.view(B, T, D) @ self.lm_head
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor, #associative memory tensor of the model
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

if __name__ == "__main__":
    config = BDHConfig()
    model = BDH(config)
    print(model)

    MAX_TOKENS = 100
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    state_dict = torch.load(MODEL_PATH)


    state_dict = {
    k.replace("_orig_mod.", ""):
    v for k, v in state_dict.items()
    } 

    model.load_state_dict(state_dict)
    model.eval() # set model to evaluation mode, disables dropout and other training specific layers

    print("Generating a sample from the model...")
    prompt = torch.tensor(
        bytearray("To be or ", "utf-8"), dtype=torch.long, device=device # correct would be "To be or not to be,", let the model predict that
    ).unsqueeze(0)
    ret = model.generate(prompt, max_new_tokens=100, top_k=3)
    ret_decoded = bytes(ret.to(torch.uint8).to("cpu").squeeze(0)).decode(
        errors="backslashreplace"
    )
    print(ret_decoded)
    
    while True:
        user_prompt = input("\033[36mEnter a prompt (or 'exit' to quit): \033[0m")
        if user_prompt.lower() == "exit":
            break
        prompt = torch.tensor(
            bytearray(user_prompt, "utf-8"), dtype=torch.long, device=device
        ).unsqueeze(0)
        ret = model.generate(prompt, max_new_tokens=100, top_k=3)
        ret_decoded = bytes(ret.to(torch.uint8).to("cpu").squeeze(0)).decode(
            errors="backslashreplace"
        )
        print(ret_decoded)
