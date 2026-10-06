# Copyright 2025 Pathway Technology, Inc.

import dataclasses
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import mode, nn

from config import TARGET_SIZE


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
MODEL_PATH = _resolve_path("JDH_MODEL_PATH", Path(__file__).resolve().parent / "parameters" / "JDH_model.pt")


@dataclasses.dataclass
class JDHConfig:
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

### simple JEPA style predictor
# takes x of form B,1,T,D
# per batch operation,
# attention already merged and only single target, so DxD for each B
# Returns B,T,D vector
class Predictor(torch.nn.Module):
    def __init__(self, config, target_T: int = TARGET_SIZE):
        super().__init__()
        self.config = config
        D = config.n_embd #embedding dimension
        self.t_tgt = target_T #target sequence length

        # self.pos_embed = nn.Parameter(torch.zeros(1, self.t_tgt, D).normal_(std=0.02)) #(B,1,T,D)
        self.net = nn.Sequential(
            # nn.LayerNorm(D),
            nn.Linear(D, D), 
            nn.GELU(), 
            nn.Linear(D, D),
            # nn.GELU()
            )

        #self attention
        self.Wq = nn.Linear(D, D, bias=False)
        self.Wk = nn.Linear(D, D, bias=False)
        self.Wv = nn.Linear(D, D, bias=False)

    #takes B,T,D
    def attend(self, x):
        k = self.Wk(x)
        v = self.Wv(x)
        q = self.Wq(x)

        attn_scores = q @ torch.transpose(k, -2, -1) / math.sqrt(k.size(-1))
        #attending to only previous tokens since this is autoregressive
        upper_triangular  = torch.triu(attn_scores, diagonal=1).bool()
        attn_scores[upper_triangular] = float("-inf")
        # Apply softmax to get attention weights
        att_score_softmax = F.softmax(attn_scores, dim=-1)
        weighted_v = att_score_softmax @ v
        return weighted_v

    def forward(self, x): #B,1,T,D
        x = x.squeeze(1) #B,T,D
        h = self.attend(x) #B,T,D
        h = h[:, -self.t_tgt:] #match expected shape
        out = self.net(h) #B,T,D
        return out

class Attention(torch.nn.Module): #Attention takes
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

#https://medium.com/@heyamit10/exponential-moving-average-ema-in-pytorch-eb8b6f1718eb
class EMA(torch.nn.Module):
    def __init__(self, model, decay):
        """
        Initialize EMA class to manage exponential moving average of model parameters.
        
        Args:
            model (torch.nn.Module): The model for which EMA will track parameters.
            decay (float): Decay rate, typically a value close to 1, e.g., 0.999.
        """
        super().__init__()
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # Store initial parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """
        Update shadow parameters with exponential decay.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """
        Apply shadow (EMA) parameters to model.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """
        Restore original model parameters from backup.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]

class JDH(nn.Module):
    def __init__(self, config: JDHConfig):
        super().__init__()
        assert config.vocab_size is not None
        self.config = config
        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh
        self.decoder = nn.Parameter(torch.zeros((nh * N, D)).normal_(std=0.02))
        self.encoder = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))

        self.attn = Attention(config)
        self.pred = Predictor(config)
        # NOTE: the EMA target encoder lives in train.py (a frozen deepcopy of
        # the model). Do NOT attach an EMA module here: registering the model
        # as a submodule of itself creates a parameter-registration cycle.

        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.embed = nn.Embedding(config.vocab_size, D)
        self.drop = nn.Dropout(config.dropout)
        self.encoder_v = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))

        # self.lm_head = nn.Parameter(
        #     torch.zeros((D, config.vocab_size)).normal_(std=0.02)
        # )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        C = self.config

        B, T = idx.size() #batch size and sequence length
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

            #merg attention heads
            yMLP = (
                xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ self.decoder
            )  # B, 1, T, D
            y = self.ln(yMLP)
            x = self.ln(x + y) #residual connection not in graph
        return x

    def predict(self, x):
        return self.pred(x)

    ###TODO: remove logits for jdh <done>
    ###TODO: make encoder for the target vectors <done: EMA copy in train.py>
    ###TODO: make separate decoder for prediction <done: Predictor>
    ###TODO: remove generate <done>
    ###TODO: patch what doesn't work <done: see train.py>

if __name__ == "__main__":
    config = JDHConfig()
    model = JDH(config)
    print(model)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    if MODEL_PATH.exists():
        state_dict = torch.load(MODEL_PATH, map_location=device)
        state_dict = {
            k.replace("_orig_mod.", ""): v for k, v in state_dict.items()
        }
        model.load_state_dict(state_dict)
    else:
        print(f"No trained model found at {MODEL_PATH}; using random init.")

    model.eval()

    # JEPA models have no output head, so qualitative eval = compare
    # predicted vs. target embeddings for a held-out chunk.
    prompt = torch.tensor(
        bytearray("To be or ", "utf-8"), dtype=torch.long, device=device
    ).unsqueeze(0)
    with torch.no_grad():
        emb = model(prompt)
        pred = model.predict(emb)
    print(f"context embedding: {tuple(emb.shape)}, predicted target embedding: {tuple(pred.shape)}")
