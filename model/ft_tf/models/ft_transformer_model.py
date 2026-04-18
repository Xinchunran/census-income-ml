
"""
FT-Transformer model for binary classification on mixed-type tabular data.

Designed for the census-income take-home project:
- Inputs:
    x_num: float tensor of shape [batch_size, n_num_features]
    x_cat: long tensor of shape [batch_size, n_cat_features]
- Output:
    logits: float tensor of shape [batch_size]
- Supports:
    - categorical embeddings
    - numerical feature tokenization
    - CLS token
    - transformer encoder blocks
    - BCEWithLogitsLoss
    - optional sample weights
    - early stopping on validation loss

This implementation is intentionally self-contained and avoids extra third-party
tabular libraries so it can be audited and modified easily.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import math
import copy
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

class TabularTensorDataset(Dataset):
    def __init__(
        self,
        x_num: Optional[torch.Tensor],
        x_cat: Optional[torch.Tensor],
        y: Optional[torch.Tensor] = None,
        sample_weight: Optional[torch.Tensor] = None,
    ) -> None:
        if x_num is None and x_cat is None:
            raise ValueError("At least one of x_num or x_cat must be provided.")

        n = x_num.shape[0] if x_num is not None else x_cat.shape[0]
        if x_num is not None and x_num.shape[0] != n:
            raise ValueError("x_num has inconsistent number of rows.")
        if x_cat is not None and x_cat.shape[0] != n:
            raise ValueError("x_cat has inconsistent number of rows.")
        if y is not None and y.shape[0] != n:
            raise ValueError("y has inconsistent number of rows.")
        if sample_weight is not None and sample_weight.shape[0] != n:
            raise ValueError("sample_weight has inconsistent number of rows.")

        self.x_num = x_num
        self.x_cat = x_cat
        self.y = y
        self.sample_weight = sample_weight

    def __len__(self) -> int:
        return self.x_num.shape[0] if self.x_num is not None else self.x_cat.shape[0]

    def __getitem__(self, idx: int):
        item = {
            "x_num": None if self.x_num is None else self.x_num[idx],
            "x_cat": None if self.x_cat is None else self.x_cat[idx],
        }
        if self.y is not None:
            item["y"] = self.y[idx]
        if self.sample_weight is not None:
            item["sample_weight"] = self.sample_weight[idx]
        return item


# ---------------------------------------------------------------------
# FT-Transformer building blocks
# ---------------------------------------------------------------------

class NumericalFeatureTokenizer(nn.Module):
    """
    Converts each numerical feature into a token of size d_token.

    For each scalar feature x_j, token_j = x_j * w_j + b_j
    where w_j and b_j are learnable vectors of length d_token.
    """
    def __init__(self, n_num_features: int, d_token: int) -> None:
        super().__init__()
        self.n_num_features = n_num_features
        self.d_token = d_token
        self.weight = nn.Parameter(torch.empty(n_num_features, d_token))
        self.bias = nn.Parameter(torch.empty(n_num_features, d_token))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x_num: torch.Tensor) -> torch.Tensor:
        if x_num.ndim != 2:
            raise ValueError(f"x_num must be 2D, got shape {tuple(x_num.shape)}")
        if x_num.shape[1] != self.n_num_features:
            raise ValueError(
                f"Expected {self.n_num_features} numerical features, got {x_num.shape[1]}"
            )
        # [B, N_num, 1] * [N_num, d] -> [B, N_num, d]
        return x_num.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class CategoricalFeatureTokenizer(nn.Module):
    """
    Embeds each categorical feature into a token of size d_token.

    Each categorical column has its own embedding table. Inputs should already be
    integer-encoded from 0..cardinality-1, with a dedicated code for unknown if needed.
    """
    def __init__(self, cat_cardinalities: List[int], d_token: int) -> None:
        super().__init__()
        self.cat_cardinalities = cat_cardinalities
        self.d_token = d_token
        self.embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, d_token) for cardinality in cat_cardinalities]
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for emb in self.embeddings:
            nn.init.xavier_uniform_(emb.weight)

    def forward(self, x_cat: torch.Tensor) -> torch.Tensor:
        if x_cat.ndim != 2:
            raise ValueError(f"x_cat must be 2D, got shape {tuple(x_cat.shape)}")
        if x_cat.shape[1] != len(self.embeddings):
            raise ValueError(
                f"Expected {len(self.embeddings)} categorical features, got {x_cat.shape[1]}"
            )
        tokens = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        return torch.stack(tokens, dim=1)  # [B, N_cat, d]


class GEGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return x * torch.nn.functional.gelu(gate)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_token: int,
        n_heads: int,
        attention_dropout: float,
        ffn_dropout: float,
        residual_dropout: float,
        d_ffn_factor: float = 4.0 / 3.0,
    ) -> None:
        super().__init__()
        if d_token % n_heads != 0:
            raise ValueError("d_token must be divisible by n_heads.")

        self.norm1 = nn.LayerNorm(d_token)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_token,
            num_heads=n_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(residual_dropout)

        d_hidden = int(d_token * d_ffn_factor)
        self.norm2 = nn.LayerNorm(d_token)
        self.ffn = nn.Sequential(
            nn.Linear(d_token, d_hidden * 2),
            GEGLU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(d_hidden, d_token),
        )
        self.dropout2 = nn.Dropout(residual_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_input = self.norm1(x)
        attn_out, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        x = x + self.dropout1(attn_out)

        ffn_input = self.norm2(x)
        ffn_out = self.ffn(ffn_input)
        x = x + self.dropout2(ffn_out)
        return x


class FTTransformer(nn.Module):
    def __init__(
        self,
        n_num_features: int,
        cat_cardinalities: Optional[List[int]],
        d_token: int = 192,
        n_blocks: int = 3,
        n_heads: int = 8,
        attention_dropout: float = 0.2,
        ffn_dropout: float = 0.1,
        residual_dropout: float = 0.0,
        head_hidden_dim: int = 128,
        head_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_num_features = n_num_features
        self.cat_cardinalities = cat_cardinalities or []

        self.num_tokenizer = (
            NumericalFeatureTokenizer(n_num_features, d_token)
            if n_num_features > 0
            else None
        )
        self.cat_tokenizer = (
            CategoricalFeatureTokenizer(self.cat_cardinalities, d_token)
            if len(self.cat_cardinalities) > 0
            else None
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_token=d_token,
                    n_heads=n_heads,
                    attention_dropout=attention_dropout,
                    ffn_dropout=ffn_dropout,
                    residual_dropout=residual_dropout,
                )
                for _ in range(n_blocks)
            ]
        )
        self.final_norm = nn.LayerNorm(d_token)
        self.head = nn.Sequential(
            nn.Linear(d_token, head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden_dim, 1),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.cls_token, std=0.02)
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        x_num: Optional[torch.Tensor] = None,
        x_cat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        tokens = []

        if self.num_tokenizer is not None:
            if x_num is None:
                raise ValueError("Model expects numerical features but x_num is None.")
            tokens.append(self.num_tokenizer(x_num))

        if self.cat_tokenizer is not None:
            if x_cat is None:
                raise ValueError("Model expects categorical features but x_cat is None.")
            tokens.append(self.cat_tokenizer(x_cat))

        if not tokens:
            raise ValueError("No input tokens available. Provide x_num and/or x_cat.")

        x = torch.cat(tokens, dim=1)  # [B, N_tokens, d]
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)  # prepend CLS token

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        cls_repr = x[:, 0, :]
        logits = self.head(cls_repr).squeeze(-1)
        return logits


# ---------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------

@dataclass
class TrainingConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 1024
    max_epochs: int = 100
    patience: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    positive_class_weight: Optional[float] = None
    grad_clip_norm: Optional[float] = 1.0
    num_workers: int = 0
    verbose: bool = True


def _move_batch_to_device(batch: Dict[str, Optional[torch.Tensor]], device: str):
    out = {}
    for k, v in batch.items():
        if v is None:
            out[k] = None
        else:
            out[k] = v.to(device)
    return out


def _weighted_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    sample_weight: Optional[torch.Tensor] = None,
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    per_item = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
        pos_weight=pos_weight,
    )
    if sample_weight is not None:
        weighted = per_item * sample_weight
        denom = sample_weight.sum().clamp_min(1e-12)
        return weighted.sum() / denom
    return per_item.mean()


@torch.no_grad()
def predict_logits(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> np.ndarray:
    model.eval()
    outputs: List[np.ndarray] = []
    for batch in loader:
        batch = _move_batch_to_device(batch, device)
        logits = model(batch["x_num"], batch["x_cat"])
        outputs.append(logits.detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


@torch.no_grad()
def predict_proba(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> np.ndarray:
    logits = predict_logits(model, loader, device)
    return 1.0 / (1.0 + np.exp(-logits))


def fit_ft_transformer(
    model: FTTransformer,
    train_loader: DataLoader,
    valid_loader: Optional[DataLoader],
    config: TrainingConfig,
) -> Tuple[FTTransformer, Dict[str, List[float]]]:
    device = config.device
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    pos_weight_tensor = None
    if config.positive_class_weight is not None:
        pos_weight_tensor = torch.tensor(
            config.positive_class_weight,
            dtype=torch.float32,
            device=device,
        )

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "valid_loss": [],
    }

    best_state = None
    best_valid_loss = math.inf
    epochs_without_improvement = 0

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        running_loss = 0.0
        running_weight = 0.0

        for batch in train_loader:
            batch = _move_batch_to_device(batch, device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["x_num"], batch["x_cat"])
            loss = _weighted_bce_with_logits(
                logits=logits,
                targets=batch["y"].float(),
                sample_weight=batch.get("sample_weight"),
                pos_weight=pos_weight_tensor,
            )
            loss.backward()

            if config.grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)

            optimizer.step()

            batch_weight = (
                batch["sample_weight"].sum().item()
                if batch.get("sample_weight") is not None
                else logits.shape[0]
            )
            running_loss += loss.item() * batch_weight
            running_weight += batch_weight

        train_loss = running_loss / max(running_weight, 1e-12)
        history["train_loss"].append(train_loss)

        if valid_loader is not None:
            model.eval()
            valid_running_loss = 0.0
            valid_running_weight = 0.0

            for batch in valid_loader:
                batch = _move_batch_to_device(batch, device)
                logits = model(batch["x_num"], batch["x_cat"])
                loss = _weighted_bce_with_logits(
                    logits=logits,
                    targets=batch["y"].float(),
                    sample_weight=batch.get("sample_weight"),
                    pos_weight=pos_weight_tensor,
                )

                batch_weight = (
                    batch["sample_weight"].sum().item()
                    if batch.get("sample_weight") is not None
                    else logits.shape[0]
                )
                valid_running_loss += loss.item() * batch_weight
                valid_running_weight += batch_weight

            valid_loss = valid_running_loss / max(valid_running_weight, 1e-12)
            history["valid_loss"].append(valid_loss)

            if config.verbose:
                print(
                    f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | "
                    f"valid_loss={valid_loss:.6f}"
                )

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= config.patience:
                if config.verbose:
                    print(f"Early stopping at epoch {epoch}.")
                break
        else:
            if config.verbose:
                print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


# ---------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------

def make_dataloader(
    x_num: Optional[np.ndarray],
    x_cat: Optional[np.ndarray],
    y: Optional[np.ndarray] = None,
    sample_weight: Optional[np.ndarray] = None,
    batch_size: int = 1024,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    x_num_tensor = None if x_num is None else torch.tensor(x_num, dtype=torch.float32)
    x_cat_tensor = None if x_cat is None else torch.tensor(x_cat, dtype=torch.long)
    y_tensor = None if y is None else torch.tensor(y, dtype=torch.float32)
    w_tensor = None if sample_weight is None else torch.tensor(sample_weight, dtype=torch.float32)

    dataset = TabularTensorDataset(
        x_num=x_num_tensor,
        x_cat=x_cat_tensor,
        y=y_tensor,
        sample_weight=w_tensor,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


# ---------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------

if __name__ == "__main__":
    # Minimal smoke test with synthetic data.
    rng = np.random.default_rng(42)
    n = 2048
    n_num = 5
    cat_cardinalities = [4, 6, 10]

    x_num = rng.normal(size=(n, n_num)).astype(np.float32)
    x_cat = np.column_stack(
        [rng.integers(0, c, size=n, endpoint=False) for c in cat_cardinalities]
    ).astype(np.int64)

    logit_signal = (
        0.8 * x_num[:, 0]
        - 0.5 * x_num[:, 1]
        + 0.7 * (x_cat[:, 0] == 1)
        + 0.9 * (x_cat[:, 2] == 3)
        - 2.0
    )
    prob = 1.0 / (1.0 + np.exp(-logit_signal))
    y = rng.binomial(1, prob).astype(np.float32)

    idx = np.arange(n)
    rng.shuffle(idx)
    train_idx = idx[:1600]
    valid_idx = idx[1600:]

    train_loader = make_dataloader(
        x_num=x_num[train_idx],
        x_cat=x_cat[train_idx],
        y=y[train_idx],
        batch_size=256,
        shuffle=True,
    )
    valid_loader = make_dataloader(
        x_num=x_num[valid_idx],
        x_cat=x_cat[valid_idx],
        y=y[valid_idx],
        batch_size=512,
        shuffle=False,
    )

    model = FTTransformer(
        n_num_features=n_num,
        cat_cardinalities=cat_cardinalities,
        d_token=64,
        n_blocks=3,
        n_heads=8,
        attention_dropout=0.1,
        ffn_dropout=0.1,
        residual_dropout=0.0,
        head_hidden_dim=64,
        head_dropout=0.1,
    )

    config = TrainingConfig(
        lr=1e-3,
        weight_decay=1e-5,
        batch_size=256,
        max_epochs=20,
        patience=5,
        verbose=True,
    )

    model, history = fit_ft_transformer(model, train_loader, valid_loader, config)
    valid_logits = predict_logits(model, valid_loader, device=config.device)
    valid_prob = 1.0 / (1.0 + np.exp(-valid_logits))
    print("Validation logits shape:", valid_logits.shape)
    print("Validation probability range:", float(valid_prob.min()), float(valid_prob.max()))
