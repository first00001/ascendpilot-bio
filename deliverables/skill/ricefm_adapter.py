"""Ascend-compatible riceFM embedding runtime for the public lcxlcx/riceFM code."""
from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path
from typing import Any, Iterable


class RiceFMUnavailable(RuntimeError):
    pass


class RiceFMRuntime:
    """Load a riceFM pretraining checkpoint and produce cell embeddings.

    The supplied checkpoint has no trained classification head. Cell-type labels
    must therefore be assigned by a separately fitted labeled reference model.
    """

    def __init__(self, model_dir: str, repo_dir: str, device: str = "npu:0", seed: int = 2026):
        self.model_dir = Path(model_dir)
        self.repo_dir = Path(repo_dir)
        self.device_name = device
        self.seed = seed
        required = ["args.json", "best_model.pt", "vocab.json"]
        missing = [name for name in required if not (self.model_dir / name).is_file()]
        if missing:
            raise RiceFMUnavailable(f"riceFM checkpoint incomplete; missing: {missing}")

        import numpy as np
        import torch

        if device.startswith("npu"):
            try:
                import torch_npu  # noqa: F401
            except ImportError as exc:
                raise RiceFMUnavailable("torch_npu is required for an NPU device") from exc

        package_name = "_ricefm_runtime_model"
        package = types.ModuleType(package_name)
        package.__path__ = [str(self.repo_dir / "ricefm" / "model")]
        sys.modules[package_name] = package
        importlib.import_module(f"{package_name}.ricefm_layers")
        TransformerModel = importlib.import_module(f"{package_name}.model").TransformerModel

        self.np = np
        self.torch = torch
        self.config = json.loads((self.model_dir / "args.json").read_text(encoding="utf-8"))
        self.vocab: dict[str, int] = json.loads(
            (self.model_dir / "vocab.json").read_text(encoding="utf-8")
        )
        if set(("<pad>", "<cls>", "<eoc>")) - self.vocab.keys():
            raise RiceFMUnavailable("vocab.json is missing required special tokens")
        if max(self.vocab.values()) + 1 != len(self.vocab):
            raise RiceFMUnavailable("vocab indices are not consecutive")

        self.pad_id = self.vocab["<pad>"]
        self.cls_id = self.vocab["<cls>"]
        self.pad_value = int(self.config.get("pad_value", -2))
        self.n_bins = int(self.config.get("n_bins", 51))
        self.max_seq_len = int(self.config.get("max_seq_len", 1200))
        self.model = TransformerModel(
            len(self.vocab),
            d_model=self.config["embsize"],
            nhead=self.config["nheads"],
            d_hid=self.config["d_hid"],
            nlayers=self.config["nlayers"],
            nlayers_cls=self.config.get("n_layers_cls", 3),
            n_cls=1,
            padding_idx=self.pad_id,
            dropout=self.config.get("dropout", 0.2),
            pad_token="<pad>",
            pad_value=self.pad_value,
            do_mvc=True,
            input_emb_style=self.config.get("input_emb_style", "continuous"),
            n_input_bins=self.n_bins,
            use_generative_training=self.config.get("training_tasks") in ("gen", "both"),
            use_fast_transformer=self.config.get("fast_transformer", True),
        )
        state = torch.load(self.model_dir / "best_model.pt", map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.device = torch.device(device)
        self.model.to(self.device).eval()

    def _bin_values(self, values: Any) -> Any:
        values = self.np.asarray(values, dtype=self.np.float32)
        if not len(values):
            return values
        bins = self.np.quantile(values, self.np.linspace(0, 1, self.n_bins - 1))
        return self.np.digitize(values, bins).astype(self.np.float32)

    def _prepare_batch(self, counts: Any, gene_token_ids: Any, batch_offset: int):
        torch = self.torch
        prepared: list[tuple[Any, Any]] = []
        for row_number, row in enumerate(counts):
            row = self.np.asarray(row).reshape(-1)
            expressed = self.np.flatnonzero((row > 0) & (gene_token_ids >= 0))
            values = self._bin_values(row[expressed])
            tokens = gene_token_ids[expressed]
            if len(tokens) >= self.max_seq_len:
                rng = self.np.random.default_rng(self.seed + batch_offset + row_number)
                chosen = rng.choice(len(tokens), self.max_seq_len - 1, replace=False)
                tokens, values = tokens[chosen], values[chosen]
            tokens = self.np.concatenate(([self.cls_id], tokens)).astype(self.np.int64)
            values = self.np.concatenate(([self.pad_value], values)).astype(self.np.float32)
            prepared.append((tokens, values))

        length = min(self.max_seq_len, max(len(tokens) for tokens, _ in prepared))
        genes = torch.full((len(prepared), length), self.pad_id, dtype=torch.long)
        expressions = torch.full((len(prepared), length), self.pad_value, dtype=torch.float32)
        for index, (tokens, values) in enumerate(prepared):
            genes[index, : len(tokens)] = torch.from_numpy(tokens)
            expressions[index, : len(values)] = torch.from_numpy(values)
        return genes, expressions

    def cell_embeddings(
        self,
        counts: Any,
        genes: Iterable[str],
        batch_size: int = 16,
        cell_offset: int = 0,
    ) -> Any:
        """Return finite 256-dimensional riceFM embeddings for cells x genes counts."""
        is_sparse = hasattr(counts, "toarray")
        if not is_sparse:
            counts = self.np.asarray(counts)
        if len(counts.shape) != 2:
            raise ValueError("counts must be a two-dimensional cells x genes matrix")
        genes = list(genes)
        if counts.shape[1] != len(genes):
            raise ValueError("the number of count columns must equal the number of genes")
        gene_token_ids = self.np.asarray([self.vocab.get(gene, -1) for gene in genes], dtype=int)
        if not (gene_token_ids >= 0).any():
            raise ValueError("none of the input genes occur in the riceFM vocabulary")

        outputs = []
        with self.torch.inference_mode():
            for start in range(0, counts.shape[0], batch_size):
                batch_counts = counts[start : start + batch_size]
                if is_sparse:
                    batch_counts = batch_counts.toarray()
                genes_pt, values_pt = self._prepare_batch(
                    batch_counts, gene_token_ids, cell_offset + start
                )
                genes_pt = genes_pt.to(self.device)
                values_pt = values_pt.to(self.device)
                result = self.model(
                    pcpt_genes=genes_pt,
                    pcpt_values=values_pt,
                    pcpt_key_padding_mask=genes_pt.eq(self.pad_id),
                    gen_genes=None,
                    gen_key_padding_mask=None,
                    mod_types=self.torch.zeros(
                        len(genes_pt), dtype=self.torch.long, device=self.device
                    ),
                    generative_training=True,
                )["cell_emb"]
                outputs.append(result.float().cpu().numpy())
        embeddings = self.np.concatenate(outputs)
        if not self.np.isfinite(embeddings).all():
            raise RuntimeError("riceFM produced non-finite cell embeddings")
        return embeddings

    def analyze(self, query: str, dataset: dict[str, Any]) -> dict[str, Any]:
        return {
            "status": "ricefm-ready",
            "query": query,
            "model_role": "cell_embedding",
            "embedding_size": int(self.config["embsize"]),
            "classification_head": False,
            "dataset": dataset,
        }
