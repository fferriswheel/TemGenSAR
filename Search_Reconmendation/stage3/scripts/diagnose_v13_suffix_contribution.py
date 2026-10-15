#!/usr/bin/env python3
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from build_neural_rq_sid_v2 import (
    aggregate_search_queries,
    load_item_rows,
    parse_sizes,
    rq_assign_codes_np,
)
from build_neural_rq_sid_v2_text_encoder_fusion_plc import (
    build_search_signal_embeddings,
    l2_normalize_np,
)
from build_neural_rq_sid_v3_aligned_delta import (
    load_behavior_signal,
    load_precomputed_embedding,
)
from build_neural_rq_sid_v9_direct_private import DirectPrivateTokenizer, encode_private
from train_multiview_anchor_prefix_v13 import MultiViewAnchorRQVAE


def reconstruction_metrics(prediction, target):
    prediction = np.asarray(prediction, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    cosine = np.sum(l2_normalize_np(prediction) * l2_normalize_np(target), axis=1)
    mse = np.mean((prediction - target) ** 2, axis=1)
    return {
        "cosine_mean": float(cosine.mean()),
        "cosine_std": float(cosine.std()),
        "mse_mean": float(mse.mean()),
    }


def reconstruct_codes(codes, codebooks):
    result = None
    for level, codebook in enumerate(codebooks):
        values = codebook.detach().cpu().numpy()[codes[:, level]]
        result = values.copy() if result is None else result + values
    return result.astype(np.float32)


@torch.no_grad()
def decode_all(decoder, suffix, prefix, base, device, batch_size):
    outputs = []
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(suffix),
            torch.from_numpy(prefix),
            torch.from_numpy(base),
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    decoder.eval()
    for suffix_b, prefix_b, base_b in loader:
        residual = decoder(suffix_b.to(device), prefix_b.to(device))
        outputs.append(F.normalize(base_b.to(device) + residual, dim=1).cpu().numpy())
    return np.concatenate(outputs).astype(np.float32)


@torch.no_grad()
def decode_residual_all(decoder, suffix, prefix, device, batch_size):
    outputs = []
    loader = DataLoader(
        TensorDataset(torch.from_numpy(suffix), torch.from_numpy(prefix)),
        batch_size=batch_size,
        shuffle=False,
    )
    decoder.eval()
    for suffix_b, prefix_b in loader:
        outputs.append(decoder(suffix_b.to(device), prefix_b.to(device)).cpu().numpy())
    return np.concatenate(outputs).astype(np.float32)


def conditional_suffix_stats(prefix_codes, suffix_codes):
    groups = defaultdict(list)
    suffix_counts = Counter()
    for prefix, suffix in zip(prefix_codes, suffix_codes):
        prefix_key = tuple(int(value) for value in prefix)
        suffix_key = tuple(int(value) for value in suffix)
        groups[prefix_key].append(suffix_key)
        suffix_counts[suffix_key] += 1

    n_items = len(prefix_codes)
    conditional_entropy = 0.0
    multi_item_count = 0
    resolved_multi_item_count = 0
    unique_per_group = []
    for values in groups.values():
        counts = Counter(values)
        probabilities = np.asarray(list(counts.values()), dtype=np.float64) / len(values)
        conditional_entropy += len(values) / n_items * float(
            -(probabilities * np.log(probabilities)).sum()
        )
        unique_per_group.append(len(counts))
        if len(values) > 1:
            multi_item_count += len(values)
            if len(counts) > 1:
                resolved_multi_item_count += len(values)

    probabilities = np.asarray(list(suffix_counts.values()), dtype=np.float64) / n_items
    suffix_entropy = float(-(probabilities * np.log(probabilities)).sum())
    return {
        "conditional_entropy_nats": conditional_entropy,
        "suffix_entropy_nats": suffix_entropy,
        "conditional_entropy_ratio": conditional_entropy / max(suffix_entropy, 1e-12),
        "mean_unique_suffixes_per_prefix": float(np.mean(unique_per_group)),
        "max_unique_suffixes_per_prefix": int(max(unique_per_group)),
        "multi_item_prefix_item_fraction": multi_item_count / n_items,
        "multi_item_prefix_resolved_fraction": resolved_multi_item_count
        / max(multi_item_count, 1),
    }


def within_prefix_permutation(prefix_codes, seed):
    groups = defaultdict(list)
    for index, prefix in enumerate(prefix_codes):
        groups[tuple(int(value) for value in prefix)].append(index)
    rng = np.random.default_rng(seed)
    permutation = np.arange(len(prefix_codes))
    for indices in groups.values():
        if len(indices) > 1:
            permutation[indices] = rng.permutation(indices)
    return permutation


def subset_metrics(prediction, target, mask):
    return reconstruction_metrics(prediction[mask], target[mask])


def diagnose(args):
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    params = argparse.Namespace(**checkpoint["params"])
    device = torch.device(args.device)

    item_ids, _, _ = load_item_rows(
        Path(params.data_dir), params.item_ids, params.max_items
    )
    _, query_counter = aggregate_search_queries(
        Path(params.data_dir), item_ids, params.max_queries_per_item
    )
    common_x = load_precomputed_embedding(params.common_embedding, item_ids, "common")
    behavior_x, _ = load_behavior_signal(params.rec_behavior_embedding, item_ids)
    search_x, _ = build_search_signal_embeddings(
        params,
        item_ids,
        query_counter,
        device,
        Path(params.embedding_cache_dir),
    )

    anchor_checkpoint = torch.load(
        params.anchor_checkpoint, map_location="cpu", weights_only=False
    )
    anchor_args = argparse.Namespace(**anchor_checkpoint["params"])
    anchor = MultiViewAnchorRQVAE(
        common_x.shape[1],
        behavior_x.shape[1],
        search_x.shape[1],
        parse_sizes(params.prefix_codebook_sizes),
        anchor_args,
    )
    model = DirectPrivateTokenizer(
        anchor,
        common_x.shape[1],
        search_x.shape[1],
        behavior_x.shape[1],
        checkpoint["search_sizes"],
        checkpoint["rec_sizes"],
        params,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    lookup = np.load(params.anchor_lookup)
    if [str(value) for value in lookup["item_ids"].tolist()] != item_ids:
        raise ValueError("anchor lookup item order does not match dataset")
    prefix_codes = lookup["prefix_codes"].astype(np.int64)
    z_shared = lookup["z_anchor"].astype(np.float32)
    q_shared = lookup["q_anchor"].astype(np.float32)
    semantic_base = l2_normalize_np(lookup["common_base"].astype(np.float32))
    behavior_base = l2_normalize_np(lookup["behavior_base"].astype(np.float32))
    search_target = l2_normalize_np(common_x + params.search_target_scale * search_x)

    search_private, rec_private, relevance, identity = encode_private(
        model,
        common_x,
        search_x,
        behavior_x,
        device,
        args.batch_size,
        return_parts=True,
        fixed_z_shared=z_shared,
        fixed_q_shared=q_shared,
        fixed_semantic_base=semantic_base,
    )
    search_codes = np.column_stack(
        [
            np.argmin(
                ((relevance[:, None, :] - model.search_relevance_quantizer.codebooks[0].detach().cpu().numpy()[None]) ** 2).sum(2),
                axis=1,
            ),
            np.argmin(
                ((identity[:, None, :] - model.search_identity_quantizer.codebooks[0].detach().cpu().numpy()[None]) ** 2).sum(2),
                axis=1,
            ),
        ]
    ).astype(np.int64)
    rec_codes = np.asarray(
        rq_assign_codes_np(rec_private, model.rec_quantizer.codebooks, "argmin"),
        dtype=np.int64,
    )
    q_search = np.concatenate(
        [
            reconstruct_codes(
                search_codes[:, :1], model.search_relevance_quantizer.codebooks
            ),
            reconstruct_codes(
                search_codes[:, 1:], model.search_identity_quantizer.codebooks
            ),
        ],
        axis=1,
    ).astype(np.float32)
    q_rec = reconstruct_codes(rec_codes, model.rec_quantizer.codebooks)

    rng = np.random.default_rng(args.seed)
    global_permutation = rng.permutation(len(item_ids))
    group_permutation = within_prefix_permutation(prefix_codes, args.seed + 1)
    multi_prefix_mask = np.zeros(len(item_ids), dtype=bool)
    prefix_count = Counter(tuple(row) for row in prefix_codes.tolist())
    for index, prefix in enumerate(prefix_codes.tolist()):
        multi_prefix_mask[index] = prefix_count[tuple(prefix)] > 1

    result = {
        "checkpoint": str(checkpoint_path),
        "anchor_mode": anchor_args.anchor_mode,
        "num_items": len(item_ids),
        "multi_item_prefix_items": int(multi_prefix_mask.sum()),
        "search": {},
        "rec": {},
    }
    task_inputs = {
        "search": (model.search_decoder, q_search, search_private, semantic_base, search_target),
        "rec": (model.rec_decoder, q_rec, rec_private, behavior_base, behavior_x),
    }
    for task, (decoder, quantized, continuous, base, target) in task_inputs.items():
        full = decode_all(decoder, quantized, q_shared, base, device, args.batch_size)
        continuous_full = decode_all(
            decoder, continuous, q_shared, base, device, args.batch_size
        )
        zero_full = decode_all(
            decoder,
            np.zeros_like(quantized),
            q_shared,
            base,
            device,
            args.batch_size,
        )
        global_shuffle = decode_all(
            decoder,
            quantized[global_permutation],
            q_shared,
            base,
            device,
            args.batch_size,
        )
        group_shuffle = decode_all(
            decoder,
            quantized[group_permutation],
            q_shared,
            base,
            device,
            args.batch_size,
        )
        residual = decode_residual_all(
            decoder, quantized, q_shared, device, args.batch_size
        )
        task_codes = search_codes if task == "search" else rec_codes
        full_metrics = reconstruction_metrics(full, target)
        base_metrics = reconstruction_metrics(base, target)
        result[task] = {
            "prefix_only": base_metrics,
            "zero_suffix_decoder": reconstruction_metrics(zero_full, target),
            "continuous_suffix": reconstruction_metrics(continuous_full, target),
            "quantized_suffix": full_metrics,
            "global_shuffled_suffix": reconstruction_metrics(global_shuffle, target),
            "within_prefix_shuffled_suffix": reconstruction_metrics(group_shuffle, target),
            "within_prefix_shuffled_multi_item_only": subset_metrics(
                group_shuffle, target, multi_prefix_mask
            ),
            "quantized_multi_item_only": subset_metrics(full, target, multi_prefix_mask),
            "cosine_gain_over_prefix": full_metrics["cosine_mean"]
            - base_metrics["cosine_mean"],
            "cosine_drop_global_shuffle": full_metrics["cosine_mean"]
            - reconstruction_metrics(global_shuffle, target)["cosine_mean"],
            "cosine_drop_within_prefix_shuffle_multi_item": subset_metrics(
                full, target, multi_prefix_mask
            )["cosine_mean"]
            - subset_metrics(group_shuffle, target, multi_prefix_mask)["cosine_mean"],
            "decoded_residual_norm_mean": float(np.linalg.norm(residual, axis=1).mean()),
            "conditional_suffix": conditional_suffix_stats(prefix_codes, task_codes),
        }

    output_path = Path(args.output or checkpoint_path.parent / "suffix_contribution.json")
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


def parser():
    value = argparse.ArgumentParser()
    value.add_argument("--checkpoint", required=True)
    value.add_argument("--output", default="")
    value.add_argument("--device", default="cuda:0")
    value.add_argument("--batch-size", type=int, default=1024)
    value.add_argument("--seed", type=int, default=2026)
    return value


if __name__ == "__main__":
    diagnose(parser().parse_args())
