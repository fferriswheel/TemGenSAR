#!/usr/bin/env python3
"""Update only the v13b Rec low-rank adapter with a fixed prefix and codebooks."""

import argparse
import copy
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from build_neural_rq_sid_v2 import code_usage, load_item_rows, rq_assign_codes_np, write_json
from build_neural_rq_sid_v3_aligned_delta import (
    PrefixBatchSampler,
    conditional_collision_stats,
    inprefix_topology_loss,
    load_behavior_signal,
    load_precomputed_embedding,
)
from build_neural_rq_sid_v7_anchored_cross_view import mapping_from_codes
from build_neural_rq_sid_v9_direct_private import DirectPrivateTokenizer, cosine_loss
from build_neural_rq_sid_v2_text_encoder_fusion_plc import l2_normalize_np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--t0-mapping", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--common-embedding", required=True)
    p.add_argument("--updated-rec-embedding", required=True)
    p.add_argument("--t0-rec-embedding", required=True)
    p.add_argument("--anchor-lookup", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--topology-weight", type=float, default=0.5)
    p.add_argument("--commit-weight", type=float, default=0.05)
    p.add_argument("--stability-weight", type=float, default=0.1)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--device", default="cuda:4")
    p.add_argument("--seed", type=int, default=2026)
    return p.parse_args()


def mapping_arrays(mapping, item_ids):
    search, rec = [], []
    for item_id in item_ids:
        row = mapping[item_id]
        search_codes = row.get("search_suffix_codes")
        rec_codes = row.get("rec_suffix_codes")
        if search_codes is None or rec_codes is None:
            raise ValueError(f"mapping lacks numeric suffix codes for {item_id}")
        if len(search_codes) != 2 or len(rec_codes) != 2:
            raise ValueError(f"expected two-level suffixes for {item_id}")
        search.append(search_codes)
        rec.append(rec_codes)
    return np.asarray(search, dtype=np.int64), np.asarray(rec, dtype=np.int64)


@torch.no_grad()
def assign_rec(model, z_shared, behavior_x, batch_size, device):
    model.eval()
    private = []
    for start in range(0, len(behavior_x), batch_size):
        end = min(start + batch_size, len(behavior_x))
        private.append(
            model.rec_adapter(
                torch.from_numpy(z_shared[start:end]).to(device),
                torch.from_numpy(behavior_x[start:end]).to(device),
            ).cpu().numpy()
        )
    private = np.concatenate(private).astype(np.float32)
    codes = np.asarray(
        rq_assign_codes_np(private, model.rec_quantizer.codebooks, "argmin"), dtype=np.int64
    )
    return private, codes


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    params = argparse.Namespace(**checkpoint["params"])
    item_ids, _, _ = load_item_rows(args.data_dir, "", params.max_items)
    common_x = load_precomputed_embedding(args.common_embedding, item_ids, "common")
    behavior_x, behavior_stats = load_behavior_signal(args.updated_rec_embedding, item_ids)
    t0_behavior_x, _ = load_behavior_signal(args.t0_rec_embedding, item_ids)
    lookup = np.load(args.anchor_lookup)
    if [str(value) for value in lookup["item_ids"].tolist()] != item_ids:
        raise ValueError("anchor lookup item order mismatch")
    prefix_codes = lookup["prefix_codes"].astype(np.int64)
    z_shared = lookup["z_anchor"].astype(np.float32)
    q_shared = lookup["q_anchor"].astype(np.float32)
    behavior_base = l2_normalize_np(lookup["behavior_base"].astype(np.float32)).astype(np.float32)

    # The shared module is present in the checkpoint state and only supplies frozen parameters.
    from train_multiview_anchor_prefix_v13 import MultiViewAnchorRQVAE
    anchor_ckpt = torch.load(params.anchor_checkpoint, map_location="cpu", weights_only=False)
    anchor_args = argparse.Namespace(**anchor_ckpt["params"])
    shared = MultiViewAnchorRQVAE(
        common_x.shape[1], behavior_x.shape[1], 768,
        checkpoint["prefix_sizes"], anchor_args,
    )
    model = DirectPrivateTokenizer(
        shared, common_x.shape[1], 768, behavior_x.shape[1],
        checkpoint["search_sizes"], checkpoint["rec_sizes"], params,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    initial_state = copy.deepcopy(model.state_dict())
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.rec_adapter.parameters():
        parameter.requires_grad = True
    model.train()
    model.shared.eval()
    model.search_adapter.eval()
    model.rec_quantizer.eval()
    model.rec_decoder.eval()

    t0_mapping = json.loads(args.t0_mapping.read_text(encoding="utf-8"))
    t0_search, t0_rec = mapping_arrays(t0_mapping, item_ids)
    with torch.no_grad():
        old_private, old_rec = assign_rec(model, z_shared, t0_behavior_x, args.batch_size, device)
        _, pre_update_rec = assign_rec(model, z_shared, behavior_x, args.batch_size, device)

    dataset = TensorDataset(
        torch.from_numpy(behavior_x), torch.from_numpy(behavior_base),
        torch.from_numpy(prefix_codes), torch.from_numpy(z_shared), torch.from_numpy(q_shared),
        torch.from_numpy(old_private),
    )
    loader = DataLoader(
        dataset,
        batch_sampler=PrefixBatchSampler(prefix_codes, args.batch_size, 16, args.seed + 17),
        num_workers=0,
    )
    trainable = list(model.rec_adapter.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    best_state, best_score, best_epoch, stale = None, float("inf"), None, 0
    logs = []
    start_time = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        totals, steps = Counter(), 0
        for behavior_b, base_b, prefix_b, z_b, q_b, old_private_b in loader:
            behavior_b, base_b = behavior_b.to(device), base_b.to(device)
            prefix_b, z_b, q_b = prefix_b.to(device), z_b.to(device), q_b.to(device)
            old_private_b = old_private_b.to(device)
            private = model.rec_adapter(z_b, behavior_b)
            q_rec, _, commit, _ = model.rec_quantizer(private)
            recon = model.rec_decoder(q_rec, q_b)
            full_recon = F.normalize(base_b + recon, dim=1)
            reconstruction = cosine_loss(full_recon, behavior_b)
            topology = inprefix_topology_loss(prefix_b, q_rec, behavior_b)
            stability = F.mse_loss(private, old_private_b)
            loss = reconstruction + args.topology_weight * topology
            loss = loss + args.commit_weight * commit + args.stability_weight * stability
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            for key, value in {"loss": loss, "reconstruction": reconstruction,
                               "topology": topology, "commit": commit,
                               "stability": stability}.items():
                totals[key] += float(value.detach().cpu())
            steps += 1
        private_np, rec_codes = assign_rec(model, z_shared, behavior_x, args.batch_size, device)
        collision = conditional_collision_stats(prefix_codes, rec_codes)
        changed = float(np.any(rec_codes != t0_rec, axis=1).mean())
        row = {"epoch": epoch, **{k: v / max(steps, 1) for k, v in totals.items()},
               "rec_duplicate": collision["duplicate_fraction"],
               "rec_changed_from_t0_raw": changed}
        score = (
            row["reconstruction"]
            + 0.5 * max(row["rec_duplicate"] - 0.09, 0.0)
            + max(changed - 0.15, 0.0)
        )
        row["selection_score"] = score
        logs.append(row)
        if epoch == 1 or epoch % args.log_every == 0:
            print(json.dumps(row), flush=True)
        if score < best_score - 1e-5:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.load_state_dict(best_state)
    private_np, rec_codes = assign_rec(model, z_shared, behavior_x, args.batch_size, device)

    final_state = model.state_dict()
    changed_tensors = [k for k in final_state if not torch.equal(final_state[k].cpu(), initial_state[k].cpu())]
    illegal = [k for k in changed_tensors if not k.startswith("rec_adapter.")]
    if illegal:
        raise RuntimeError(f"non-Rec-adapter parameters changed: {illegal}")
    if not np.array_equal(checkpoint["fixed_prefix_codes"], prefix_codes):
        raise RuntimeError("prefix codes changed")

    mapping = mapping_from_codes(item_ids, prefix_codes, t0_search, rec_codes)
    write_json(args.output_dir / "sid_mapping.rec_adapter_update.json", mapping)
    summary = {
        "method": "v13b Rec adapter-only online update",
        "items": len(item_ids), "best_epoch": best_epoch, "best_score": best_score,
        "wall_seconds": time.perf_counter() - start_time,
        "trainable_parameters": sum(p.numel() for p in trainable),
        "changed_parameter_tensors": changed_tensors,
        "prefix_exactly_preserved": True, "search_suffix_exactly_preserved": True,
        "rec_changed_from_t0_raw": float(np.any(rec_codes != t0_rec, axis=1).mean()),
        "rec_changed_before_adapter_update": float(
            np.any(pre_update_rec != t0_rec, axis=1).mean()
        ),
        "rec_collision": conditional_collision_stats(prefix_codes, rec_codes),
        "rec_usage": code_usage(rec_codes, checkpoint["rec_sizes"]),
        "updated_behavior": behavior_stats, "history": logs,
    }
    write_json(args.output_dir / "update_stats.json", summary)
    torch.save({"model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "base_checkpoint": str(args.checkpoint), "summary": summary},
               args.output_dir / "rec_adapter_best.pt")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
