#!/usr/bin/env python3
import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader, TensorDataset


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def parse_sizes(value):
    sizes = [int(part) for part in value.split(",") if part.strip()]
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError(f"invalid codebook sizes: {value}")
    return sizes


def flatten_categories(categories):
    if not categories:
        return ""
    terms = []
    for path in categories:
        if isinstance(path, list):
            terms.extend(str(part) for part in path if part)
        elif path:
            terms.append(str(path))
    return " ".join(terms)


def load_id_file(path):
    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            ids = []
            for row in obj:
                if isinstance(row, str):
                    ids.append(row)
                elif isinstance(row, dict) and "item_id" in row:
                    ids.append(row["item_id"])
            return ids
        if isinstance(obj, dict) and "item_ids" in obj:
            return [str(item_id) for item_id in obj["item_ids"]]
    except json.JSONDecodeError:
        pass

    ids = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            if isinstance(row, dict) and "item_id" in row:
                ids.append(row["item_id"])
            elif isinstance(row, str):
                ids.append(row)
        except json.JSONDecodeError:
            ids.append(line)
    return ids


def load_item_rows(data_dir, item_ids_arg, max_items):
    data_dir = Path(data_dir)
    item_ids_path = Path(item_ids_arg) if item_ids_arg else data_dir / "item_ids.jsonl"
    selected_ids = load_id_file(item_ids_path)
    selected_set = set(selected_ids)

    rows_by_id = {}
    items_path = data_dir / "items.jsonl"
    if items_path.exists():
        for row in read_jsonl(items_path):
            item_id = row["item_id"]
            if item_id in selected_set:
                rows_by_id[item_id] = row
    else:
        for item_id in selected_ids:
            rows_by_id[item_id] = {
                "item_id": item_id,
                "title": "",
                "description": "",
                "categories": [],
                "brand": "",
                "item_text": item_id,
            }

    item_ids = [item_id for item_id in selected_ids if item_id in rows_by_id]
    if max_items > 0:
        item_ids = item_ids[:max_items]
    item_rows = {item_id: rows_by_id[item_id] for item_id in item_ids}
    return item_ids, item_rows, items_path.exists()


def field_coverage(item_rows):
    total = len(item_rows)
    coverage = {}
    for field in ["title", "description", "brand", "categories", "item_text"]:
        count = 0
        for row in item_rows.values():
            value = row.get(field)
            if isinstance(value, str):
                count += bool(value.strip())
            else:
                count += bool(value)
        coverage[field] = {"count": count, "ratio": count / max(total, 1)}
    return coverage


def aggregate_search_queries(data_dir, item_ids, max_queries_per_item):
    item_set = set(item_ids)
    query_counter = defaultdict(Counter)
    for row in read_jsonl(Path(data_dir) / "search_interactions.jsonl"):
        if row.get("split") != "train":
            continue
        item_id = row["item_id"]
        if item_id in item_set:
            query_counter[item_id][row["query"]] += 1

    query_docs = {}
    for item_id in item_ids:
        queries = [query for query, _ in query_counter[item_id].most_common(max_queries_per_item)]
        query_docs[item_id] = " ".join(queries)
    return query_docs, query_counter


def update_neighbor_counts(user_rows, item_set, neighbor_window, counters):
    items = [row["item_id"] for row in user_rows]
    for pos, item_id in enumerate(items):
        if item_id not in item_set:
            continue
        left = max(0, pos - neighbor_window)
        right = min(len(items), pos + neighbor_window + 1)
        for neighbor_pos in range(left, right):
            if neighbor_pos == pos:
                continue
            counters[item_id][items[neighbor_pos]] += 1


def aggregate_rec_neighbors(data_dir, item_ids, neighbor_window, max_neighbors_per_item):
    item_set = set(item_ids)
    neighbor_counter = defaultdict(Counter)
    cur_user = None
    cur_rows = []
    for row in read_jsonl(Path(data_dir) / "rec_interactions_chrono.jsonl"):
        user_id = row["user_id"]
        if cur_user is None:
            cur_user = user_id
        if user_id != cur_user:
            update_neighbor_counts(cur_rows, item_set, neighbor_window, neighbor_counter)
            cur_user = user_id
            cur_rows = []
        cur_rows.append(row)
    if cur_rows:
        update_neighbor_counts(cur_rows, item_set, neighbor_window, neighbor_counter)

    rec_docs = {}
    for item_id in item_ids:
        parts = []
        for neighbor, count in neighbor_counter[item_id].most_common(max_neighbors_per_item):
            parts.extend([neighbor] * min(count, 5))
        rec_docs[item_id] = " ".join(parts)
    return rec_docs, neighbor_counter


def make_docs(item_ids, item_rows, query_docs, rec_docs_by_id):
    text_docs = []
    search_docs = []
    rec_docs = []
    for item_id in item_ids:
        row = item_rows[item_id]
        title = row.get("title", "")
        description = row.get("description", "")
        brand = row.get("brand", "")
        categories = flatten_categories(row.get("categories", []))
        item_text = row.get("item_text", "")
        queries = query_docs.get(item_id, "")
        neighbors = rec_docs_by_id.get(item_id, "")

        text = " ".join(part for part in [title, categories, brand, description, item_text, item_id] if part)
        search = " ".join(part for part in [title, categories, brand, queries, item_id] if part)
        rec = " ".join(part for part in [title, categories, brand, neighbors, item_id] if part)
        text_docs.append(text if text else item_id)
        search_docs.append(search if search else item_id)
        rec_docs.append(rec if rec else item_id)
    return text_docs, search_docs, rec_docs


def _pad_dim(x, dim):
    if x.shape[1] == dim:
        return x.astype(np.float32)
    out = np.zeros((x.shape[0], dim), dtype=np.float32)
    out[:, : min(dim, x.shape[1])] = x[:, :dim]
    return out


def vectorize_reduce_text(docs, max_features, out_dim, seed):
    vectorizer = TfidfVectorizer(min_df=1, max_features=max_features, norm="l2")
    x = vectorizer.fit_transform(docs)
    dim = min(out_dim, max(1, min(x.shape) - 1))
    if dim < 2:
        dense = x.toarray().astype(np.float32)
    else:
        dense = TruncatedSVD(n_components=dim, random_state=seed).fit_transform(x).astype(np.float32)
    return _pad_dim(normalize(dense, norm="l2"), out_dim)


def vectorize_reduce_hash(docs, n_features, out_dim, seed):
    vectorizer = HashingVectorizer(n_features=n_features, alternate_sign=False, norm="l2")
    x = vectorizer.transform(docs)
    dim = min(out_dim, max(1, min(x.shape) - 1))
    if dim < 2:
        dense = x.toarray().astype(np.float32)
    else:
        dense = TruncatedSVD(n_components=dim, random_state=seed).fit_transform(x).astype(np.float32)
    return _pad_dim(normalize(dense, norm="l2"), out_dim)


def count_summary(values):
    values = sorted(values)
    if not values:
        return {"count": 0}

    def pct(q):
        return values[int(round((len(values) - 1) * q))]

    return {
        "count": len(values),
        "min": values[0],
        "p50": pct(0.50),
        "p90": pct(0.90),
        "p95": pct(0.95),
        "p99": pct(0.99),
        "max": values[-1],
        "mean": sum(values) / len(values),
    }


def collision_rate(sid_by_item):
    reverse = defaultdict(list)
    for item_id, sid in sid_by_item.items():
        reverse[tuple(sid)].append(item_id)
    collided_items = sum(len(items) for items in reverse.values() if len(items) > 1)
    return {
        "unique_sid_count": len(reverse),
        "collided_item_count": collided_items,
        "collision_rate": collided_items / max(len(sid_by_item), 1),
        "max_bucket_size": max((len(items) for items in reverse.values()), default=0),
    }


def code_usage(level_codes, codebook_sizes):
    usage = []
    for level, size in enumerate(codebook_sizes):
        counts = Counter(int(codes[level]) for codes in level_codes)
        probs = np.array([counts.get(idx, 0) for idx in range(size)], dtype=np.float64)
        probs = probs / max(probs.sum(), 1.0)
        entropy = float(-(probs[probs > 0] * np.log(probs[probs > 0])).sum() / max(math.log(size), 1e-12))
        usage.append(
            {
                "level": level + 1,
                "codebook_size": size,
                "used_codes": len(counts),
                "entropy_norm": entropy,
                "count": dict(sorted((str(k), int(v)) for k, v in counts.items())),
            }
        )
    return usage


def squared_distances_np(x, centers):
    x_norm = np.sum(x * x, axis=1, keepdims=True)
    c_norm = np.sum(centers * centers, axis=1, keepdims=True).T
    return x_norm + c_norm - 2 * x @ centers.T


def balanced_assign_np(distances):
    n_items, n_clusters = distances.shape
    base_capacity = n_items // n_clusters
    extra = n_items % n_clusters
    capacities = np.full(n_clusters, base_capacity, dtype=np.int64)
    capacities[:extra] += 1

    order = np.argsort(distances, axis=1)
    assigned = np.full(n_items, -1, dtype=np.int64)
    remaining = capacities.copy()
    item_order = np.argsort(distances[np.arange(n_items), order[:, 0]])
    for item_idx in item_order:
        for cluster_idx in order[item_idx]:
            if remaining[cluster_idx] > 0:
                assigned[item_idx] = cluster_idx
                remaining[cluster_idx] -= 1
                break

    unassigned = np.where(assigned < 0)[0]
    if len(unassigned):
        open_clusters = np.repeat(np.arange(n_clusters), remaining)
        for item_idx, cluster_idx in zip(unassigned, open_clusters):
            assigned[item_idx] = cluster_idx
    return assigned


def rq_assign_codes_np(values, codebooks, assignment):
    residual = values.astype(np.float32)
    all_codes = []
    for codebook in codebooks:
        centers = codebook.detach().cpu().numpy().astype(np.float32)
        distances = squared_distances_np(residual, centers)
        if assignment == "balanced":
            codes = balanced_assign_np(distances)
        else:
            codes = distances.argmin(axis=1)
        all_codes.append(codes.astype(np.int64))
        residual = residual - centers[codes]
    return np.stack(all_codes, axis=1).tolist()


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class ResidualVectorQuantizer(nn.Module):
    def __init__(self, codebook_sizes, dim, beta, balance_temperature, sk_epsilons=None, sk_iters=100):
        super().__init__()
        self.codebook_sizes = list(codebook_sizes)
        self.beta = beta
        self.balance_temperature = balance_temperature
        if sk_epsilons is None:
            sk_epsilons = [0.0] * len(self.codebook_sizes)
        if len(sk_epsilons) != len(self.codebook_sizes):
            raise ValueError(f"sk_epsilons length {len(sk_epsilons)} != codebook sizes length {len(self.codebook_sizes)}")
        self.sk_epsilons = [float(epsilon) for epsilon in sk_epsilons]
        self.sk_iters = int(sk_iters)
        self.codebooks = nn.ParameterList(
            [nn.Parameter(torch.randn(size, dim) * 0.02) for size in self.codebook_sizes]
        )

    @staticmethod
    def _center_distances(dist):
        max_distance = dist.max()
        min_distance = dist.min()
        middle = (max_distance + min_distance) / 2
        amplitude = max_distance - middle + 1e-5
        return (dist - middle) / amplitude

    @staticmethod
    def _sinkhorn_assign(dist, epsilon, iters):
        q = torch.exp(-dist.double() / epsilon)
        batch_size = q.shape[0]
        n_codes = q.shape[1]
        q = q / q.sum().clamp(min=1e-300)
        for _ in range(iters):
            q = q / q.sum(dim=1, keepdim=True).clamp(min=1e-300) / batch_size
            q = q / q.sum(dim=0, keepdim=True).clamp(min=1e-300) / n_codes
        q = q * batch_size
        if torch.isnan(q).any() or torch.isinf(q).any():
            return dist.argmin(dim=1)
        return torch.argmax(q, dim=1).to(dist.device)

    def forward(self, z):
        residual = z
        quantized = torch.zeros_like(z)
        indices = []
        commit_loss = z.new_tensor(0.0)
        balance_loss = z.new_tensor(0.0)
        for codebook, sk_epsilon in zip(self.codebooks, self.sk_epsilons):
            dist = (
                residual.pow(2).sum(dim=1, keepdim=True)
                - 2 * residual @ codebook.t()
                + codebook.pow(2).sum(dim=1).unsqueeze(0)
            )
            soft_probs = torch.softmax(-dist / self.balance_temperature, dim=1).mean(dim=0)
            entropy = -(soft_probs * (soft_probs + 1e-8).log()).sum() / math.log(codebook.shape[0])
            balance_loss = balance_loss + (1.0 - entropy)
            if sk_epsilon > 0:
                idx = self._sinkhorn_assign(self._center_distances(dist), sk_epsilon, self.sk_iters)
            else:
                idx = dist.argmin(dim=1)
            q = F.embedding(idx, codebook)
            quantized = quantized + q
            commit_loss = commit_loss + F.mse_loss(q, residual.detach()) + self.beta * F.mse_loss(
                residual, q.detach()
            )
            residual = residual - q.detach()
            indices.append(idx)
        quantized_st = z + (quantized - z).detach()
        return quantized_st, indices, commit_loss, balance_loss


class NeuralRQSidModel(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        latent_dim,
        prefix_sizes,
        search_sizes,
        rec_sizes,
        beta,
        dropout,
        balance_temperature,
    ):
        super().__init__()
        self.search_encoder = MLP(input_dim, hidden_dim, latent_dim, dropout)
        self.rec_encoder = MLP(input_dim, hidden_dim, latent_dim, dropout)
        self.gate = nn.Sequential(
            nn.Linear(latent_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.Sigmoid(),
        )
        self.search_residual = MLP(latent_dim * 3, hidden_dim, latent_dim, dropout)
        self.rec_residual = MLP(latent_dim * 3, hidden_dim, latent_dim, dropout)
        self.shared_quantizer = ResidualVectorQuantizer(prefix_sizes, latent_dim, beta, balance_temperature)
        self.search_quantizer = ResidualVectorQuantizer(search_sizes, latent_dim, beta, balance_temperature)
        self.rec_quantizer = ResidualVectorQuantizer(rec_sizes, latent_dim, beta, balance_temperature)
        self.search_decoder = MLP(latent_dim, hidden_dim, input_dim, dropout)
        self.rec_decoder = MLP(latent_dim, hidden_dim, input_dim, dropout)

    def forward(self, x_search, x_rec):
        z_search = self.search_encoder(x_search)
        z_rec = self.rec_encoder(x_rec)
        gate_input = torch.cat([z_search, z_rec, z_search * z_rec, (z_search - z_rec).abs()], dim=1)
        gate = self.gate(gate_input)
        z_shared = gate * z_search + (1.0 - gate) * z_rec
        r_search = self.search_residual(torch.cat([z_search, z_shared, z_search - z_shared], dim=1))
        r_rec = self.rec_residual(torch.cat([z_rec, z_shared, z_rec - z_shared], dim=1))

        q_shared, prefix_idx, prefix_commit, prefix_balance = self.shared_quantizer(z_shared)
        q_search, search_idx, search_commit, search_balance = self.search_quantizer(r_search)
        q_rec, rec_idx, rec_commit, rec_balance = self.rec_quantizer(r_rec)
        search_recon = self.search_decoder(q_shared + q_search)
        rec_recon = self.rec_decoder(q_shared + q_rec)
        return {
            "z_search": z_search,
            "z_rec": z_rec,
            "z_shared": z_shared,
            "r_search": r_search,
            "r_rec": r_rec,
            "search_recon": search_recon,
            "rec_recon": rec_recon,
            "prefix_idx": prefix_idx,
            "search_idx": search_idx,
            "rec_idx": rec_idx,
            "commit_loss": prefix_commit + search_commit + rec_commit,
            "balance_loss": prefix_balance + search_balance + rec_balance,
        }


def _kmeans_centers(x, size, seed):
    if len(x) == 0:
        raise ValueError("cannot initialize codebook from empty array")
    if len(x) < size:
        repeats = int(math.ceil(size / len(x)))
        tiled = np.tile(x, (repeats, 1))[:size]
        return tiled.astype(np.float32)
    model = MiniBatchKMeans(
        n_clusters=size,
        random_state=seed,
        batch_size=min(4096, max(256, len(x))),
        n_init="auto",
        max_iter=100,
    )
    return model.fit(x).cluster_centers_.astype(np.float32)


def init_rq_codebooks(quantizer, values, seed):
    residual = values.astype(np.float32)
    for level, codebook in enumerate(quantizer.codebooks):
        centers = _kmeans_centers(residual, codebook.shape[0], seed + level)
        codebook.data.copy_(torch.from_numpy(centers).to(codebook.device))
        distances = ((residual[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        assigned = centers[distances.argmin(axis=1)]
        residual = residual - assigned


@torch.no_grad()
def initialize_codebooks(model, x_search, x_rec, device, seed):
    model.eval()
    xs = torch.from_numpy(x_search).to(device)
    xr = torch.from_numpy(x_rec).to(device)
    z_search = model.search_encoder(xs)
    z_rec = model.rec_encoder(xr)
    gate_input = torch.cat([z_search, z_rec, z_search * z_rec, (z_search - z_rec).abs()], dim=1)
    gate = model.gate(gate_input)
    z_shared = gate * z_search + (1.0 - gate) * z_rec
    r_search = model.search_residual(torch.cat([z_search, z_shared, z_search - z_shared], dim=1))
    r_rec = model.rec_residual(torch.cat([z_rec, z_shared, z_rec - z_shared], dim=1))
    init_rq_codebooks(model.shared_quantizer, z_shared.cpu().numpy(), seed)
    init_rq_codebooks(model.search_quantizer, r_search.cpu().numpy(), seed + 1000)
    init_rq_codebooks(model.rec_quantizer, r_rec.cpu().numpy(), seed + 2000)


def train_model(args, x_search, x_rec, prefix_sizes, search_sizes, rec_sizes):
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    dataset = TensorDataset(torch.from_numpy(x_search), torch.from_numpy(x_rec))
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=generator)
    model = NeuralRQSidModel(
        input_dim=x_search.shape[1],
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        prefix_sizes=prefix_sizes,
        search_sizes=search_sizes,
        rec_sizes=rec_sizes,
        beta=args.vq_beta,
        dropout=args.dropout,
        balance_temperature=args.balance_temperature,
    ).to(device)
    if args.init_codebooks == "kmeans":
        initialize_codebooks(model, x_search, x_rec, device, args.seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    logs = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = Counter()
        num_batches = 0
        for batch_search, batch_rec in loader:
            batch_search = batch_search.to(device)
            batch_rec = batch_rec.to(device)
            out = model(batch_search, batch_rec)
            recon_loss = F.mse_loss(out["search_recon"], batch_search) + F.mse_loss(out["rec_recon"], batch_rec)
            align_loss = 1.0 - F.cosine_similarity(out["z_search"], out["z_rec"], dim=1).mean()
            disentangle_loss = F.cosine_similarity(out["z_shared"], out["r_search"], dim=1).abs().mean()
            disentangle_loss = disentangle_loss + F.cosine_similarity(out["z_shared"], out["r_rec"], dim=1).abs().mean()
            balance_loss = out["balance_loss"]
            loss = (
                recon_loss
                + args.commit_weight * out["commit_loss"]
                + args.align_weight * align_loss
                + args.disentangle_weight * disentangle_loss
                + args.balance_weight * balance_loss
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            totals["loss"] += float(loss.detach().cpu())
            totals["recon"] += float(recon_loss.detach().cpu())
            totals["commit"] += float(out["commit_loss"].detach().cpu())
            totals["align"] += float(align_loss.detach().cpu())
            totals["disentangle"] += float(disentangle_loss.detach().cpu())
            totals["balance"] += float(balance_loss.detach().cpu())
            num_batches += 1

        epoch_log = {
            "epoch": epoch,
            **{name: value / max(num_batches, 1) for name, value in totals.items()},
        }
        logs.append(epoch_log)
        if epoch == 1 or epoch == args.epochs or epoch % args.log_every == 0:
            print(json.dumps(epoch_log, ensure_ascii=False))
    return model, logs, device


@torch.no_grad()
def encode_codes(model, x_search, x_rec, device, batch_size):
    model.eval()
    dataset = TensorDataset(torch.from_numpy(x_search), torch.from_numpy(x_rec))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    prefix_codes = []
    search_codes = []
    rec_codes = []
    for batch_search, batch_rec in loader:
        out = model(batch_search.to(device), batch_rec.to(device))
        prefix_batch = torch.stack(out["prefix_idx"], dim=1).cpu().numpy().tolist()
        search_batch = torch.stack(out["search_idx"], dim=1).cpu().numpy().tolist()
        rec_batch = torch.stack(out["rec_idx"], dim=1).cpu().numpy().tolist()
        prefix_codes.extend(prefix_batch)
        search_codes.extend(search_batch)
        rec_codes.extend(rec_batch)
    return prefix_codes, search_codes, rec_codes


@torch.no_grad()
def encode_codes_with_assignment(model, x_search, x_rec, device, batch_size, assignment):
    if assignment == "argmin":
        return encode_codes(model, x_search, x_rec, device, batch_size)

    model.eval()
    z_shared_all = []
    r_search_all = []
    r_rec_all = []
    dataset = TensorDataset(torch.from_numpy(x_search), torch.from_numpy(x_rec))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for batch_search, batch_rec in loader:
        batch_search = batch_search.to(device)
        batch_rec = batch_rec.to(device)
        z_search = model.search_encoder(batch_search)
        z_rec = model.rec_encoder(batch_rec)
        gate_input = torch.cat([z_search, z_rec, z_search * z_rec, (z_search - z_rec).abs()], dim=1)
        gate = model.gate(gate_input)
        z_shared = gate * z_search + (1.0 - gate) * z_rec
        r_search = model.search_residual(torch.cat([z_search, z_shared, z_search - z_shared], dim=1))
        r_rec = model.rec_residual(torch.cat([z_rec, z_shared, z_rec - z_shared], dim=1))
        z_shared_all.append(z_shared.cpu().numpy())
        r_search_all.append(r_search.cpu().numpy())
        r_rec_all.append(r_rec.cpu().numpy())

    z_shared_np = np.concatenate(z_shared_all, axis=0)
    r_search_np = np.concatenate(r_search_all, axis=0)
    r_rec_np = np.concatenate(r_rec_all, axis=0)
    prefix_codes = rq_assign_codes_np(z_shared_np, model.shared_quantizer.codebooks, assignment)
    search_codes = rq_assign_codes_np(r_search_np, model.search_quantizer.codebooks, assignment)
    rec_codes = rq_assign_codes_np(r_rec_np, model.rec_quantizer.codebooks, assignment)
    return prefix_codes, search_codes, rec_codes


def token_list(prefix, tag):
    return [f"<{tag}{idx + 1}_{int(code)}>" for idx, code in enumerate(prefix)]


def build(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix_sizes = parse_sizes(args.prefix_codebook_sizes)
    search_sizes = parse_sizes(args.search_codebook_sizes)
    rec_sizes = parse_sizes(args.rec_codebook_sizes)

    item_ids, item_rows, has_metadata = load_item_rows(data_dir, args.item_ids, args.max_items)
    if not item_ids:
        raise RuntimeError("no items loaded; check --data-dir/--item-ids")

    query_docs, query_counter = aggregate_search_queries(data_dir, item_ids, args.max_queries_per_item)
    rec_docs_by_id, neighbor_counter = aggregate_rec_neighbors(
        data_dir,
        item_ids,
        args.neighbor_window,
        args.max_neighbors_per_item,
    )
    text_docs, search_docs, rec_docs = make_docs(item_ids, item_rows, query_docs, rec_docs_by_id)

    text_x = vectorize_reduce_text(text_docs, args.text_max_features, args.feature_dim, args.seed)
    search_x = vectorize_reduce_text(search_docs, args.text_max_features, args.feature_dim, args.seed + 100)
    rec_x = vectorize_reduce_hash(rec_docs, args.hash_features, args.feature_dim, args.seed + 200)
    x_search = np.concatenate([text_x, search_x], axis=1).astype(np.float32)
    x_rec = np.concatenate([text_x, rec_x], axis=1).astype(np.float32)

    model, train_log, device = train_model(args, x_search, x_rec, prefix_sizes, search_sizes, rec_sizes)
    prefix_codes, search_codes, rec_codes = encode_codes_with_assignment(
        model,
        x_search,
        x_rec,
        device,
        args.batch_size,
        args.assignment,
    )

    sid_mapping = {}
    search_sid_by_item = {}
    rec_sid_by_item = {}
    prefix_counter = Counter()
    for item_id, prefix_code, search_code, rec_code in zip(item_ids, prefix_codes, search_codes, rec_codes):
        prefix = token_list(prefix_code, "P")
        search_suffix = token_list(search_code, "S")
        rec_suffix = token_list(rec_code, "R")
        search_sid = prefix + search_suffix
        rec_sid = prefix + rec_suffix
        sid_mapping[item_id] = {
            "prefix": prefix,
            "search_sid": search_sid,
            "rec_sid": rec_sid,
            "search_suffix": search_suffix,
            "rec_suffix": rec_suffix,
            "prefix_codes": [int(x) for x in prefix_code],
            "search_suffix_codes": [int(x) for x in search_code],
            "rec_suffix_codes": [int(x) for x in rec_code],
        }
        search_sid_by_item[item_id] = search_sid
        rec_sid_by_item[item_id] = rec_sid
        prefix_counter[tuple(prefix)] += 1

    mapping_v2_path = output_dir / "sid_mapping.v2.json"
    write_json(mapping_v2_path, sid_mapping)
    write_json(output_dir / "sid_mapping.v1.json", sid_mapping)
    write_json(output_dir / "train_log.v2.json", train_log)
    torch.save(
        {
            "model_state_dict": model.cpu().state_dict(),
            "params": vars(args),
            "prefix_sizes": prefix_sizes,
            "search_sizes": search_sizes,
            "rec_sizes": rec_sizes,
        },
        output_dir / "neural_rq_sid_model.pt",
    )

    prefix_labels = ["_".join(map(str, codes)) for codes in prefix_codes]
    search_suffix_labels = ["_".join(map(str, codes)) for codes in search_codes]
    rec_suffix_labels = ["_".join(map(str, codes)) for codes in rec_codes]
    query_counts = [sum(query_counter[item_id].values()) for item_id in item_ids]
    neighbor_counts = [sum(neighbor_counter[item_id].values()) for item_id in item_ids]
    stats = {
        "num_items": len(item_ids),
        "has_metadata": has_metadata,
        "mapping_path": str(mapping_v2_path),
        "field_coverage": field_coverage(item_rows),
        "query_interaction_count": count_summary(query_counts),
        "rec_neighbor_count": count_summary(neighbor_counts),
        "codebook_sizes": {
            "prefix": prefix_sizes,
            "search_suffix": search_sizes,
            "rec_suffix": rec_sizes,
        },
        "prefix": {
            "unique_count": len(prefix_counter),
            "bucket_size": count_summary(list(prefix_counter.values())),
            "usage": code_usage(prefix_codes, prefix_sizes),
        },
        "search_suffix_usage": code_usage(search_codes, search_sizes),
        "rec_suffix_usage": code_usage(rec_codes, rec_sizes),
        "search_sid_collision": collision_rate(search_sid_by_item),
        "rec_sid_collision": collision_rate(rec_sid_by_item),
        "redundancy_proxy_nmi": {
            "prefix_search_suffix": normalized_mutual_info_score(prefix_labels, search_suffix_labels),
            "prefix_rec_suffix": normalized_mutual_info_score(prefix_labels, rec_suffix_labels),
        },
        "final_train_log": train_log[-1] if train_log else {},
        "params": vars(args),
        "note": "v2 uses neural search/rec encoders, gating-fused shared representation, residual suffix representations, and residual vector quantization.",
    }
    write_json(output_dir / "sid_stats.v2.json", stats)
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="/data/cjk/Search_Reconmendation/stage3/data/product_search_electronics",
    )
    parser.add_argument(
        "--output-dir",
        default="/data/cjk/Search_Reconmendation/stage3/data/product_search_electronics/sid_v2_neural_rq",
    )
    parser.add_argument("--item-ids", default="")
    parser.add_argument("--max-items", type=int, default=10000)
    parser.add_argument("--prefix-codebook-sizes", default="8,4")
    parser.add_argument("--search-codebook-sizes", default="32,8")
    parser.add_argument("--rec-codebook-sizes", default="32,8")
    parser.add_argument("--max-queries-per-item", type=int, default=20)
    parser.add_argument("--neighbor-window", type=int, default=2)
    parser.add_argument("--max-neighbors-per-item", type=int, default=200)
    parser.add_argument("--text-max-features", type=int, default=50000)
    parser.add_argument("--hash-features", type=int, default=32768)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--vq-beta", type=float, default=0.25)
    parser.add_argument("--commit-weight", type=float, default=0.25)
    parser.add_argument("--align-weight", type=float, default=0.05)
    parser.add_argument("--disentangle-weight", type=float, default=0.02)
    parser.add_argument("--balance-weight", type=float, default=0.05)
    parser.add_argument("--balance-temperature", type=float, default=0.05)
    parser.add_argument("--init-codebooks", choices=["kmeans", "random"], default="kmeans")
    parser.add_argument("--assignment", choices=["argmin", "balanced"], default="argmin")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    stats = build(args)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
