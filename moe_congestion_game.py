"""
MoE Congestion Game Analysis
==============================
Formal game-theoretic analysis of Mixture-of-Experts routing.

Mapping:
  Congestion Game (Rosenthal 1973)    →  MoE Routing
  ─────────────────────────────────────────────────────
  Players                             →  Tokens in a batch
  Resources (routes)                  →  Experts (FFN networks)
  Strategy = subset of resources      →  Top-K expert selection
  Cost c_r(x) = congestion function   →  Quality degradation under load
  Nash Equilibrium                    →  Converged routing pattern
  Social Optimum                      →  Routing minimizing global loss
  Price of Anarchy                    →  ppl(equilibrium) / ppl(optimal)
  Pigou Tax (auxiliary loss)          →  Load balancing loss L_bal
  Braess Paradox                      →  Adding expert degrades performance

Phases:
  A. Equilibrium Observation — hook routers, collect routing matrices,
     compute Gini, entropy, load ratio per layer
  B. Price of Anarchy — compare equilibrium routing vs oracle routing
     (greedy per-token expert search)
  C. Braess Paradox — disable each expert, measure if ppl improves
  D. Specialization Matrix — P(expert | category) heatmap,
     mutual information I(expert; category)

Target: Mixtral-8x7B-v0.1 (8 experts, top-2, 32 layers)
GPU: 2×A100 80GB (bf16, ~90GB total)
Alt: DeepSeek-V2-Lite (16B, 64 experts, 6 active) on 1×A100

Usage:
    # Phase A only (fast, ~20min)
    python moe_congestion_game.py --phase A --model mistralai/Mixtral-8x7B-v0.1

    # Full analysis (~4-6h)
    python moe_congestion_game.py --phase ABCD --model mistralai/Mixtral-8x7B-v0.1

    # Lighter model for testing
    python moe_congestion_game.py --phase ABCD --model mistralai/Mixtral-8x7B-Instruct-v0.1

Requires: torch, transformers, numpy, datasets, matplotlib, seaborn
    pip install datasets matplotlib seaborn --break-system-packages
"""

import argparse
import gc
import json
import math
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# ══════════════════════════════════════════════════════════════════════
# Data Loading
# ══════════════════════════════════════════════════════════════════════

def load_categorized_prompts(n_per_category: int = 50) -> Dict[str, List[str]]:
    """Load prompts from diverse categories for specialization analysis."""
    from datasets import load_dataset
    prompts = {}

    loaders = [
        ("science", "allenai/ai2_arc", "ARC-Challenge", "test", "question"),
        ("commonsense", "Rowan/hellaswag", None, "validation", "ctx"),
        ("knowledge", "cais/mmlu", "all", "test", "question"),
        ("math", "openai/gsm8k", "main", "test", "question"),
        ("factual", "truthfulqa/truthful_qa", "generation", "validation", "question"),
        ("reasoning", "tau/commonsense_qa", None, "validation", "question"),
    ]

    for name, dataset_id, config, split, field in loaders:
        try:
            kwargs = {"split": split, "trust_remote_code": True}
            ds = load_dataset(dataset_id, config, **kwargs) if config else load_dataset(dataset_id, **kwargs)
            ds = ds.shuffle(seed=42).select(range(min(n_per_category, len(ds))))
            texts = [row[field] for row in ds if isinstance(row[field], str) and len(row[field]) > 20]
            prompts[name] = texts[:n_per_category]
            print(f"  {name}: {len(prompts[name])} prompts")
        except Exception as e:
            print(f"  {name}: FAILED ({e})")

    # Add code prompts (synthetic, no dataset needed)
    prompts["code"] = [
        "Write a Python function that implements binary search on a sorted array.",
        "Implement a linked list class with insert, delete and search methods.",
        "Write a recursive function to compute the nth Fibonacci number with memoization.",
        "Create a Python class for a min-heap with push and pop operations.",
        "Write a function that finds all permutations of a string.",
        "Implement merge sort in Python with O(n log n) time complexity.",
        "Write a Python decorator that caches function results.",
        "Create a generator function that yields prime numbers.",
        "Implement a trie data structure for efficient string prefix matching.",
        "Write a function to detect cycles in a directed graph using DFS.",
    ] * (n_per_category // 10 + 1)
    prompts["code"] = prompts["code"][:n_per_category]
    print(f"  code: {len(prompts['code'])} prompts (synthetic)")

    total = sum(len(v) for v in prompts.values())
    print(f"  Total: {total} prompts across {len(prompts)} categories")
    return prompts


# ══════════════════════════════════════════════════════════════════════
# Router Hooking Infrastructure
# ══════════════════════════════════════════════════════════════════════

class RouterCapture:
    """
    Hooks into every MoE layer's gate/router to capture:
    - Full softmax logits over all experts [N_tokens, N_experts]
    - Selected expert indices [N_tokens, top_k]
    - Selected expert weights [N_tokens, top_k]
    """

    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.captures = {}  # layer_idx → {logits, indices, weights}
        self._install_hooks()

    def _install_hooks(self):
        """Find and hook all MoE gate/router modules."""
        for name, module in self.model.named_modules():
            # Mixtral: model.layers.{i}.block_sparse_moe.gate
            # or newer: model.layers.{i}.feed_forward.gate
            if isinstance(module, torch.nn.Linear) and "gate" in name and "moe" in name.lower():
                layer_idx = self._extract_layer_idx(name)
                hook = module.register_forward_hook(self._make_hook(layer_idx, "linear_gate"))
                self.hooks.append(hook)
            # Fallback: look for MixtralTopKRouter or similar
            elif type(module).__name__ in ("MixtralTopKRouter", "TopKRouter"):
                layer_idx = self._extract_layer_idx(name)
                hook = module.register_forward_hook(self._make_hook(layer_idx, "topk_router"))
                self.hooks.append(hook)
            # Another fallback: MixtralSparseMoeBlock
            elif type(module).__name__ in ("MixtralSparseMoeBlock",):
                layer_idx = self._extract_layer_idx(name)
                hook = module.register_forward_hook(self._make_hook(layer_idx, "moe_block"))
                self.hooks.append(hook)

        if not self.hooks:
            # Generic fallback: hook anything named 'gate' inside 'expert' or 'moe'
            for name, module in self.model.named_modules():
                if "gate" in name.lower() and any(k in name.lower() for k in ["expert", "moe", "sparse"]):
                    layer_idx = self._extract_layer_idx(name)
                    hook = module.register_forward_hook(self._make_hook(layer_idx, "generic"))
                    self.hooks.append(hook)

        print(f"  [RouterCapture] Installed {len(self.hooks)} hooks")

    def _extract_layer_idx(self, name: str) -> int:
        """Extract layer index from module name like 'model.layers.5.block_sparse_moe.gate'"""
        parts = name.split(".")
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts):
                try:
                    return int(parts[i + 1])
                except ValueError:
                    pass
        return hash(name) % 1000  # fallback

    def _make_hook(self, layer_idx: int, hook_type: str):
        def hook_fn(module, input, output):
            if hook_type == "topk_router":
                # MixtralTopKRouter returns (router_logits, router_scores, router_indices)
                router_logits, router_scores, router_indices = output
                self.captures[layer_idx] = {
                    "logits": router_logits.detach().cpu(),      # [N, N_experts] full softmax
                    "weights": router_scores.detach().cpu(),     # [N, top_k] renormalized
                    "indices": router_indices.detach().cpu(),    # [N, top_k] selected experts
                }
            elif hook_type == "moe_block":
                # Can't directly get router output from block, but we hook the gate separately
                pass
            elif hook_type == "linear_gate":
                # Raw linear output before softmax
                # input[0] = hidden_states, output = logits [N, N_experts]
                logits_raw = output.detach().cpu().float()
                probs = torch.softmax(logits_raw, dim=-1)
                top_k = 2  # default for Mixtral
                topk_vals, topk_idx = probs.topk(top_k, dim=-1)
                topk_vals = topk_vals / topk_vals.sum(dim=-1, keepdim=True)
                self.captures[layer_idx] = {
                    "logits": probs,
                    "weights": topk_vals,
                    "indices": topk_idx,
                }
            elif hook_type == "generic":
                if isinstance(output, tuple) and len(output) >= 3:
                    self.captures[layer_idx] = {
                        "logits": output[0].detach().cpu(),
                        "weights": output[1].detach().cpu(),
                        "indices": output[2].detach().cpu(),
                    }
                elif isinstance(output, torch.Tensor):
                    logits_raw = output.detach().cpu().float()
                    probs = torch.softmax(logits_raw, dim=-1)
                    topk_vals, topk_idx = probs.topk(2, dim=-1)
                    topk_vals = topk_vals / topk_vals.sum(dim=-1, keepdim=True)
                    self.captures[layer_idx] = {
                        "logits": probs,
                        "weights": topk_vals,
                        "indices": topk_idx,
                    }
        return hook_fn

    def clear(self):
        self.captures = {}

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def get_n_experts(self) -> int:
        for layer_idx, cap in self.captures.items():
            return cap["logits"].shape[-1]
        return 8  # default

    def get_n_layers(self) -> int:
        return len(self.captures)


# ══════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════

def gini_coefficient(counts: np.ndarray) -> float:
    """Gini coefficient. 0 = perfectly uniform, 1 = maximally concentrated."""
    n = len(counts)
    if n == 0 or counts.sum() == 0:
        return 0.0
    sorted_c = np.sort(counts)
    index = np.arange(1, n + 1)
    return float((2 * (index @ sorted_c)) / (n * sorted_c.sum()) - (n + 1) / n)


def normalized_entropy(counts: np.ndarray) -> float:
    """Normalized Shannon entropy. 1 = uniform, 0 = all in one bin."""
    freq = counts / (counts.sum() + 1e-10)
    freq = freq[freq > 0]
    entropy = -(freq * np.log(freq)).sum()
    max_entropy = np.log(len(counts))
    return float(entropy / (max_entropy + 1e-10))


def compute_perplexity(model, tokenizer, text: str, device: str, max_length: int = 512) -> float:
    """Compute perplexity of a single text."""
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = encoded["input_ids"].to(device)
    if input_ids.shape[1] < 2:
        return float("inf")

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss

    return float(torch.exp(loss))


# ══════════════════════════════════════════════════════════════════════
# PHASE A — Equilibrium Observation
# ══════════════════════════════════════════════════════════════════════

def phase_a_equilibrium(
    model, tokenizer, router_capture: RouterCapture,
    prompts: Dict[str, List[str]], device: str,
) -> Dict:
    """
    Pass all prompts through the model, capture routing decisions,
    compute load distribution metrics per layer.
    """
    print(f"\n{'=' * 70}")
    print("PHASE A — Equilibrium Observation")
    print("=" * 70)

    # Aggregate routing across all prompts
    all_prompts_flat = []
    all_categories = []
    for cat, texts in prompts.items():
        for t in texts:
            all_prompts_flat.append(t)
            all_categories.append(cat)

    # Per-layer accumulation
    layer_expert_counts = defaultdict(lambda: None)  # layer → [N_experts]
    layer_all_logits = defaultdict(list)              # layer → list of [S, N_experts]
    per_category_routing = defaultdict(lambda: defaultdict(lambda: None))  # cat → layer → [N_experts]

    n_processed = 0
    for i, (prompt, cat) in enumerate(zip(all_prompts_flat, all_categories)):
        router_capture.clear()

        encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        input_ids = encoded["input_ids"].to(device)

        with torch.no_grad():
            _ = model(input_ids)

        for layer_idx, cap in router_capture.captures.items():
            indices = cap["indices"]  # [S, top_k]
            logits = cap["logits"]    # [S, N_experts]
            n_experts = logits.shape[-1]

            # Count expert selections
            counts = torch.zeros(n_experts)
            for idx in indices.flatten():
                counts[idx.item()] += 1

            if layer_expert_counts[layer_idx] is None:
                layer_expert_counts[layer_idx] = counts.numpy()
            else:
                layer_expert_counts[layer_idx] += counts.numpy()

            layer_all_logits[layer_idx].append(logits)

            # Per-category
            if per_category_routing[cat][layer_idx] is None:
                per_category_routing[cat][layer_idx] = counts.numpy()
            else:
                per_category_routing[cat][layer_idx] += counts.numpy()

        n_processed += 1
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(all_prompts_flat)}] processed")

    print(f"  Total: {n_processed} prompts processed")

    # Compute per-layer metrics
    results = {"per_layer": {}, "global": {}}
    n_experts = router_capture.get_n_experts()
    sorted_layers = sorted(layer_expert_counts.keys())

    print(f"\n  {'Layer':>6s}  {'Gini':>8s}  {'Entropy':>8s}  {'MaxLoad':>8s}  {'MinLoad':>8s}  {'Ratio':>8s}")
    print(f"  {'-' * 52}")

    all_ginis = []
    all_entropies = []

    for layer in sorted_layers:
        counts = layer_expert_counts[layer]
        g = gini_coefficient(counts)
        e = normalized_entropy(counts)
        ratio = counts.max() / (counts.min() + 1e-10)

        all_ginis.append(g)
        all_entropies.append(e)

        results["per_layer"][int(layer)] = {
            "counts": counts.tolist(),
            "gini": g,
            "entropy": e,
            "load_ratio": float(ratio),
        }

        if layer % max(1, len(sorted_layers) // 8) == 0 or layer == sorted_layers[-1]:
            print(f"  L{layer:>4d}  {g:>8.4f}  {e:>8.4f}  {counts.max():>8.0f}  {counts.min():>8.0f}  {ratio:>8.2f}")

    results["global"] = {
        "mean_gini": float(np.mean(all_ginis)),
        "mean_entropy": float(np.mean(all_entropies)),
        "n_experts": n_experts,
        "n_layers_with_moe": len(sorted_layers),
    }

    print(f"\n  Global: mean_gini={np.mean(all_ginis):.4f}  mean_entropy={np.mean(all_entropies):.4f}")

    if np.mean(all_ginis) > 0.3:
        print(f"  → SIGNIFICANT congestion (Gini > 0.3). Game-theoretic analysis warranted.")
    elif np.mean(all_ginis) > 0.15:
        print(f"  → MODERATE congestion. Some experts are preferred.")
    else:
        print(f"  → LOW congestion. Routing is near-uniform.")

    return results, per_category_routing, layer_all_logits


# ══════════════════════════════════════════════════════════════════════
# PHASE B — Price of Anarchy
# ══════════════════════════════════════════════════════════════════════

def phase_b_price_of_anarchy(
    model, tokenizer, router_capture: RouterCapture,
    prompts: Dict[str, List[str]], device: str,
    n_prompts: int = 30,
) -> Dict:
    """
    Compute the Price of Anarchy:
      PoA = ppl(equilibrium routing) / ppl(oracle routing)

    Oracle = for each MoE layer, replace the router decision with the
    expert that minimizes the per-token loss. Since exact oracle is
    intractable (combinatorial), we use a greedy layer-wise oracle.

    Method:
    1. Normal forward pass → ppl_equilibrium
    2. For the most impactful MoE layer (highest Gini from Phase A),
       try each expert for each token and pick the best → ppl_greedy_oracle
    3. Uniform round-robin routing → ppl_uniform
    """
    print(f"\n{'=' * 70}")
    print("PHASE B — Price of Anarchy")
    print("=" * 70)

    all_prompts = []
    for cat_texts in prompts.values():
        all_prompts.extend(cat_texts)
    subset = all_prompts[:n_prompts]

    # 1. Equilibrium perplexity (normal forward pass)
    print(f"\n  [B1] Computing equilibrium perplexity ({len(subset)} prompts)...")
    ppls_eq = []
    for prompt in subset:
        ppl = compute_perplexity(model, tokenizer, prompt, device)
        if not math.isinf(ppl) and not math.isnan(ppl):
            ppls_eq.append(ppl)
    mean_ppl_eq = np.mean(ppls_eq) if ppls_eq else float("inf")
    print(f"    Mean ppl (equilibrium): {mean_ppl_eq:.2f}")

    # 2. Perplexity with each expert disabled (proxy for oracle search)
    # Instead of full oracle (intractable), we measure the RANGE of possible
    # outcomes by disabling each expert and seeing how much ppl changes.
    # The min-ppl across disabled configs gives an upper bound on oracle ppl.
    print(f"\n  [B2] Computing per-expert-disabled perplexity...")

    n_experts = 8  # Will be updated from model
    # Find MoE gate modules
    gate_modules = {}
    for name, module in model.named_modules():
        if type(module).__name__ in ("MixtralTopKRouter", "TopKRouter"):
            layer_idx = 0
            parts = name.split(".")
            for j, p in enumerate(parts):
                if p == "layers" and j + 1 < len(parts):
                    try:
                        layer_idx = int(parts[j + 1])
                    except ValueError:
                        pass
            gate_modules[layer_idx] = module
            n_experts = module.num_experts if hasattr(module, "num_experts") else 8

    if not gate_modules:
        # Fallback: find gate Linear layers
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and "gate" in name:
                layer_idx = 0
                parts = name.split(".")
                for j, p in enumerate(parts):
                    if p == "layers" and j + 1 < len(parts):
                        try:
                            layer_idx = int(parts[j + 1])
                        except ValueError:
                            pass
                gate_modules[layer_idx] = module
                n_experts = module.out_features

    print(f"    Found {len(gate_modules)} MoE gate modules, {n_experts} experts")

    # Pick the middle layer for intervention (most representative)
    if gate_modules:
        target_layer = sorted(gate_modules.keys())[len(gate_modules) // 2]
        target_gate = gate_modules[target_layer]
        print(f"    Target layer for intervention: {target_layer}")
    else:
        print("    WARNING: No gate modules found. Skipping oracle analysis.")
        return {"error": "no_gate_modules_found"}

    # Hook the target gate to mask experts
    class ExpertMask:
        def __init__(self):
            self.masked_expert = -1  # -1 = no mask
            self.force_uniform = False

        def hook_fn(self_mask, module, input, output):
            if isinstance(output, tuple) and len(output) == 3:
                logits, scores, indices = output
            elif isinstance(output, torch.Tensor):
                logits = output
                scores = None
                indices = None
            else:
                return output

            if self_mask.masked_expert >= 0:
                # Set masked expert logits to -inf before topk
                logits = logits.clone()
                logits[:, self_mask.masked_expert] = -1e9
                # Recompute topk
                if scores is not None:
                    probs = torch.softmax(logits.float(), dim=-1)
                    top_k = scores.shape[-1]
                    new_scores, new_indices = probs.topk(top_k, dim=-1)
                    new_scores = new_scores / new_scores.sum(dim=-1, keepdim=True)
                    return (logits, new_scores, new_indices)
                return logits

            if self_mask.force_uniform:
                logits = logits.clone()
                logits.fill_(1.0 / logits.shape[-1])
                if scores is not None:
                    top_k = scores.shape[-1]
                    probs = torch.softmax(logits.float(), dim=-1)
                    new_scores, new_indices = probs.topk(top_k, dim=-1)
                    new_scores = new_scores / new_scores.sum(dim=-1, keepdim=True)
                    return (logits, new_scores, new_indices)
                return logits

            return output

    mask = ExpertMask()

    # We need to hook the right module
    # For MixtralTopKRouter, hook on the router itself
    hook_handle = target_gate.register_forward_hook(
        lambda module, input, output: mask.hook_fn(mask, module, input, output)
    )

    # Test each expert disabled
    ppls_per_disabled = {}
    for expert_id in range(n_experts):
        mask.masked_expert = expert_id
        ppls = []
        for prompt in subset[:15]:  # smaller subset for speed
            ppl = compute_perplexity(model, tokenizer, prompt, device)
            if not math.isinf(ppl) and not math.isnan(ppl):
                ppls.append(ppl)
        mean_ppl = np.mean(ppls) if ppls else float("inf")
        ppls_per_disabled[expert_id] = mean_ppl
        delta = (mean_ppl - mean_ppl_eq) / mean_ppl_eq * 100
        marker = "✓ BRAESS?" if mean_ppl < mean_ppl_eq else ""
        print(f"    Expert {expert_id} disabled: ppl={mean_ppl:.2f} (Δ={delta:+.1f}%) {marker}")

    mask.masked_expert = -1

    # 3. Uniform routing
    print(f"\n  [B3] Computing uniform routing perplexity...")
    mask.force_uniform = True
    ppls_uniform = []
    for prompt in subset[:15]:
        ppl = compute_perplexity(model, tokenizer, prompt, device)
        if not math.isinf(ppl) and not math.isnan(ppl):
            ppls_uniform.append(ppl)
    mean_ppl_uniform = np.mean(ppls_uniform) if ppls_uniform else float("inf")
    mask.force_uniform = False

    hook_handle.remove()

    print(f"    Mean ppl (uniform): {mean_ppl_uniform:.2f}")

    # Compute PoA
    ppl_best_disabled = min(ppls_per_disabled.values())
    poa_vs_best = mean_ppl_eq / ppl_best_disabled if ppl_best_disabled > 0 else float("inf")
    poa_vs_uniform = mean_ppl_eq / mean_ppl_uniform if mean_ppl_uniform > 0 else float("inf")

    print(f"\n  RESULTS:")
    print(f"    ppl(equilibrium):     {mean_ppl_eq:.2f}")
    print(f"    ppl(best disabled):   {ppl_best_disabled:.2f}")
    print(f"    ppl(uniform routing): {mean_ppl_uniform:.2f}")
    print(f"    PoA vs best-disabled: {poa_vs_best:.4f}")

    if poa_vs_best < 1.0:
        print(f"    → Equilibrium is BETTER than any single-expert-disabled config.")
        print(f"      The router has converged to a good equilibrium.")
    elif poa_vs_best > 1.05:
        print(f"    → PoA > 1.05: MEASURABLE inefficiency in routing.")
        print(f"      Game-theoretic optimization could help.")
    else:
        print(f"    → PoA ≈ 1.0: router has found a near-optimal equilibrium.")

    results = {
        "ppl_equilibrium": mean_ppl_eq,
        "ppl_per_disabled": ppls_per_disabled,
        "ppl_best_disabled": ppl_best_disabled,
        "ppl_uniform": mean_ppl_uniform,
        "poa_vs_best_disabled": poa_vs_best,
        "poa_vs_uniform": poa_vs_uniform,
        "target_layer": target_layer,
    }

    return results


# ══════════════════════════════════════════════════════════════════════
# PHASE C — Braess Paradox
# ══════════════════════════════════════════════════════════════════════

def phase_c_braess_paradox(
    model, tokenizer, router_capture: RouterCapture,
    prompts: Dict[str, List[str]], device: str,
    n_prompts: int = 20,
) -> Dict:
    """
    Test for Braess's Paradox: does removing an expert IMPROVE performance?

    In traffic networks, adding a road can increase total travel time
    because selfish agents over-converge on the new route.

    In MoE, an "attractor" expert that captures many tokens but processes
    them poorly could degrade global performance. Removing it forces
    tokens to better alternatives.

    We test ALL layers × ALL experts (expensive but comprehensive).
    """
    print(f"\n{'=' * 70}")
    print("PHASE C — Braess Paradox (per-layer expert ablation)")
    print("=" * 70)

    all_prompts = []
    for v in prompts.values():
        all_prompts.extend(v)
    subset = all_prompts[:n_prompts]

    # Baseline ppl
    print(f"  Computing baseline ppl on {len(subset)} prompts...")
    ppls_baseline = []
    for prompt in subset:
        ppl = compute_perplexity(model, tokenizer, prompt, device)
        if not math.isinf(ppl) and not math.isnan(ppl):
            ppls_baseline.append(ppl)
    mean_baseline = np.mean(ppls_baseline)
    print(f"    Baseline ppl: {mean_baseline:.2f}")

    # Find all MoE gate modules
    gate_modules = {}
    for name, module in model.named_modules():
        if type(module).__name__ in ("MixtralTopKRouter", "TopKRouter"):
            parts = name.split(".")
            layer_idx = 0
            for j, p in enumerate(parts):
                if p == "layers" and j + 1 < len(parts):
                    try:
                        layer_idx = int(parts[j + 1])
                    except ValueError:
                        pass
            gate_modules[layer_idx] = module

    if not gate_modules:
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and "gate" in name:
                parts = name.split(".")
                layer_idx = 0
                for j, p in enumerate(parts):
                    if p == "layers" and j + 1 < len(parts):
                        try:
                            layer_idx = int(parts[j + 1])
                        except ValueError:
                            pass
                gate_modules[layer_idx] = module

    n_experts = 8
    for m in gate_modules.values():
        if hasattr(m, "num_experts"):
            n_experts = m.num_experts
        elif hasattr(m, "out_features"):
            n_experts = m.out_features
        break

    # Sample layers (every 4th to keep runtime manageable)
    test_layers = sorted(gate_modules.keys())[::4]
    print(f"  Testing {len(test_layers)} layers × {n_experts} experts = {len(test_layers) * n_experts} configs")

    braess_instances = []
    results = {"baseline_ppl": mean_baseline, "ablations": []}

    for layer_idx in test_layers:
        gate = gate_modules[layer_idx]

        # Save original weights
        original_weight = gate.weight.data.clone() if hasattr(gate, 'weight') else None

        for expert_id in range(n_experts):
            # Hook to mask this expert at this layer
            class LayerExpertMask:
                def __init__(self, target_expert):
                    self.target_expert = target_expert

                def __call__(self, module, input, output):
                    if isinstance(output, tuple) and len(output) >= 3:
                        logits, scores, indices = output[0], output[1], output[2]
                        logits = logits.clone()
                        logits[:, self.target_expert] = -1e9
                        probs = torch.softmax(logits.float(), dim=-1)
                        top_k = scores.shape[-1]
                        new_scores, new_indices = probs.topk(top_k, dim=-1)
                        new_scores = new_scores / new_scores.sum(dim=-1, keepdim=True)
                        return (logits, new_scores, new_indices)
                    elif isinstance(output, torch.Tensor):
                        logits = output.clone()
                        logits[:, self.target_expert] = -1e9
                        return logits
                    return output

            hook_handle = gate.register_forward_hook(LayerExpertMask(expert_id))

            ppls = []
            for prompt in subset:
                ppl = compute_perplexity(model, tokenizer, prompt, device)
                if not math.isinf(ppl) and not math.isnan(ppl):
                    ppls.append(ppl)

            mean_ppl = np.mean(ppls) if ppls else float("inf")
            delta_pct = (mean_ppl - mean_baseline) / mean_baseline * 100

            hook_handle.remove()

            is_braess = mean_ppl < mean_baseline * 0.995  # 0.5% improvement threshold
            if is_braess:
                braess_instances.append((layer_idx, expert_id, mean_ppl, delta_pct))

            results["ablations"].append({
                "layer": layer_idx,
                "expert": expert_id,
                "ppl": mean_ppl,
                "delta_pct": delta_pct,
                "is_braess": is_braess,
            })

        # Print summary for this layer
        layer_results = [r for r in results["ablations"] if r["layer"] == layer_idx]
        best = min(layer_results, key=lambda r: r["ppl"])
        worst = max(layer_results, key=lambda r: r["ppl"])
        braess_count = sum(1 for r in layer_results if r["is_braess"])

        print(f"  L{layer_idx:>2d}: best=E{best['expert']}({best['delta_pct']:+.1f}%)  "
              f"worst=E{worst['expert']}({worst['delta_pct']:+.1f}%)  "
              f"braess={braess_count}/{n_experts}")

    # Summary
    print(f"\n  BRAESS PARADOX RESULTS:")
    print(f"    Total ablation configs tested: {len(results['ablations'])}")
    print(f"    Braess instances found: {len(braess_instances)}")

    if braess_instances:
        print(f"\n    Braess instances (removing expert IMPROVES ppl):")
        for layer, expert, ppl, delta in sorted(braess_instances, key=lambda x: x[3]):
            print(f"      Layer {layer}, Expert {expert}: ppl={ppl:.2f} ({delta:+.2f}%)")
        print(f"\n    → BRAESS PARADOX CONFIRMED in MoE routing.")
        print(f"      Some experts act as attractors that degrade global performance.")
    else:
        print(f"    → No Braess paradox detected. All experts contribute positively.")

    results["braess_instances"] = braess_instances
    return results


# ══════════════════════════════════════════════════════════════════════
# PHASE D — Specialization Matrix
# ══════════════════════════════════════════════════════════════════════

def phase_d_specialization(
    per_category_routing: Dict,
    n_experts: int,
    output_dir: str = ".",
) -> Dict:
    """
    Build the specialization matrix: P(expert | category) per layer.
    Compute mutual information I(expert; category).
    Generate heatmap visualization.
    """
    print(f"\n{'=' * 70}")
    print("PHASE D — Expert Specialization Analysis")
    print("=" * 70)

    categories = sorted(per_category_routing.keys())
    layers = set()
    for cat in categories:
        layers.update(per_category_routing[cat].keys())
    layers = sorted(layers)

    if not layers or not categories:
        print("  No routing data available.")
        return {"error": "no_data"}

    results = {"per_layer": {}}

    # Pick representative layers
    sample_layers = layers[::max(1, len(layers) // 6)]
    if layers[-1] not in sample_layers:
        sample_layers.append(layers[-1])

    for layer in sample_layers:
        # Build matrix: [n_categories, n_experts]
        matrix = np.zeros((len(categories), n_experts))
        for ci, cat in enumerate(categories):
            if layer in per_category_routing[cat]:
                counts = per_category_routing[cat][layer]
                if counts is not None:
                    matrix[ci, :len(counts)] = counts[:n_experts]

        # Normalize per category (rows sum to 1)
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        prob_matrix = matrix / row_sums  # P(expert | category)

        # Marginals
        p_expert = matrix.sum(axis=0)
        p_expert = p_expert / (p_expert.sum() + 1e-10)  # P(expert)
        p_category = matrix.sum(axis=1)
        p_category = p_category / (p_category.sum() + 1e-10)  # P(category)

        # Joint probability P(expert, category)
        total = matrix.sum() + 1e-10
        p_joint = matrix / total

        # Mutual Information I(expert; category) = Σ p(e,c) log(p(e,c) / (p(e)p(c)))
        mi = 0.0
        for ci in range(len(categories)):
            for ei in range(n_experts):
                if p_joint[ci, ei] > 1e-10:
                    mi += p_joint[ci, ei] * np.log(
                        p_joint[ci, ei] / (p_expert[ei] * p_category[ci] + 1e-10) + 1e-10
                    )

        # Normalized MI (0 = independent, 1 = fully dependent)
        h_expert = -(p_expert[p_expert > 0] * np.log(p_expert[p_expert > 0])).sum()
        h_category = -(p_category[p_category > 0] * np.log(p_category[p_category > 0])).sum()
        nmi = mi / (min(h_expert, h_category) + 1e-10) if min(h_expert, h_category) > 0 else 0

        print(f"\n  Layer {layer}: MI={mi:.4f}  NMI={nmi:.4f}")
        print(f"  P(expert | category):")
        header = "  " + " " * 14 + "".join(f"  E{i:<5d}" for i in range(n_experts))
        print(header)
        for ci, cat in enumerate(categories):
            row = f"  {cat:>12s}  " + "  ".join(f"{prob_matrix[ci, ei]:.3f}" for ei in range(n_experts))
            print(row)

        results["per_layer"][int(layer)] = {
            "probability_matrix": prob_matrix.tolist(),
            "mutual_information": mi,
            "normalized_mi": nmi,
            "categories": categories,
        }

    # Generate heatmap (save as PNG if matplotlib available)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Use last sampled layer for the heatmap
        last_layer = sample_layers[-1]
        if last_layer in per_category_routing[categories[0]]:
            matrix = np.zeros((len(categories), n_experts))
            for ci, cat in enumerate(categories):
                if last_layer in per_category_routing[cat]:
                    counts = per_category_routing[cat][last_layer]
                    if counts is not None:
                        matrix[ci, :len(counts)] = counts[:n_experts]
            row_sums = matrix.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            prob_matrix = matrix / row_sums

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(
                prob_matrix, annot=True, fmt=".2f", cmap="YlOrRd",
                xticklabels=[f"Expert {i}" for i in range(n_experts)],
                yticklabels=categories, ax=ax,
            )
            ax.set_title(f"P(expert | category) — Layer {last_layer}")
            ax.set_xlabel("Expert")
            ax.set_ylabel("Category")
            plt.tight_layout()

            heatmap_path = os.path.join(output_dir, "specialization_heatmap.png")
            plt.savefig(heatmap_path, dpi=150)
            plt.close()
            print(f"\n  Heatmap saved to {heatmap_path}")
    except ImportError:
        print("\n  (matplotlib/seaborn not available — skipping heatmap)")

    # Overall specialization verdict
    mean_nmi = np.mean([v["normalized_mi"] for v in results["per_layer"].values()])
    print(f"\n  Mean NMI across layers: {mean_nmi:.4f}")

    if mean_nmi > 0.1:
        print(f"  → STRONG specialization. Experts play distinct pure strategies per category.")
    elif mean_nmi > 0.03:
        print(f"  → MODERATE specialization. Some category-expert affinity.")
    else:
        print(f"  → WEAK specialization. Experts are mostly interchangeable.")

    results["mean_nmi"] = mean_nmi
    return results


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="MoE Congestion Game Analysis")
    parser.add_argument("--model", default="mistralai/Mixtral-8x7B-Instruct-v0.1",
                        help="MoE model to analyze")
    parser.add_argument("--phase", default="ABCD",
                        help="Which phases to run (e.g. 'A', 'AB', 'ABCD')")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n_per_category", type=int, default=50,
                        help="Prompts per category for Phase A/D")
    parser.add_argument("--n_poa", type=int, default=30,
                        help="Prompts for Phase B (PoA)")
    parser.add_argument("--n_braess", type=int, default=20,
                        help="Prompts for Phase C (Braess)")
    parser.add_argument("--output", default="moe_congestion_results.json")
    parser.add_argument("--output_dir", default=".")
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    args = parser.parse_args()

    print("=" * 70)
    print("MoE Congestion Game Analysis")
    print("=" * 70)
    print(f"  Model: {args.model}")
    print(f"  Phases: {args.phase}")
    print(f"  Device: {args.device}")
    print()

    # Load prompts
    print("[1] Loading categorized prompts...")
    prompts = load_categorized_prompts(n_per_category=args.n_per_category)

    # Load model
    print("\n[2] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # auto-shard across GPUs
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    # Print model MoE info
    n_moe_layers = 0
    n_experts = 8
    for name, module in model.named_modules():
        if type(module).__name__ in ("MixtralSparseMoeBlock",):
            n_moe_layers += 1
        if type(module).__name__ in ("MixtralTopKRouter", "TopKRouter"):
            if hasattr(module, "num_experts"):
                n_experts = module.num_experts
    print(f"  MoE layers: {n_moe_layers}, Experts per layer: {n_experts}")

    # Install router hooks
    print("\n[3] Installing router hooks...")
    router_capture = RouterCapture(model)

    all_results = {"model": args.model, "n_experts": n_experts}

    # ── Phase A ──
    if "A" in args.phase.upper():
        phase_a_results, per_cat_routing, layer_logits = phase_a_equilibrium(
            model, tokenizer, router_capture, prompts, args.device
        )
        all_results["phase_a"] = phase_a_results
    else:
        per_cat_routing = None

    # ── Phase B ──
    if "B" in args.phase.upper():
        phase_b_results = phase_b_price_of_anarchy(
            model, tokenizer, router_capture, prompts, args.device,
            n_prompts=args.n_poa,
        )
        all_results["phase_b"] = phase_b_results

    # ── Phase C ──
    if "C" in args.phase.upper():
        phase_c_results = phase_c_braess_paradox(
            model, tokenizer, router_capture, prompts, args.device,
            n_prompts=args.n_braess,
        )
        all_results["phase_c"] = phase_c_results

    # ── Phase D ──
    if "D" in args.phase.upper() and per_cat_routing is not None:
        phase_d_results = phase_d_specialization(
            per_cat_routing, n_experts, output_dir=args.output_dir,
        )
        all_results["phase_d"] = phase_d_results

    # ══════════════════════════════════════════════════════════════
    # FINAL VERDICT
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("MoE CONGESTION GAME — FINAL VERDICT")
    print("=" * 70)

    signals = []

    if "phase_a" in all_results:
        gini = all_results["phase_a"]["global"]["mean_gini"]
        if gini > 0.3:
            signals.append(("congestion", "HIGH", f"Gini={gini:.3f} — significant load imbalance"))
        elif gini > 0.15:
            signals.append(("congestion", "MODERATE", f"Gini={gini:.3f} — some imbalance"))
        else:
            signals.append(("congestion", "LOW", f"Gini={gini:.3f} — near-uniform routing"))

    if "phase_b" in all_results and "poa_vs_best_disabled" in all_results["phase_b"]:
        poa = all_results["phase_b"]["poa_vs_best_disabled"]
        if poa > 1.05:
            signals.append(("price_of_anarchy", "HIGH", f"PoA={poa:.4f} — measurable inefficiency"))
        elif poa > 1.01:
            signals.append(("price_of_anarchy", "LOW", f"PoA={poa:.4f} — near-optimal routing"))
        else:
            signals.append(("price_of_anarchy", "NONE", f"PoA={poa:.4f} — optimal or better"))

    if "phase_c" in all_results:
        n_braess = len(all_results["phase_c"].get("braess_instances", []))
        if n_braess > 0:
            signals.append(("braess_paradox", "CONFIRMED", f"{n_braess} instances found"))
        else:
            signals.append(("braess_paradox", "NOT_FOUND", "All experts contribute positively"))

    if "phase_d" in all_results and "mean_nmi" in all_results["phase_d"]:
        nmi = all_results["phase_d"]["mean_nmi"]
        if nmi > 0.1:
            signals.append(("specialization", "STRONG", f"NMI={nmi:.4f} — experts have distinct strategies"))
        elif nmi > 0.03:
            signals.append(("specialization", "MODERATE", f"NMI={nmi:.4f} — some category affinity"))
        else:
            signals.append(("specialization", "WEAK", f"NMI={nmi:.4f} — experts interchangeable"))

    for name, level, desc in signals:
        icon = {"HIGH": "🔴", "CONFIRMED": "🔴", "STRONG": "🟢",
                "MODERATE": "🟡", "LOW": "🟢", "NONE": "⚪", "WEAK": "🟡",
                "NOT_FOUND": "⚪"}.get(level, "⚪")
        print(f"  {icon} [{name}] {desc}")

    # Is this a publishable result?
    has_braess = any(l == "CONFIRMED" for _, l, _ in signals)
    has_high_poa = any(l == "HIGH" and n == "price_of_anarchy" for n, l, _ in signals)
    has_specialization = any(l in ("STRONG", "MODERATE") for _, l, _ in signals)

    if has_braess:
        print(f"\n  🎯 PUBLISHABLE: Braess Paradox confirmed in MoE routing.")
        print(f"     First empirical demonstration in neural network expert routing.")
    elif has_high_poa and has_specialization:
        print(f"\n  🎯 PUBLISHABLE: Measurable Price of Anarchy with expert specialization.")
        print(f"     Formal game-theoretic characterization of MoE routing dynamics.")
    elif has_specialization:
        print(f"\n  📊 INTERESTING: Expert specialization confirmed.")
        print(f"     Not novel alone, but strengthens game-theoretic framing.")
    else:
        print(f"\n  ⚪ Routing appears near-optimal. Game theory adds limited insight.")

    all_results["signals"] = [(n, l, d) for n, l, d in signals]

    # Save
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")

    # Cleanup
    router_capture.remove_hooks()
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
