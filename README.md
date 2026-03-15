# MoE Congestion Game Analysis

**Is Mixture-of-Experts routing a congestion game? Measuring the Price of Anarchy and testing for Braess's Paradox in neural expert routing.**

## The Question

MoE models route tokens to experts via a learned router. This routing has a formal structure identical to a **congestion game** (Rosenthal 1973): tokens are players, experts are resources, and the router minimizes each token's local cost. But local optimization ≠ global optimization. How much performance is lost to this selfish routing?

Nobody has asked this question formally.

## The Mapping

| Congestion Game | MoE Routing |
|---|---|
| Players | Tokens |
| Resources | Experts (FFN) |
| Strategy | Top-K expert selection |
| Cost function c(x) | Quality degradation under load |
| Nash Equilibrium | Converged routing pattern |
| Social Optimum | Routing minimizing global loss |
| **Price of Anarchy** | **ppl(equilibrium) / ppl(optimal)** |
| Pigou Tax | Load balancing loss |
| **Braess Paradox** | **Removing expert improves ppl** |

## Phases

**Phase A — Equilibrium Observation** (~20 min)
Hook all MoE routers, pass diverse prompts, compute:
- Gini coefficient per layer (load imbalance)
- Normalized entropy (uniformity of routing)
- Max/min load ratio

**Phase B — Price of Anarchy** (~2h)
Compare routing regimes:
1. Equilibrium (normal router)
2. Each expert disabled (greedy search for oracle)
3. Uniform round-robin

PoA = ppl(equilibrium) / ppl(best alternative)

**Phase C — Braess Paradox** (~2-4h)
For each layer × each expert: disable expert, measure ppl.
If ppl *decreases* → Braess paradox confirmed.

**Phase D — Specialization Matrix** (~30 min)
Build P(expert | category) for 7 categories.
Compute mutual information I(expert; category).
Heatmap visualization.

## Usage

```bash
pip install torch transformers datasets matplotlib seaborn accelerate --break-system-packages

# Quick test (Phase A only)
python moe_congestion_game.py --phase A --model mistralai/Mixtral-8x7B-Instruct-v0.1

# Full analysis
python moe_congestion_game.py --phase ABCD --model mistralai/Mixtral-8x7B-Instruct-v0.1

# Or use the runner
bash run.sh mistralai/Mixtral-8x7B-Instruct-v0.1 ABCD ./results
```

**GPU requirements:**
- Mixtral-8x7B: 2×A100 80GB (bf16, uses `device_map="auto"`)
- Alternative: any HuggingFace MoE model

## What Would Be Publishable

- **Braess Paradox confirmed** → first empirical demonstration in neural networks
- **PoA > 1.05** → measurable inefficiency in MoE routing, with game-theoretic characterization
- **Expert specialization + congestion** → formal connection between MoE load balancing and algorithmic game theory

## License

MIT
