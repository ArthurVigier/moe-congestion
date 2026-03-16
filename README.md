# MoE Congestion Game

**Game-theoretic analysis of expert routing in Mixture-of-Experts language models.**

MoE routing has the same formal structure as a congestion game (Rosenthal 1973): tokens are players, experts are resources, the router minimizes local cost. This repository tests whether classical game-theoretic pathologies — load imbalance, the Price of Anarchy, and Braess's Paradox — appear in trained MoE models.

---

## Background

In a congestion game, selfish agents choose resources to minimize their individual cost. The resulting Nash equilibrium can be worse than the social optimum — the gap is the Price of Anarchy (Roughgarden & Tardos 2002). A more counterintuitive result is Braess's Paradox: adding a resource (road, link) can increase total cost, because selfish agents over-converge on it.

MoE models route each token to a subset of experts via a learned router. Load balancing losses act as Pigou taxes to distribute tokens across experts. The question: does this work? Or do trained routers exhibit the same pathologies as selfish routing in networks?

| Congestion Game | MoE Routing |
|---|---|
| Players | Tokens |
| Resources | Experts (FFN) |
| Strategy | Top-K selection |
| Cost c(x) | Quality degradation under load |
| Nash Equilibrium | Converged routing pattern |
| Social Optimum | Routing minimizing global loss |
| Price of Anarchy | ppl(equilibrium) / ppl(optimal) |
| Pigou Tax | Load balancing auxiliary loss |
| Braess Paradox | Removing expert improves ppl |

## Models

- **Mixtral-8x7B-Instruct** (8 experts, top-2) — baseline with low strategic complexity
- **ERNIE-4.5-21B-A3B-PT** (64 experts, top-6, 2 shared) — high strategic complexity, C(64,6) ≈ 74M routing combinations per token

## Phase A — Load Distribution

We hook all MoE routers and pass 349 prompts across 7 categories (science, commonsense, knowledge, math, factual, reasoning, code).

| Model | Experts | Mean Gini | Dead experts | Max load ratio |
|---|---|---|---|---|
| Mixtral-8x7B | 8 | 0.085 | 0 | 2.6:1 |
| ERNIE-4.5-21B | 64 | 0.386 | Yes (L24, L27) | 1252:1 |

Mixtral's routing is near-uniform. ERNIE's is not: some experts receive 2500 tokens while others receive zero. Imbalance increases with depth.

Per-layer detail (ERNIE):

| Layer | Gini | Max Load | Min Load | Ratio |
|---|---|---|---|---|
| L3 | 0.33 | 1433 | 89 | 16:1 |
| L12 | 0.40 | 1625 | 33 | 49:1 |
| L15 | 0.42 | 2504 | 2 | 1252:1 |
| L24 | 0.41 | 1315 | 0 | ∞ |
| L27 | 0.50 | 1575 | 0 | ∞ |

## Phase B — Price of Anarchy

We intervene on the router at layer 14 of ERNIE, comparing three regimes:

| Routing regime | Mean perplexity |
|---|---|
| Learned router | 49.65 |
| Best single-expert disabled | 57.93 (+16.7%) |
| Uniform random | 65.13 (+31.2%) |

The learned router is 31% better than random and outperforms every single-ablation variant. At this layer, the equilibrium is quasi-optimal.

## Phase C — Braess's Paradox

For each of 7 sampled layers × 64 experts = 448 configurations, we disable one expert (set its router logits to -∞) and re-measure perplexity on 20 prompts.

**115 out of 448 configurations (25.7%) improve perplexity when the expert is removed.**

| Layer | Braess instances | Best ablation |
|---|---|---|
| L1 | 50/64 (78%) | E17: -4.85% ppl |
| L5 | 33/64 (52%) | E23: -3.47% |
| L9 | 6/64 | E18: -4.77% |
| L13 | 5/64 | E49: -2.24% |
| L17 | 3/64 | E37: -1.20% |
| L21 | 13/64 | E5: -2.62% |
| L25 | 5/64 | E43: -0.81% |

The effect is concentrated in early layers. At Layer 1, removing almost any expert improves the model. At Layer 25, only 8% of ablations help. The depth gradient is monotonic except for a bump at L21.

Top five instances:

```
Layer 1,  Expert 17:  ppl 52.81  (-4.85%)
Layer 9,  Expert 18:  ppl 52.86  (-4.77%)
Layer 5,  Expert 23:  ppl 53.58  (-3.47%)
Layer 9,  Expert 16:  ppl 53.61  (-3.41%)
Layer 1,  Expert 47:  ppl 53.77  (-3.12%)
```

## Phase D — Expert Specialization

We compute I(expert; category) — the mutual information between expert selection and prompt category — to test whether congestion is driven by thematic specialization.

| Layer | NMI |
|---|---|
| L1 | 0.029 |
| L5 | 0.017 |
| L9 | 0.026 |
| L13 | 0.057 |
| L17 | 0.046 |
| L21 | 0.044 |
| L25 | 0.046 |
| L27 | 0.017 |

NMI is low everywhere (< 0.06). Experts do not specialize by category. Popular experts attract tokens from all categories indiscriminately — they are attractors, not specialists.

This explains the Braess result: early-layer experts that hurt performance have no specialty. The router converged on them during training without them learning useful computation. Removing them forces redistribution to other experts that happen to compute better.

## Interpretation

The Braess depth gradient (L1: 78% → L25: 8%) suggests that load balancing losses work in deep layers but fail in early layers. Early-layer routing has not converged to a useful equilibrium.

This is consistent with MiniMax's report that early routing decisions degrade downstream reasoning in their hybrid attention models: suboptimal early routing compounds through the network.

The low NMI rules out the hypothesis that congestion is a byproduct of specialization. The mechanism is purely structural: the router has learned routing habits that are locally stable (Nash equilibrium) but globally suboptimal.

## Implications

**Architecture**: early MoE layers may not need all their experts. Replacing them with dense FFN or pruning Braess experts could improve both performance and efficiency.

**Load balancing**: current auxiliary losses are insufficient for early layers. Mechanism design tools from algorithmic game theory (VCG auctions, congestion pricing) may provide alternatives.

**Pruning**: expert importance is non-monotonic with load. A popular expert can be harmful (Braess), an unpopular one essential. Pruning by load frequency is not reliable.

## Reproducing

```bash
pip install torch transformers datasets accelerate matplotlib seaborn --break-system-packages

# Phase A — Load distribution (~30 min, 1×A100 80GB)
python moe_congestion_game.py --phase A --model baidu/ERNIE-4.5-21B-A3B-PT --n_per_category 50

# Phase B — Price of Anarchy (~2h)
python moe_congestion_game.py --phase B --model baidu/ERNIE-4.5-21B-A3B-PT --n_poa 30

# Phase C — Braess Paradox (~3h, standalone script)
python phase_c_braess.py --model baidu/ERNIE-4.5-21B-A3B-PT --n_prompts 20

# Phase A+D — Specialization (~40 min, must run together)
python moe_congestion_game.py --phase AD --model baidu/ERNIE-4.5-21B-A3B-PT --n_per_category 50
```

## Files

```
moe_congestion_game.py    # Phases A, B, D
phase_c_braess.py         # Phase C standalone
ernie_phase_a.json        # Load distribution results
ernie_phase_b.json        # Price of Anarchy results
ernie_phase_c.json        # Braess Paradox results (115 instances)
ernie_phase_ad.json       # Specialization matrix + NMI
README.md
```

## Related Work

- Rosenthal (1973). A class of games possessing pure-strategy Nash equilibria. *Int. J. Game Theory*.
- Roughgarden & Tardos (2002). How bad is selfish routing? *JACM*.
- Braess (1968). Über ein Paradoxon aus der Verkehrsplanung. *Unternehmensforschung*.
- Fedus et al. (2022). Switch Transformers. *JMLR*. — introduced auxiliary load balancing loss.
- Zhou et al. (2022). Expert-choice routing. *NeurIPS*. — experts select tokens instead of vice versa.
- DeepSeek-V3 (2024). — gradient-free bias load balancing.
- MiniMax (2025). Why did M2 end up as a full attention model? — on early routing failures.

## License

MIT
