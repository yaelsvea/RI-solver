# Finite Rational Inattention - a small solver

This is a simple Python implementation of the finite Rational Inattention (RI) model with Shannon entropy costs, solved using the Blahut-Arimoto algorithm. 

I built it while working through the Armenter, Müller-Itten & Stangebye paper *Geometric Methods for Finite Rational Inattention* in Quantitative Economics (2024).
 
I wanted to understand the model by actually building a solver myself, rather than just reading through the theory.


## The problem
 
Finite states $\omega \in \Omega$ with prior $\mu$, finite actions $a \in A$,
payoffs $u(a,\omega)$. The agent chooses a state-contingent action distribution
$p(a\mid\omega)$ to solve
 
$$\max_{p(a\mid\omega)} \sum_\omega \mu_\omega \sum_a p(a\mid\omega)u(a,\omega) - \lambda I(\omega; a)$$
 
with $I$ the mutual information between state and action, and
$p(a) = \sum_\omega \mu_\omega p(a\mid\omega)$.
 

## Usage
```python
import numpy as np
from ri_solver import ri_solver_log, mutual_information, gross_utility

mu = np.array([0.7, 0.3])
u = np.array([[1.0, 0.0], [0.0, 1.0]])
lam = 0.5

p, p_cond, iters = ri_solver_log(mu, u, lam)
info = mutual_information(mu, p, p_cond)
gross = gross_utility(mu, u, p_cond)
value = gross - lam * info
```

## What's implemented
- The naive Blahut–Arimoto solver.
- A log-space solver, built because the naive implementation suffers from numerical overflow at very low values of λ.
- Functions to calculate mutual information and gross utility.
- A convergence criterion based on the maximum change in the unconditional probability distribution between iterations.



## What I found

Overflow. The first issue I ran into was numerical overflow in the exponential calculation when λ became very small. I estimated the failure point to be around λ ≈ 0.0014 using the float64 limit (ln(1.8 × 10^308) ≈ 709.8), and then confirmed it by checking values between 0.0012 and 0.0015. This showed that the naive implementation needed more careful numerical handling for very small λ.

Zero-probability actions. Handling zeros proved awkward in two ways. First, the entropy calculation needed a small numerical fix. In floating-point arithmetic, calculating log(0) gives an invalid value (nan), even though the mathematical limit of 0 log 0 is zero. I fixed this by explicitly treating terms with zero probability as zero in the entropy calculation. Second, the Blahut–Arimoto algorithm is structurally unable to identify exactly which actions remain in the consideration set. For example, at λ = 4.0, the algorithm returns a probability of 1.7e-10 for the safe action (as seen in the table below). The true value there is exactly zero, but the algorithm only asymptotically approaches zero and never arrives. Therefore, identifying the consideration set requires picking a cutoff, and what falls below that cutoff depends heavily on how long you run the algorithm.

Convergence and consideration sets. I found that the number of iterations changed a lot depending on λ. With a prior of (0.7, 0.3), the solver took 2 iterations at λ = 0.001, 38 at λ = 0.5, and 2,906 at λ = 50. The solver converged much more slowly when attention was expensive and the agent's behaviour concentrated on fewer actions. I don't know whether that concentration is what causes the slow convergence.
To see how this affects consideration sets, consider a three-action example with a uniform prior: two specialists paying 2 in their own state and 0 otherwise, plus a safe action paying 1.1 in both. Below λ ≈ 4–5, the agent splits between the specialists and drops the safe action entirely. Above that threshold, it takes the safe action and stops learning. This shifting of the consideration set is an economically interesting feature, but the algorithm's performance degrades sharply near the boundary as the answer you get depends on your iteration budget. 
Therefore the iteration limit can give a highly misleading answer. In the three-action setup, with a 5,000-iteration limit, λ = 5 gave p(safe) = 0.492, which looks like a reasonable result. However, after increasing the limit to 50,000, the same case gave p(safe) = 0.997. The first result was not a property of the economic model; it was simply the algorithm stopping before it had converged.

| λ | p(safe), 5k iters | p(safe), 50k iters | iters used (50k cap) |
|---|---|---|---|
| 6.0 | 1.000 | 1.000 | 7,905 |
| 5.6 | 0.9999 | 1.000 | 11,070 |
| 5.2 | 0.973 | 1.000 | 24,935 |
| 5.0 | 0.492 | 0.997 | 50,000 (hit cap) |
| 4.0 | 1.7e-10 | 1.7e-10 | 3,679 |


## Validation

The test-suite runs with pytest and consists of seven tests covering the following: 
- The symmetric 2x2 case matches a hand-derived closed-form solution.
- The naive and log-space solvers produce identical results.
- The model exhibits full learning at low λ (mutual information approaches ln 2) and no learning at high λ.
- Strictly dominated actions are correctly dropped.
- Input validation correctly raises errors for mismatched array shapes. 


## Not yet done

Comparison against the replication package. The outputs here have not been checked against the authors' published code (Zenodo 10.5281/zenodo.8316359). 

The paper's proposed algorithm. This repository implements the Blahut–Arimoto baseline, not the novel method introduced in the paper. The paper's contribution is a superior algorithm designed to bypass the boundary and convergence struggles documented in the findings above. Comparing this baseline against the proposed algorithm is the obvious uncompleted next step.

Higher-dimensional state spaces. Everything run so far is limited to two states. With three states, the belief space becomes a triangle rather than a simple interval, which is where the geometry the paper investigates actually becomes visible.