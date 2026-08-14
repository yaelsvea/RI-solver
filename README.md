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

## What's implemented

## What I found


Overflow. The first issue I ran into was numerical overflow in the exponential calculation when λ became very small. I estimated the failure point to be around λ ≈ 0.0014 using the float64 limit (ln(1.8 × 10^308) ≈ 709.8), and then confirmed it by checking values between 0.0012 and 0.0015. This showed that the naive implementation needed more careful numerical handling for very small λ.

Convergence. I also found that the number of iterations changed a lot depending on λ. With a prior of (0.7, 0.3), the solver took 2 iterations at λ = 0.001, 38 at λ = 0.5, and 2,906 at λ = 50. The solver converged much more slowly when attention was expensive and the agent's behaviour concentrated on fewer actions. I don't know whether that concentration is what causes the slow convergence.

0 log 0. The entropy calculation also needed a small numerical fix. In floating-point arithmetic, calculating log(0) gives an invalid value (nan), even though the mathematical limit of 0 log 0 is zero. I fixed this by explicitly treating terms with zero probability as zero in the entropy calculation.


## Not yet done
