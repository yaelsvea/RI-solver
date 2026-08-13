"""Solving a finite Rational Inattention problem by Blahut-Arimoto iteration."""

import numpy as np


def ri_solver(mu, u, lam, iters=5000):
    """Parameters:
    mu: array, shape (n,) - prior distribution over states, sums to 1
    u: array, shape (m, n) - utility matrix, where u[a, w] is the utility of action a in state w
    lam: float - Lagrange multiplier for the information constraint

    Returns:
    p: array, shape (m,) - unconditional choice probabilities, p(a) = sum_w p(a|w) * mu(w)
    p_cond: array, shape (m, n) - conditional choice probabilities, p(a|w), each column sums to 1
    """
    mu = np.asarray(mu, dtype=float)
    u = np.asarray(u, dtype=float)
    lam = float(lam)
    m, n = u.shape

    p = np.ones(m) / m
    scaled = u / lam  # attention vectors initialization
    print("scaled:", scaled)
    G = np.exp(scaled)
    print("G =", G)

    num = p[:, None] * G        # num = p(a) * exp(u(a, w) / lam)
    print("num =", num)

    denom = num.sum(axis=0)     # denom = sum_a p(a) * exp(u(a, w) / lam)
    print("denom =", denom)

    p_cond = num / denom        # p_cond = p(a|w) = p(a) * exp(u(a, w) / lam) / sum_a p(a) * exp(u(a, w) / lam)
    print("p_cond =", p_cond)
    print("column sums =", p_cond.sum(axis=0))

    return p, p_cond    


if __name__ == "__main__":
    mu = np.array([0.5, 0.5])  # Prior distribution over states
    u = np.array([[1.0, 0.0], [0.0, 1.0]])  # Utility matrix
    lam = 5.0  # Attention cost
    p, p_cond = ri_solver(mu, u, lam)
    print("p:", p)
    print("p_cond:", p_cond)
