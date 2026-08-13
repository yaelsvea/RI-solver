"""Solving a finite Rational Inattention problem by Blahut-Arimoto iteration."""

import numpy as np


def ri_solver(mu, u, lam, iters=5000):
    """Parameters:
    mu: array, shape (n,) - prior distribution over states, sums to 1
    u: array, shape (m, n) - utility matrix, where u[a, w] is the utility of action a in state w
    lam: float - attention cost, the price paid per nat of information
    iters: int - number of iterations for the Blahut-Arimoto algorithm

    Returns:
    p: array, shape (m,) - unconditional choice probabilities, p(a) = sum_w p(a|w) * mu(w)
    p_cond: array, shape (m, n) - conditional choice probabilities, p(a|w), each column sums to 1
    """
    mu = np.asarray(mu, dtype=float)
    u = np.asarray(u, dtype=float)
    lam = float(lam)
    m, n = u.shape

    if mu.size != n:
        raise ValueError(f"u has {n} states but mu has {mu.size}")

    p = np.ones(m) / m
    G = np.exp(u / lam)

    for _ in range(iters):
        num = (p[:, None] * G)  # num = p(a) * exp(u(a, w) / lam), p[:, None] makes p a column so it broadcasts down rows
        denom = num.sum(axis=0)  # denom = sum_a p(a) * exp(u(a, w) / lam), actions are rows: axis 0 normalises within each state
        p_cond = num / denom
        p = p_cond @ mu  # p(a) = sum_w p(a|w) * mu(w)

    return p, p_cond


if __name__ == "__main__":
    mu = np.array([0.5, 0.5])  # Prior distribution over states
    u = np.array([[1.0, 0.0], [0.0, 1.0]])  # Utility matrix
    lam = 0.5  # Attention cost
    p, p_cond = ri_solver(mu, u, lam)
    print("p:", p)
    print("p_cond:", p_cond)
