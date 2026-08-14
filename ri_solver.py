"""Solving a finite Rational Inattention problem by Blahut-Arimoto iteration."""

import numpy as np
from scipy.special import logsumexp


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


def ri_solver_log(mu, u, lam, iters=5000):
    """Same as ri_solver, but in log space so small lam does not overflow."""
    mu = np.asarray(mu, dtype=float)
    u = np.asarray(u, dtype=float)
    lam = float(lam)
    m, n = u.shape

    if mu.size != n:
        raise ValueError(f"u has {n} states but mu has {mu.size}")

    log_p = np.full(m, -np.log(m))  # log of 1/m
    g = u / lam

    for _ in range(iters):
        score = log_p[:, None] + g
        log_denom = logsumexp(score, axis=0)
        log_p_cond = score - log_denom
        log_p = logsumexp(log_p_cond + np.log(mu), axis=1)

    return np.exp(log_p), np.exp(log_p_cond)

def mutual_information(mu, p, p_cond):
    """I(w; a) in nats. Zero if behaviour is state-independent."""
    joint = p_cond * mu[None, :]     # P(a, w)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = p_cond / p[:, None]
        terms = np.where(joint > 0, joint * np.log(ratio), 0.0)
    return float(np.sum(terms))

if __name__ == "__main__":
    mu = np.array([0.5, 0.5])  # Prior distribution over states
    u = np.array([[1.0, 0.0], [0.0, 1.0]])  # Utility matrix
    lam = 0.5  # Attention cost
    p, p_cond = ri_solver(mu, u, lam)
    print("p:", p)
    print("p_cond:", p_cond)
    p_log, pc_log = ri_solver_log(mu, u, lam)
    print("log version p:", p_log)
    print("log version p_cond:", pc_log)
    for lam_test in [0.5, 0.1, 0.05, 0.01, 0.005, 0.002, 0.0018, 0.0015, 0.0012,0.001]:
        naive = ri_solver(mu, u, lam_test)[1][0, 0]
        log = ri_solver_log(mu, u, lam_test)[1][0, 0]
        print(f"lam={lam_test:<8} naive={naive:<12.8f} log={log:.8f}")
    for lam_test in [0.001, 0.5, 50.0]:
        p_t, pc_t = ri_solver_log(mu, u, lam_test)
        print(f"lam={lam_test:<8} I={mutual_information(mu, p_t, pc_t):.6f}")

