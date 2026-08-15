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


def ri_solver_log(mu, u, lam, iters=50000, tol=1e-12):
    """Same as ri_solver, but in log space so small lam does not overflow."""
    mu = np.asarray(mu, dtype=float)
    u = np.asarray(u, dtype=float)
    lam = float(lam)
    m, n = u.shape

    if mu.size != n:
        raise ValueError(f"u has {n} states but mu has {mu.size}")

    log_p = np.full(m, -np.log(m))  # log of 1/m
    g = u / lam

    for it in range(1, iters + 1):
        score = log_p[:, None] + g
        log_denom = logsumexp(score, axis=0)
        log_p_cond = score - log_denom
        log_p_new = logsumexp(log_p_cond + np.log(mu), axis=1)

        shift = np.max(np.abs(np.exp(log_p_new) - np.exp(log_p)))
        log_p = log_p_new
        if shift < tol:
            break

    return np.exp(log_p), np.exp(log_p_cond), it


def mutual_information(mu, p, p_cond):
    """I(w; a) in nats. Zero if behaviour is state-independent."""
    joint = p_cond * mu[None, :]     # P(a, w)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = p_cond / p[:, None]
        terms = np.where(joint > 0, joint * np.log(ratio), 0.0)
    return float(np.sum(terms))


def gross_utility(mu, u, p_cond):
    """Expected payoff before subtracting attention costs."""
    joint = p_cond * mu[None, :]
    return float(np.sum(joint * u))


if __name__ == "__main__":
    mu = np.array([0.7, 0.3])  # Prior distribution over states
    u = np.array([[1.0, 0.0], [0.0, 1.0]])  # Utility matrix
    lam = 0.5  # Attention cost
    p, p_cond = ri_solver(mu, u, lam)
    print("p:", p)
    print("p_cond:", p_cond)
    p_log, pc_log, iters_used = ri_solver_log(mu, u, lam)
    print("log version p:", p_log)
    print("log version p_cond:", pc_log)
    print("log version iters used:", iters_used)
    for lam_test in [0.5, 0.1, 0.05, 0.01, 0.005, 0.002, 0.0018, 0.0015, 0.0012,0.001]:
        naive = ri_solver(mu, u, lam_test)[1][0, 0]
        log = ri_solver_log(mu, u, lam_test)[1][0, 0]
        print(f"lam={lam_test:<8} naive={naive:<12.8f} log={log:.8f}")
    for lam_test in [0.001, 0.5, 50.0]:
        p_t, pc_t, it_t = ri_solver_log(mu, u, lam_test)
        I = mutual_information(mu, p_t, pc_t)
        gross = gross_utility(mu, u, pc_t)
        value = gross - lam_test * I
        print(f"lam={lam_test:<8} I={I:.6f}  gross={gross:.6f}  value={value:.6f} iters={it_t}")

    u3 = np.array([[2.0, 0.0], [0.0, 2.0], [1.1, 1.1]])
    mu3 = np.array([0.5, 0.5])
    print(f"\n{'lam':>7} {'p(spec1)':>10} {'p(spec2)':>10} {'p(safe)':>12} {'iters':>7}")
    for lam_test in [10.0, 6.0, 5.8, 5.6, 5.4, 5.2, 5.0, 4.0, 2.0, 1.0, 0.5, 0.1]:
        p_t, pc_t, it_t = ri_solver_log(mu3, u3, lam_test)
        print(f"{lam_test:7} {p_t[0]:10.6f} {p_t[1]:10.6f} {p_t[2]:12.3e} {it_t:7d}")
