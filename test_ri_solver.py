import numpy as np
import pytest

from ri_solver import mutual_information, ri_solver, ri_solver_log


def test_symmetric_closed_form():
    mu = np.array([0.5, 0.5])
    u = np.array([[1.0, 0.0], [0.0, 1.0]])
    lam = 0.5
    _p, p_cond, _ = ri_solver_log(mu, u, lam)
    expected = 1.0 / (1.0 + np.exp(-1.0 / lam))
    assert np.isclose(p_cond[0, 0], expected)


def test_columns_sum_to_one():
    mu = np.array([0.7, 0.3])
    u = np.array([[1.0, 0.0], [0.0, 1.0]])
    _p, p_cond, _ = ri_solver_log(mu, u, 0.5)
    assert np.allclose(p_cond.sum(axis=0), 1.0)


def test_solvers_agree():
    mu = np.array([0.7, 0.3])
    u = np.array([[1.0, 0.0], [0.0, 1.0]])
    lam = 0.5
    p_naive, p_cond_naive = ri_solver(mu, u, lam)
    p_log, p_cond_log, _ = ri_solver_log(mu, u, lam)
    assert np.allclose(p_naive, p_log)
    assert np.allclose(p_cond_naive, p_cond_log)


def test_full_learning_when_attention_is_cheap():
    mu = np.array([0.5, 0.5])
    u = np.array([[1.0, 0.0], [0.0, 1.0]])
    p, p_cond, _ = ri_solver_log(mu, u, 0.001)
    I = mutual_information(mu, p, p_cond)
    assert np.isclose(I, np.log(2), atol=1e-6)


def test_no_learning_when_attention_is_expensive():
    mu = np.array([0.5, 0.5])
    u = np.array([[1.0, 0.0], [0.0, 1.0]])
    p, p_cond, _ = ri_solver_log(mu, u, 50.0)
    I = mutual_information(mu, p, p_cond)
    assert I < 1e-3

def test_dominated_action_is_dropped():
    mu = np.array([0.5, 0.5])
    u = np.array([[2.0, 0.0], [0.0, 2.0], [-5.0, -5.0]])
    p, _p_cond, _ = ri_solver_log(mu, u, 0.5)
    assert p[2] < 1e-9

def test_mismatched_shapes_raise():
    mu = np.array([0.5, 0.3, 0.2])
    u = np.array([[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError):
        ri_solver_log(mu, u, 0.5)