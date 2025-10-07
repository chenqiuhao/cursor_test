import math

from typing import Iterable

from main import (
    ModelConfig,
    closed_form_posterior,
    compute_elbo,
    compute_gradients,
    generate_data,
    run_variational_inference,
)


def make_dataset():
    config = ModelConfig(obs_std=1.0, prior_var=25.0)
    data = generate_data(true_mean=1.5, size=40, config=config, seed=123)
    return data, config


def manual_elbo(data: Iterable[float], mean: float, log_std: float, config: ModelConfig) -> float:
    obs_var = config.obs_std**2
    prior_var = config.prior_var
    std_sq = math.exp(2.0 * log_std)
    n = data.size

    sum_sq = sum((value - mean) ** 2 for value in data)

    expected_log_likelihood = (
        -0.5 * n * math.log(2.0 * math.pi * obs_var)
        -0.5 * (sum_sq + n * std_sq) / obs_var
    )
    expected_log_prior = (
        -0.5 * math.log(2.0 * math.pi * prior_var)
        -0.5 * (std_sq + mean**2) / prior_var
    )
    entropy = 0.5 * (1.0 + math.log(2.0 * math.pi)) + log_std

    return expected_log_likelihood + expected_log_prior + entropy


def test_compute_elbo_matches_manual_derivation():
    data, config = make_dataset()
    mean = 0.8
    log_std = math.log(0.9)

    analytic = compute_elbo(data, mean, log_std, config)
    manual = manual_elbo(data, mean, log_std, config)

    assert math.isfinite(analytic)
    assert math.isfinite(manual)
    assert math.isclose(analytic, manual, rel_tol=1e-12, abs_tol=1e-12)


def test_compute_gradients_zero_at_closed_form_solution():
    data, config = make_dataset()
    post_mean, post_std = closed_form_posterior(data, config)

    grad_mean, grad_log_std = compute_gradients(data, post_mean, math.log(post_std), config)

    assert math.isclose(grad_mean, 0.0, abs_tol=1e-8)
    assert math.isclose(grad_log_std, 0.0, abs_tol=1e-8)


def test_variational_inference_converges_to_conjugate_posterior():
    data, config = make_dataset()
    result = run_variational_inference(
        data,
        config,
        lr=0.05,
        steps=500,
        init_mean=-1.0,
        init_log_std=math.log(2.5),
    )

    post_mean, post_std = closed_form_posterior(data, config)

    assert math.isclose(result.mean, post_mean, rel_tol=1e-2, abs_tol=1e-2)
    assert math.isclose(result.std, post_std, rel_tol=1e-2, abs_tol=1e-2)

