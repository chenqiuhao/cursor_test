from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class ModelConfig:
    """Configuration for the conjugate Gaussian model."""

    obs_std: float = 1.0  # Known observation noise standard deviation (σ)
    prior_var: float = 25.0  # Prior variance on the latent mean (τ^2)


@dataclass
class VIResult:
    """Container for the variational inference run."""

    mean: float
    std: float
    elbo_values: List[float]
    means: List[float]
    stds: List[float]


def generate_data(true_mean: float, size: int, config: ModelConfig, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(loc=true_mean, scale=config.obs_std, size=size)


def compute_elbo(data: np.ndarray, mean: float, log_std: float, config: ModelConfig) -> float:
    """Analytical ELBO for q(μ) = N(mean, std^2) against the conjugate model."""
    obs_var = config.obs_std ** 2
    prior_var = config.prior_var
    std_sq = math.exp(2.0 * log_std)  # Note: std = exp(log_std)
    n = data.size

    # Expected log-likelihood term E_q[log p(y | μ)]
    centered_sq = np.square(data - mean)
    expected_ll = -0.5 * n * math.log(2.0 * math.pi * obs_var)
    expected_ll -= 0.5 * (np.sum(centered_sq) + n * std_sq) / obs_var

    # Expected log prior term E_q[log p(μ)]
    expected_log_prior = -0.5 * math.log(2.0 * math.pi * prior_var)
    expected_log_prior -= 0.5 * (std_sq + mean**2) / prior_var

    # Entropy of q(μ) contributes as -E_q[log q(μ)]
    entropy = 0.5 * (1.0 + math.log(2.0 * math.pi)) + log_std

    return expected_ll + expected_log_prior + entropy


def compute_gradients(data: np.ndarray, mean: float, log_std: float, config: ModelConfig) -> Tuple[float, float]:
    """Closed-form gradients of the ELBO with respect to mean and log_std."""
    obs_var = config.obs_std ** 2
    prior_var = config.prior_var
    std_sq = math.exp(2.0 * log_std)
    n = data.size

    grad_mean = -(np.sum(mean - data) / obs_var) - mean / prior_var
    grad_log_std = -(n * std_sq / obs_var) - (std_sq / prior_var) + 1.0
    return grad_mean, grad_log_std


def run_variational_inference(
    data: np.ndarray,
    config: ModelConfig,
    lr: float = 0.02,
    steps: int = 600,
    init_mean: float = 0.0,
    init_log_std: float = math.log(1.0),
    backtracking: int = 8,
    shrink: float = 0.5,
    min_step: float = 1e-6,
) -> VIResult:
    mean = init_mean
    log_std = init_log_std

    elbo_values: List[float] = []
    means: List[float] = []
    stds: List[float] = []

    for _ in range(steps):
        current_elbo = compute_elbo(data, mean, log_std, config)
        elbo_values.append(current_elbo)
        means.append(mean)
        stds.append(math.exp(log_std))

        grad_mean, grad_log_std = compute_gradients(data, mean, log_std, config)

        step_scale = 1.0
        accepted = False
        candidate_mean = mean
        candidate_log_std = log_std
        candidate_elbo = current_elbo

        for _ in range(backtracking):
            step = lr * step_scale
            trial_mean = mean + step * grad_mean
            trial_log_std = log_std + step * grad_log_std
            trial_elbo = compute_elbo(data, trial_mean, trial_log_std, config)

            if math.isfinite(trial_elbo) and trial_elbo >= current_elbo:
                candidate_mean = trial_mean
                candidate_log_std = trial_log_std
                candidate_elbo = trial_elbo
                accepted = True
                break

            step_scale *= shrink
            if step_scale < min_step:
                break

        if not accepted:
            # If no improving step was found, fall back to the best finite trial we computed.
            candidate_mean = mean + lr * step_scale * grad_mean
            candidate_log_std = log_std + lr * step_scale * grad_log_std
            candidate_elbo = compute_elbo(data, candidate_mean, candidate_log_std, config)
            if not math.isfinite(candidate_elbo) or candidate_elbo < current_elbo:
                # Keep the previous iterate to avoid divergence and exit early.
                break

        mean = candidate_mean
        log_std = candidate_log_std
        elbo_values[-1] = candidate_elbo

    return VIResult(mean=mean, std=math.exp(log_std), elbo_values=elbo_values, means=means, stds=stds)


def closed_form_posterior(data: np.ndarray, config: ModelConfig) -> Tuple[float, float]:
    """Exact conjugate posterior for comparison."""
    obs_var = config.obs_std ** 2
    prior_var = config.prior_var
    n = data.size

    precision = n / obs_var + 1.0 / prior_var
    posterior_var = 1.0 / precision
    posterior_mean = posterior_var * (np.sum(data) / obs_var)
    return posterior_mean, math.sqrt(posterior_var)


def describe_math(data: np.ndarray, config: ModelConfig) -> None:
    """Prints an explanation of the variational objective and its components."""
    print("\n=== 模型设定 ===")
    print("观测数据: y_i ~ Normal(μ, σ^2)，其中 σ 已知。")
    print("先验: μ ~ Normal(0, τ^2)。")
    print(f"当前示例使用 σ = {config.obs_std:.1f}, τ^2 = {config.prior_var:.1f}, 样本量 N = {data.size}.")

    print("\n=== 变分族 ===")
    print("使用均值为 m、标准差为 s 的单变量正态分布 q(μ) = Normal(m, s^2)。")

    print("\n=== 证据下界 (ELBO) ===")
    print("ELBO(m, s) = E_q[log p(y | μ)] + E_q[log p(μ)] - E_q[log q(μ)]")
    print("由于模型是共轭的，每一项可以解析计算：")
    print("1) E_q[log p(y | μ)] = -½ N log(2πσ^2) - ½σ^{-2} Σ [(y_i - m)^2 + s^2]")
    print("2) E_q[log p(μ)] = -½ log(2πτ^2) - ½τ^{-2} (s^2 + m^2)")
    print("3) -E_q[log q(μ)] = ½ log(2π e s^2)")
    print("对 m 和 log s 求导即可得到梯度，从而使用梯度上升最大化 ELBO。")


def plot_results(
    data: np.ndarray,
    config: ModelConfig,
    vi_result: VIResult,
    exact_posterior: Tuple[float, float],
) -> None:
    prior_mean = 0.0
    prior_std = math.sqrt(config.prior_var)
    exact_mean, exact_std = exact_posterior

    xs = np.linspace(min(data) - 3, max(data) + 3, 400)

    def normal_pdf(x, mean, std):
        coeff = 1.0 / (std * math.sqrt(2.0 * math.pi))
        return coeff * np.exp(-0.5 * ((x - mean) / std) ** 2)

    plt.figure(figsize=(10, 8))

    # Posterior densities panel
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(xs, normal_pdf(xs, prior_mean, prior_std), label="Prior", linestyle="--")
    ax1.plot(xs, normal_pdf(xs, vi_result.mean, vi_result.std), label="Variational q(μ)")
    ax1.plot(xs, normal_pdf(xs, exact_mean, exact_std), label="Exact posterior")
    ax1.axvline(np.mean(data), color="gray", alpha=0.5, label="Sample mean")
    ax1.set_title("Variational Inference vs. Exact Posterior")
    ax1.set_xlabel("μ")
    ax1.set_ylabel("Density")
    ax1.legend()

    # ELBO convergence panel
    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(vi_result.elbo_values)
    ax2.set_title("ELBO during gradient ascent")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("ELBO")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def main():
    config = ModelConfig(obs_std=1.0, prior_var=25.0)
    data = generate_data(true_mean=2.5, size=30, config=config, seed=2024)

    describe_math(data, config)

    vi_result = run_variational_inference(data, config, lr=0.02, steps=800)
    exact_posterior = closed_form_posterior(data, config)

    print("\n=== 对比结果 ===")
    print(f"变分后验: 均值 = {vi_result.mean:.3f}, 标准差 = {vi_result.std:.3f}")
    print(f"精确后验: 均值 = {exact_posterior[0]:.3f}, 标准差 = {exact_posterior[1]:.3f}")
    print(f"样本均值: {np.mean(data):.3f}")

    plot_results(data, config, vi_result, exact_posterior)


if __name__ == "__main__":
    main()
