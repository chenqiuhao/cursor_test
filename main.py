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

    elbo_values: List[float] = [compute_elbo(data, mean, log_std, config)]
    means: List[float] = [mean]
    stds: List[float] = [math.exp(log_std)]

    for _ in range(steps):
        current_elbo = elbo_values[-1]

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
            accepted = True

        if accepted:
            mean = candidate_mean
            log_std = candidate_log_std
            elbo_values.append(candidate_elbo)
            means.append(mean)
            stds.append(math.exp(log_std))

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


def _choose_snapshot_indices(length: int, desired: int = 5) -> List[int]:
    desired = max(2, desired)
    if length <= desired:
        return list(range(length))
    step = (length - 1) / (desired - 1)
    indices = {int(round(step * i)) for i in range(desired)}
    indices.add(length - 1)
    return sorted(indices)


def visualize_inference_steps(
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

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # 数据分布与先验
    ax_hist = axes[0, 0]
    bins = min(20, max(6, data.size // 2))
    ax_hist.hist(
        data,
        bins=bins,
        density=True,
        alpha=0.55,
        color="#4C72B0",
        edgecolor="white",
        label="观测数据",
    )
    ax_hist.plot(xs, normal_pdf(xs, prior_mean, prior_std), linestyle="--", color="#8C8C8C", label="先验 N(0, τ²)")
    ax_hist.axvline(np.mean(data), color="#555555", linestyle=":", label="样本均值")
    ax_hist.set_title("数据分布与先验")
    ax_hist.set_xlabel("μ")
    ax_hist.set_ylabel("密度")
    ax_hist.legend(loc="best")

    # 迭代中的 q(μ) 形状
    ax_q = axes[0, 1]
    snapshot_indices = _choose_snapshot_indices(len(vi_result.means))
    colors = plt.cm.Oranges(np.linspace(0.35, 0.9, len(snapshot_indices)))
    for idx, color in zip(snapshot_indices, colors):
        label = "初始化" if idx == 0 else ("收敛" if idx == len(vi_result.means) - 1 else f"第 {idx} 步")
        ax_q.plot(xs, normal_pdf(xs, vi_result.means[idx], vi_result.stds[idx]), color=color, label=label)
    ax_q.plot(xs, normal_pdf(xs, exact_mean, exact_std), color="#2CA02C", linestyle="--", linewidth=1.6, label="精确后验")
    ax_q.set_title("q(μ) 形状的演化")
    ax_q.set_xlabel("μ")
    ax_q.set_ylabel("密度")
    ax_q.legend(loc="best")

    iterations = np.arange(len(vi_result.means))

    # 均值和标准差轨迹
    ax_params = axes[1, 0]
    ax_params.plot(iterations, vi_result.means, color="#D62728", label="q(μ) 均值")
    ax_params.axhline(exact_mean, color="#2CA02C", linestyle="--", label="精确均值")
    ax_params.set_title("均值的梯度上升轨迹")
    ax_params.set_xlabel("迭代步")
    ax_params.set_ylabel("均值")
    ax_params.grid(alpha=0.3)

    ax_std = ax_params.twinx()
    ax_std.plot(iterations, vi_result.stds, color="#1F77B4", label="q(μ) 标准差")
    ax_std.axhline(exact_std, color="#FF7F0E", linestyle="--", label="精确标准差")
    ax_std.set_ylabel("标准差")

    handles, labels = ax_params.get_legend_handles_labels()
    handles2, labels2 = ax_std.get_legend_handles_labels()
    ax_params.legend(handles + handles2, labels + labels2, loc="best")

    # ELBO 曲线
    ax_elbo = axes[1, 1]
    ax_elbo.plot(np.arange(len(vi_result.elbo_values)), vi_result.elbo_values, color="#9467BD")
    ax_elbo.set_title("ELBO 收敛情况")
    ax_elbo.set_xlabel("迭代步")
    ax_elbo.set_ylabel("ELBO")
    ax_elbo.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def print_step_explanations(
    data: np.ndarray,
    config: ModelConfig,
    vi_result: VIResult,
    exact_posterior: Tuple[float, float],
) -> None:
    print("\n=== 迭代过程解读 ===")
    sample_mean = float(np.mean(data))
    exact_mean, exact_std = exact_posterior
    print(
        f"样本均值约为 {sample_mean:.3f}，精确后验为 N({exact_mean:.3f}, {exact_std:.3f}^2)。"
        " 下面挑选若干关键迭代步进行说明："
    )

    snapshot_indices = _choose_snapshot_indices(len(vi_result.means))
    for idx in snapshot_indices:
        mean = vi_result.means[idx]
        std = vi_result.stds[idx]
        elbo = vi_result.elbo_values[idx]
        grad_mean, grad_log_std = compute_gradients(data, mean, math.log(std), config)

        if idx == 0:
            stage = "初始化"
        elif idx == len(vi_result.means) - 1:
            stage = "收敛状态"
        else:
            stage = f"第 {idx} 步"

        direction_mean = "向右（增大均值）" if grad_mean > 1e-6 else ("向左（减小均值）" if grad_mean < -1e-6 else "保持均值")
        direction_std = "收缩方差" if grad_log_std < -1e-6 else ("增大方差" if grad_log_std > 1e-6 else "保持方差")

        delta_text = ""
        if idx > 0:
            delta_mean = mean - vi_result.means[idx - 1]
            delta_std = std - vi_result.stds[idx - 1]
            delta_text = f" 本步变化 Δ均值={delta_mean:+.3f}, Δ标准差={delta_std:+.3f}。"

        print(
            f"[{stage}] ELBO = {elbo:.3f}，当前 q(μ) = N({mean:.3f}, {std:.3f}^2)。"
            f" 梯度指向：{direction_mean}，{direction_std}.{delta_text}"
        )

    diff_mean = abs(vi_result.mean - exact_mean)
    diff_std = abs(vi_result.std - exact_std)
    print(
        f"最终误差：均值差 {diff_mean:.3e}，标准差差 {diff_std:.3e}。"
        " 这说明变分分布已很好地贴近精确后验。"
    )


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

    print_step_explanations(data, config, vi_result, exact_posterior)
    visualize_inference_steps(data, config, vi_result, exact_posterior)


if __name__ == "__main__":
    main()
