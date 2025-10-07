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
    elbo_terms: List[Tuple[float, float, float]]


def generate_data(true_mean: float, size: int, config: ModelConfig, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(loc=true_mean, scale=config.obs_std, size=size)


def compute_elbo_terms(
    data: np.ndarray, mean: float, log_std: float, config: ModelConfig
) -> Tuple[float, float, float]:
    """Return the three additive pieces of the ELBO."""
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

    return expected_ll, expected_log_prior, entropy


def compute_elbo(data: np.ndarray, mean: float, log_std: float, config: ModelConfig) -> float:
    """Analytical ELBO for q(μ) = N(mean, std^2) against the conjugate model."""
    expected_ll, expected_log_prior, entropy = compute_elbo_terms(data, mean, log_std, config)
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
    elbo_terms: List[Tuple[float, float, float]] = [
        compute_elbo_terms(data, mean, log_std, config)
    ]

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
            elbo_terms.append(compute_elbo_terms(data, mean, log_std, config))

    return VIResult(
        mean=mean,
        std=math.exp(log_std),
        elbo_values=elbo_values,
        means=means,
        stds=stds,
        elbo_terms=elbo_terms,
    )


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
    """Print a student-friendly walkthrough of every equation we will use."""

    print("\n=== Step 1 · Specify the probabilistic model ===")
    print("• Observations: each data point y_i is drawn from Normal(μ, σ²) with known noise σ.")
    print("  This says the data are concentrated near the unknown mean μ, but are blurred by σ.")
    print("• Prior belief: before seeing data we think μ ~ Normal(0, τ²).")
    print(
        "  The prior variance τ² measures how uncertain we are about μ at the start."
        f"  In our example σ = {config.obs_std:.1f}, τ² = {config.prior_var:.1f}, and we observe N = {data.size} points."
    )

    print("\n=== Step 2 · Choose a variational family ===")
    print(
        "• We approximate the true posterior with q(μ) = Normal(m, s²)."
        "  The parameters m and s describe where q is centred and how wide it is."
    )
    print(
        "• By optimising m and s we try to make q(μ) imitate the exact posterior as closely as possible."
    )

    print("\n=== Step 3 · Build the Evidence Lower Bound (ELBO) ===")
    print("The ELBO is our objective: maximise it to make q(μ) good.")
    print("It has three pieces. Each one has an intuitive story:")
    print("  1. Data fit term   E_q[log p(y | μ)]        → rewards explaining the observations well.")
    print("  2. Prior term      E_q[log p(μ)]            → keeps us close to what the prior expected.")
    print("  3. Entropy term   −E_q[log q(μ)]            → prefers distributions that stay expressive.")
    print(
        "Because the model and q are both Gaussian, we can evaluate these expectations analytically,"
        " giving closed-form formulas for gradient ascent."
    )
    print(
        "Taking derivatives with respect to m and log s turns the problem into a smooth optimisation"
        " task that we can solve with gradient ascent."
    )


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

    # Data histogram and prior
    ax_hist = axes[0, 0]
    bins = min(20, max(6, data.size // 2))
    ax_hist.hist(
        data,
        bins=bins,
        density=True,
        alpha=0.55,
        color="#4C72B0",
        edgecolor="white",
        label="Observed data",
    )
    ax_hist.plot(xs, normal_pdf(xs, prior_mean, prior_std), linestyle="--", color="#8C8C8C", label="Prior N(0, τ²)")
    ax_hist.axvline(np.mean(data), color="#555555", linestyle=":", label="Sample mean")
    ax_hist.set_title("Observed data vs. prior")
    ax_hist.set_xlabel("μ")
    ax_hist.set_ylabel("Density")
    ax_hist.legend(loc="best")

    # Evolution of q(μ)
    ax_q = axes[0, 1]
    snapshot_indices = _choose_snapshot_indices(len(vi_result.means))
    colors = plt.cm.Oranges(np.linspace(0.35, 0.9, len(snapshot_indices)))
    for idx, color in zip(snapshot_indices, colors):
        label = "Initial" if idx == 0 else ("Converged" if idx == len(vi_result.means) - 1 else f"Step {idx}")
        ax_q.plot(xs, normal_pdf(xs, vi_result.means[idx], vi_result.stds[idx]), color=color, label=label)
    ax_q.plot(xs, normal_pdf(xs, exact_mean, exact_std), color="#2CA02C", linestyle="--", linewidth=1.6, label="Exact posterior")
    ax_q.set_title("Evolution of q(μ)")
    ax_q.set_xlabel("μ")
    ax_q.set_ylabel("Density")
    ax_q.legend(loc="best")

    iterations = np.arange(len(vi_result.means))

    # Parameter trajectories
    ax_params = axes[1, 0]
    ax_params.plot(iterations, vi_result.means, color="#D62728", label="q(μ) mean")
    ax_params.axhline(exact_mean, color="#2CA02C", linestyle="--", label="Exact mean")
    ax_params.set_title("Gradient-ascent trajectory of the mean")
    ax_params.set_xlabel("Iteration")
    ax_params.set_ylabel("Mean")
    ax_params.grid(alpha=0.3)

    ax_std = ax_params.twinx()
    ax_std.plot(iterations, vi_result.stds, color="#1F77B4", label="q(μ) std")
    ax_std.axhline(exact_std, color="#FF7F0E", linestyle="--", label="Exact std")
    ax_std.set_ylabel("Std dev")

    handles, labels = ax_params.get_legend_handles_labels()
    handles2, labels2 = ax_std.get_legend_handles_labels()
    ax_params.legend(handles + handles2, labels + labels2, loc="best")

    # ELBO curve
    ax_elbo = axes[1, 1]
    ax_elbo.plot(np.arange(len(vi_result.elbo_values)), vi_result.elbo_values, color="#9467BD")
    ax_elbo.set_title("ELBO convergence")
    ax_elbo.set_xlabel("Iteration")
    ax_elbo.set_ylabel("ELBO")
    ax_elbo.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def visualize_elbo_decomposition(vi_result: VIResult) -> None:
    """Plot how each ELBO component evolves over iterations."""

    iterations = np.arange(len(vi_result.elbo_terms))
    expected_ll = [term[0] for term in vi_result.elbo_terms]
    expected_prior = [term[1] for term in vi_result.elbo_terms]
    entropy = [term[2] for term in vi_result.elbo_terms]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(iterations, expected_ll, color="#1f77b4")
    axes[0].set_title("Data fit term E_q[log p(y | μ)]")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Contribution")
    axes[0].grid(alpha=0.3)

    axes[1].plot(iterations, expected_prior, color="#ff7f0e")
    axes[1].set_title("Prior term E_q[log p(μ)]")
    axes[1].set_xlabel("Iteration")
    axes[1].grid(alpha=0.3)

    axes[2].plot(iterations, entropy, color="#2ca02c")
    axes[2].set_title("Entropy term −E_q[log q(μ)]")
    axes[2].set_xlabel("Iteration")
    axes[2].grid(alpha=0.3)

    fig.suptitle("How each ELBO component steers the optimisation")
    fig.tight_layout()
    plt.show()


def visualize_elbo_landscape(
    data: np.ndarray, config: ModelConfig, vi_result: VIResult
) -> None:
    """Show the ELBO surface over (mean, log_std) with the optimisation path."""

    means = np.array(vi_result.means)
    log_stds = np.log(np.array(vi_result.stds))

    mean_min = float(min(means.min(), np.mean(data) - 2.0))
    mean_max = float(max(means.max(), np.mean(data) + 2.0))
    log_std_min = float(log_stds.min() - 0.8)
    log_std_max = float(log_stds.max() + 0.8)

    grid_means = np.linspace(mean_min, mean_max, 120)
    grid_log_stds = np.linspace(log_std_min, log_std_max, 120)
    elbo_grid = np.empty((grid_log_stds.size, grid_means.size))

    for i, log_std in enumerate(grid_log_stds):
        for j, mean in enumerate(grid_means):
            elbo_grid[i, j] = compute_elbo(data, mean, log_std, config)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    contour = ax.contourf(
        grid_means,
        grid_log_stds,
        elbo_grid,
        levels=30,
        cmap="viridis",
    )
    plt.colorbar(contour, ax=ax, label="ELBO value")

    ax.plot(vi_result.means, log_stds, color="white", linewidth=2.0, label="Gradient ascent path")
    ax.scatter(vi_result.means[0], log_stds[0], color="yellow", edgecolor="black", zorder=3, label="Start")
    ax.scatter(vi_result.means[-1], log_stds[-1], color="red", edgecolor="black", zorder=3, label="Finish")

    ax.set_title("ELBO landscape over variational parameters")
    ax.set_xlabel("Mean parameter m")
    ax.set_ylabel("Log standard deviation log s")
    ax.legend(loc="best")
    ax.grid(alpha=0.2, color="white")

    plt.tight_layout()
    plt.show()


def print_step_explanations(
    data: np.ndarray,
    config: ModelConfig,
    vi_result: VIResult,
    exact_posterior: Tuple[float, float],
) -> None:
    print("\n=== Interpreting the optimisation steps ===")
    sample_mean = float(np.mean(data))
    exact_mean, exact_std = exact_posterior
    print(
        f"The sample mean is {sample_mean:.3f}; the exact posterior is N({exact_mean:.3f}, {exact_std:.3f}²)."
        " Below are several key iterations with their geometric meaning:"
    )

    snapshot_indices = _choose_snapshot_indices(len(vi_result.means))
    for idx in snapshot_indices:
        mean = vi_result.means[idx]
        std = vi_result.stds[idx]
        elbo = vi_result.elbo_values[idx]
        grad_mean, grad_log_std = compute_gradients(data, mean, math.log(std), config)

        if idx == 0:
            stage = "Initialisation"
        elif idx == len(vi_result.means) - 1:
            stage = "Converged state"
        else:
            stage = f"Step {idx}"

        direction_mean = (
            "Move right (increase mean)"
            if grad_mean > 1e-6
            else ("Move left (decrease mean)" if grad_mean < -1e-6 else "Keep mean stable")
        )
        direction_std = (
            "Shrink variance"
            if grad_log_std < -1e-6
            else ("Expand variance" if grad_log_std > 1e-6 else "Keep variance stable")
        )

        delta_text = ""
        if idx > 0:
            delta_mean = mean - vi_result.means[idx - 1]
            delta_std = std - vi_result.stds[idx - 1]
            delta_text = f" Changes this step: Δmean={delta_mean:+.3f}, Δstd={delta_std:+.3f}."

        print(
            f"[{stage}] ELBO = {elbo:.3f}; current q(μ) = N({mean:.3f}, {std:.3f}²)."
            f" Gradient direction: {direction_mean}, {direction_std}.{delta_text}"
        )

    diff_mean = abs(vi_result.mean - exact_mean)
    diff_std = abs(vi_result.std - exact_std)
    print(
        f"Final discrepancy: |mean error| = {diff_mean:.3e}, |std error| = {diff_std:.3e}."
        " This shows the variational posterior is closely tracking the exact solution."
    )


def explain_elbo_terms(
    data: np.ndarray, config: ModelConfig, vi_result: VIResult
) -> None:
    """Narrate how each ELBO term reacts to the optimisation."""

    print("\n=== Step 4 · Understand what maximising the ELBO is doing ===")
    labels = ["Data fit", "Prior regulariser", "Entropy"]
    narratives = [
        "This term becomes less negative when m moves toward the sample mean and when s is not too large.",
        "This term prefers values of m near 0 and a moderate variance, reflecting the prior belief.",
        "This term rewards keeping q broad; if s shrinks too much the entropy decreases.",
    ]

    print("We evaluate the three terms at several iterations to see their push and pull:")
    snapshot_indices = _choose_snapshot_indices(len(vi_result.elbo_terms))
    for idx in snapshot_indices:
        mean = vi_result.means[idx]
        std = vi_result.stds[idx]
        terms = vi_result.elbo_terms[idx]
        stage = "Initial" if idx == 0 else ("Converged" if idx == len(vi_result.means) - 1 else f"Step {idx}")
        print(f"  • {stage:>10}: q(μ)=N({mean:.3f}, {std:.3f}²)")
        for label, value, narrative in zip(labels, terms, narratives):
            print(f"      {label:<18}: {value:>8.3f}  → {narrative}")

    print(
        "Notice how the data-fit term increases sharply at first (we align with the sample mean),"
        " while the prior and entropy terms stabilise to balance data fidelity against prior knowledge"
        " and uncertainty."
    )


def main():
    config = ModelConfig(obs_std=1.0, prior_var=25.0)
    data = generate_data(true_mean=2.5, size=30, config=config, seed=2024)

    describe_math(data, config)

    vi_result = run_variational_inference(data, config, lr=0.02, steps=800)
    exact_posterior = closed_form_posterior(data, config)

    print("\n=== Comparing inference results ===")
    print(f"Variational posterior: mean = {vi_result.mean:.3f}, std = {vi_result.std:.3f}")
    print(f"Exact posterior: mean = {exact_posterior[0]:.3f}, std = {exact_posterior[1]:.3f}")
    print(f"Sample mean: {np.mean(data):.3f}")

    print_step_explanations(data, config, vi_result, exact_posterior)
    explain_elbo_terms(data, config, vi_result)
    visualize_inference_steps(data, config, vi_result, exact_posterior)
    visualize_elbo_decomposition(vi_result)
    visualize_elbo_landscape(data, config, vi_result)


if __name__ == "__main__":
    main()
