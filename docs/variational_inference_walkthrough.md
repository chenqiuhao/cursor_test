# 高斯均值变分推断推导与可视化详解

> 本笔记假设你已经阅读了 `main.py`，但希望逐行理解每个公式的来源、与代码的对应关系，以及脚本生成的图像如何反映推断过程。下面按照“模型 → ELBO 推导 → 梯度优化 → 可视化”四个部分展开。

## 1. 概率模型与符号约定

我们研究的共轭模型是：

- 观测模型：$y_i \mid \mu \sim \mathcal{N}(\mu, \sigma^2)$，其中 $\sigma$ 已知。
- 先验分布：$\mu \sim \mathcal{N}(0, \tau^2)$。

记观测数量为 $N$，则联合分布为
\[
 p(y, \mu) = \left( \prod_{i=1}^N \mathcal{N}(y_i \mid \mu, \sigma^2) \right) \mathcal{N}(\mu \mid 0, \tau^2).
\]

与之对应的代码初始化模型配置与生成模拟数据：

```python
@dataclass
class ModelConfig:
    obs_std: float = 1.0      # σ
    prior_var: float = 25.0   # τ²


def generate_data(true_mean: float, size: int, config: ModelConfig, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(loc=true_mean, scale=config.obs_std, size=size)
```

`ModelConfig` 保存 $(\sigma, \tau^2)$，`generate_data` 采样 $N$ 个观测点。本笔记后续默认 $\sigma = 1, \tau^2 = 25, N = 30$（脚本默认设置）。

## 2. 选择变分族并构造 ELBO

我们用单变量高斯作为变分族：
\[
q(\mu; m, s^2) = \mathcal{N}(m, s^2).
\]

ELBO（Evidence Lower Bound）定义为
\[
\mathcal{L}(m, s) = \mathbb{E}_q[\log p(y, \mu)] - \mathbb{E}_q[\log q(\mu)].
\]
将联合分布拆分得到三项：

1. **数据拟合项**
   \[
   \mathbb{E}_q[\log p(y \mid \mu)] = -\tfrac{1}{2}N\log(2\pi\sigma^2) - \tfrac{1}{2\sigma^2}\sum_{i=1}^{N} \left((y_i - m)^2 + s^2\right).
   \]
2. **先验项**
   \[
   \mathbb{E}_q[\log p(\mu)] = -\tfrac{1}{2}\log(2\pi\tau^2) - \tfrac{1}{2\tau^2}(s^2 + m^2).
   \]
3. **熵项**
   \[
   -\mathbb{E}_q[\log q(\mu)] = \tfrac{1}{2} \bigl(1 + \log(2\pi)\bigr) + \log s.
   \]

脚本中的 `compute_elbo_terms` 对应以上推导：

```python
def compute_elbo_terms(data: np.ndarray, mean: float, log_std: float, config: ModelConfig) -> Tuple[float, float, float]:
    obs_var = config.obs_std ** 2
    prior_var = config.prior_var
    std_sq = math.exp(2.0 * log_std)  # s²
    n = data.size

    centered_sq = np.square(data - mean)
    expected_ll = -0.5 * n * math.log(2.0 * math.pi * obs_var)
    expected_ll -= 0.5 * (np.sum(centered_sq) + n * std_sq) / obs_var

    expected_log_prior = -0.5 * math.log(2.0 * math.pi * prior_var)
    expected_log_prior -= 0.5 * (std_sq + mean**2) / prior_var

    entropy = 0.5 * (1.0 + math.log(2.0 * math.pi)) + log_std
    return expected_ll, expected_log_prior, entropy
```

注意实现中把标准差参数化为 $\log s$，因此通过 `std_sq = \exp(2 \log s)` 把它还原成方差。
（脚本顶部已经 `from typing import List, Tuple`，因此类型提示可直接引用 `Tuple`。）

## 3. 梯度与优化策略

为了用梯度上升最大化 ELBO，我们对 $m$ 和 $\log s$ 求偏导：

- 均值方向的梯度
  \[
  \frac{\partial \mathcal{L}}{\partial m} = -\frac{1}{\sigma^2} \sum_{i=1}^N (m - y_i) - \frac{m}{\tau^2}.
  \]
- $\log s$ 方向的梯度
  \[
  \frac{\partial \mathcal{L}}{\partial \log s} = 1 - \left(\frac{N}{\sigma^2} + \frac{1}{\tau^2}\right) s^2.
  \]

`compute_gradients` 直接实现了上述公式：

```python
def compute_gradients(data: np.ndarray, mean: float, log_std: float, config: ModelConfig) -> Tuple[float, float]:
    obs_var = config.obs_std ** 2
    prior_var = config.prior_var
    std_sq = math.exp(2.0 * log_std)
    n = data.size

    grad_mean = -(np.sum(mean - data) / obs_var) - mean / prior_var
    grad_log_std = -(n * std_sq / obs_var) - (std_sq / prior_var) + 1.0
    return grad_mean, grad_log_std
```

在 `run_variational_inference` 中，我们采用带回溯线搜索的梯度上升：

```python
def run_variational_inference(..., lr: float = 0.02, backtracking: int = 8, shrink: float = 0.5, min_step: float = 1e-6) -> VIResult:
    ...
    grad_mean, grad_log_std = compute_gradients(...)
    step_scale = 1.0
    for _ in range(backtracking):
        step = lr * step_scale
        trial_mean = mean + step * grad_mean
        trial_log_std = log_std + step * grad_log_std
        trial_elbo = compute_elbo(...)
        if trial_elbo >= current_elbo:
            mean, log_std = trial_mean, trial_log_std
            break
        step_scale *= shrink
    ...
```

只有当候选步长使 ELBO 不下降时才接受更新，否则逐步缩小步长；多次失败后提前终止，以防数值发散。函数返回的 `VIResult` 记录了每次迭代的 $m, s, \text{ELBO}$ 以及三项分解值，便于后续可视化。

## 4. 图像解读与生成

运行

```bash
python main.py --save-dir figures --no-show
```

会在 `figures/` 目录保存三张图，下文依次说明它们展示的公式含义。

### 4.1 推断全流程（文件：`figures/inference_panels.png`）

运行命令后 `visualize_inference_steps` 会生成下列四联图：

1. **左上：数据与先验** —— 直方图展示观测样本的经验分布，虚线是 $\mathcal{N}(0, \tau^2)$ 先验，灰色竖线是样本均值。直观对比“我们先验的想法”与“数据实际位置”。
2. **右上：$q(\mu)$ 的演化** —— 多条橙色曲线是不同迭代的 $\mathcal{N}(m_t, s_t^2)$，颜色越深代表越晚的迭代。绿色虚线是解析后验 $\mathcal{N}(\mu_\text{post}, \sigma_\text{post}^2)$。
3. **左下：参数轨迹** —— 红线记录 $m_t$，蓝线记录 $s_t$；虚线为解析后验的目标值。可以看到 $m_t$ 迅速靠近样本均值，而 $s_t$ 逐渐收缩到解析后验的方差。
4. **右下：ELBO 收敛** —— 显示 $\mathcal{L}(m_t, s_t)$ 随迭代单调上升，验证回溯线搜索的作用。

### 4.2 ELBO 三项分量（文件：`figures/elbo_decomposition.png`）

`visualize_elbo_decomposition` 输出的图像展示三条曲线，分别对应上一节推导的三项：

- 数据拟合项在初期迅速上升，说明 $m$ 正在贴近观测均值。
- 先验项略有下降，代表 $m$ 远离零带来的“先验惩罚”。
- 熵项随 $s$ 收缩而下降，提醒我们不要让 $q$ 过窄。

三者合起来就是 ELBO 的竞争平衡。

### 4.3 ELBO 等高线与优化轨迹（文件：`figures/elbo_landscape.png`）

`visualize_elbo_landscape` 生成的等高线图以 $m$ 为横轴、$\log s$ 为纵轴，底色编码 $\mathcal{L}(m, \log s)$ 的数值。白色折线是梯度上升路径，黄色点为初始位置，红点为收敛位置。可以直观看到：

- 目标区域呈椭圆形谷底，说明梯度是平滑的。
- 线搜索让路径沿着最陡方向逐渐逼近峰顶而不会震荡。

## 5. 文字讲解与日志

除了图像，脚本还提供逐步的文字说明：

- `describe_math`：在终端回顾模型、变分族与 ELBO 三项的含义。
- `print_step_explanations`：挑选若干迭代点，报告 $q(\mu)$ 的当前位置、梯度方向和每一步的增量。
- `explain_elbo_terms`：对比三项分量在关键迭代的取值，帮助理解它们的“拉锯战”。

通过阅读本笔记并配合运行脚本、查看生成的 PNG，你可以完整追溯每个公式的来源、代码实现以及对最终可视化的贡献。
