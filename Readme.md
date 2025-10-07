此代码仓用于熟悉 cursor 与 Codex 的配合使用。

## 使用方式

1. 在本地或容器中克隆仓库。
2. 使用 cursor 打开仓库，尝试与 Codex 进行协作式开发。
3. 可在 `Readme.md` 中记录新的心得或操作流程，方便后续参考。

## 备忘

- 如需演示，可随时新增测试文件或笔记。
- 建议在完成一次练习后提交 commit 以保留过程。

## 变分推断演示

在 `main.py` 中提供了一个共轭高斯模型的变分推断示例：观测数据满足 $y_i \sim \mathcal{N}(\mu, \sigma^2)$，先验为 $\mu \sim \mathcal{N}(0, \tau^2)$，其中 $\sigma$ 已知。我们选取单变量正态分布 $q(\mu) = \mathcal{N}(m, s^2)$ 作为变分分布，通过梯度上升最大化证据下界（ELBO）：

$$
\mathrm{ELBO}(m, s) = \mathbb{E}_q[\log p(y|\mu)] + \mathbb{E}_q[\log p(\mu)] - \mathbb{E}_q[\log q(\mu)].
$$

由于模型共轭，可以解析地写出各项：

* $\mathbb{E}_q[\log p(y|\mu)] = -\tfrac{1}{2}N\log(2\pi\sigma^2) - \tfrac{1}{2\sigma^2}\sum_i \big((y_i - m)^2 + s^2\big)$
* $\mathbb{E}_q[\log p(\mu)] = -\tfrac{1}{2}\log(2\pi\tau^2) - \tfrac{1}{2\tau^2}(s^2 + m^2)$
* $-\mathbb{E}_q[\log q(\mu)] = \tfrac{1}{2}\log(2\pi e s^2)$

对 $m$ 与 $\log s$ 求导可得梯度，从而实现梯度上升。脚本还计算解析后验 $p(\mu|y)$ 以便比较，并绘制了先验、变分近似、精确后验的密度曲线以及 ELBO 收敛过程。

### 运行方式

```bash
python main.py
```

运行后终端会打印模型与公式说明，弹出的图像窗口展示分布对比与 ELBO 曲线。
