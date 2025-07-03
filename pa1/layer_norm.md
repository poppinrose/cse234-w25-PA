好的，遵照你的要求，我们使用你给定的格式和符号，对 LayerNorm 的反向传播进行完整、无跳步的推导。

---

## 一、LayerNorm 前向公式

假设输入 $x = (x_1, x_2, ..., x_m)$，归一化的维度有 $m$ 个元素，$\epsilon$ 是一个很小的常数。为了与反向传播的链式法则保持一致，我们先不考虑可学习参数 $\gamma$ 和 $\beta$。最终输出为 $\hat{x}_i$。

1.  **均值**
    $$
    \mu = \frac{1}{m} \sum_{i=1}^m x_i
    $$

2.  **方差**
    $$
    \sigma^2 = \frac{1}{m} \sum_{i=1}^m (x_i - \mu)^2
    $$

3.  **标准差**
    $$
    \sigma_{\epsilon} = \sqrt{\sigma^2 + \epsilon}
    $$
    (为了区分，这里用 $\sigma_\epsilon$ 表示带 $\epsilon$ 的标准差)

4.  **归一化输出**
    $$
    \hat{x}_i = \frac{x_i - \mu}{\sigma_{\epsilon}}
    $$

---

## 二、反向传播推导

我们要求解的是损失函数 $L$ 对输入 $x_i$ 的梯度 $\frac{\partial L}{\partial x_i}$。
已知上游传来的梯度为 $\frac{\partial L}{\partial \hat{x}_j}$。

### 1. 链式法则展开

根据多元链式法则，损失 $L$ 对 $x_i$ 的总梯度，是 $x_i$ 通过影响所有 $\hat{x}_j$ 而对 $L$ 产生影响的梯度之和。
$$
\frac{\partial L}{\partial x_i} = \sum_{j=1}^m \frac{\partial L}{\partial \hat{x}_j} \frac{\partial \hat{x}_j}{\partial x_i}
$$

### 2. 计算偏导数 $\frac{\partial \hat{x}_j}{\partial x_i}$

由于 $\hat{x}_j = \frac{x_j - \mu}{\sigma_{\epsilon}}$，其中 $\mu$ 和 $\sigma_{\epsilon}$ 都是所有 $x_k$ 的函数，我们使用商法则 $(\frac{u}{v})' = \frac{u'v - uv'}{v^2}$ 来求导。
$$
\frac{\partial \hat{x}_j}{\partial x_i} = \frac{ \frac{\partial(x_j - \mu)}{\partial x_i} \cdot \sigma_{\epsilon} - (x_j - \mu) \cdot \frac{\partial \sigma_{\epsilon}}{\partial x_i} }{ \sigma_{\epsilon}^2 }
$$

#### 2.1 计算各部分的导数

*   **$\frac{\partial (x_j - \mu)}{\partial x_i}$**：
    $$
    \frac{\partial (x_j - \mu)}{\partial x_i} = \frac{\partial x_j}{\partial x_i} - \frac{\partial \mu}{\partial x_i} = \delta_{ij} - \frac{1}{m}
    $$
    其中 $\delta_{ij}$ 是克罗内克函数，当 $i=j$ 时为 1，否则为 0。

*   **$\frac{\partial \sigma_{\epsilon}}{\partial x_i}$**：
    $$
    \frac{\partial \sigma_{\epsilon}}{\partial x_i} = \frac{\partial \sqrt{\sigma^2 + \epsilon}}{\partial x_i} = \frac{1}{2\sqrt{\sigma^2 + \epsilon}} \frac{\partial \sigma^2}{\partial x_i} = \frac{1}{2\sigma_{\epsilon}} \frac{\partial \sigma^2}{\partial x_i}
    $$

*   **$\frac{\partial \sigma^2}{\partial x_i}$**：
    $$
    \frac{\partial \sigma^2}{\partial x_i} = \frac{1}{m} \sum_{k=1}^m \frac{\partial (x_k - \mu)^2}{\partial x_i} = \frac{1}{m} \sum_{k=1}^m 2(x_k - \mu) \frac{\partial (x_k - \mu)}{\partial x_i}
    $$
    代入 $\frac{\partial (x_k - \mu)}{\partial x_i} = \delta_{ik} - \frac{1}{m}$：
    $$
    \frac{\partial \sigma^2}{\partial x_i} = \frac{2}{m} \sum_{k=1}^m (x_k - \mu)(\delta_{ik} - \frac{1}{m}) = \frac{2}{m} \left( (x_i - \mu) - \frac{1}{m}\sum_{k=1}^m(x_k - \mu) \right)
    $$
    因为 $\sum_{k=1}^m(x_k - \mu) = 0$，所以：
    $$
    \frac{\partial \sigma^2}{\partial x_i} = \frac{2}{m}(x_i - \mu)
    $$
    将此结果代回 $\frac{\partial \sigma_{\epsilon}}{\partial x_i}$：
    $$
    \frac{\partial \sigma_{\epsilon}}{\partial x_i} = \frac{1}{2\sigma_{\epsilon}} \cdot \frac{2(x_i - \mu)}{m} = \frac{x_i - \mu}{m\sigma_{\epsilon}}
    $$

#### 2.2 组装 $\frac{\partial \hat{x}_j}{\partial x_i}$

现在我们将计算出的导数代回 $\frac{\partial \hat{x}_j}{\partial x_i}$ 的表达式：
$$
\frac{\partial \hat{x}_j}{\partial x_i} = \frac{ (\delta_{ij} - \frac{1}{m}) \cdot \sigma_{\epsilon} - (x_j - \mu) \cdot \frac{x_i - \mu}{m\sigma_{\epsilon}} }{ \sigma_{\epsilon}^2 }
$$
整理一下：
$$
\frac{\partial \hat{x}_j}{\partial x_i} = \frac{1}{\sigma_{\epsilon}}(\delta_{ij} - \frac{1}{m}) - \frac{(x_j - \mu)(x_i - \mu)}{m \sigma_{\epsilon}^3}
$$

### 3. 计算最终梯度 $\frac{\partial L}{\partial x_i}$

我们将 $\frac{\partial \hat{x}_j}{\partial x_i}$ 代入链式法则的求和式中：
$$
\frac{\partial L}{\partial x_i} = \sum_{j=1}^m \frac{\partial L}{\partial \hat{x}_j} \left[ \frac{1}{\sigma_{\epsilon}}(\delta_{ij} - \frac{1}{m}) - \frac{(x_j - \mu)(x_i - \mu)}{m \sigma_{\epsilon}^3} \right]
$$
将求和分配到各项：
$$
\frac{\partial L}{\partial x_i} = \frac{1}{\sigma_{\epsilon}} \sum_{j=1}^m \frac{\partial L}{\partial \hat{x}_j}(\delta_{ij} - \frac{1}{m}) - \frac{x_i - \mu}{m \sigma_{\epsilon}^3} \sum_{j=1}^m \frac{\partial L}{\partial \hat{x}_j}(x_j - \mu)
$$
对第一项求和，由于 $\delta_{ij}$ 的性质，求和中只有 $j=i$ 的项不为零：
$$
\sum_{j=1}^m \frac{\partial L}{\partial \hat{x}_j}(\delta_{ij} - \frac{1}{m}) = \frac{\partial L}{\partial \hat{x}_i} - \frac{1}{m}\sum_{j=1}^m \frac{\partial L}{\partial \hat{x}_j}
$$
代回原式：
$$
\frac{\partial L}{\partial x_i} = \frac{1}{\sigma_{\epsilon}} \left( \frac{\partial L}{\partial \hat{x}_i} - \frac{1}{m}\sum_{j=1}^m \frac{\partial L}{\partial \hat{x}_j} \right) - \frac{x_i - \mu}{m \sigma_{\epsilon}^3} \sum_{j=1}^m \frac{\partial L}{\partial \hat{x}_j}(x_j - \mu)
$$

### 4. 简化为最终形式

为了与代码实现匹配，我们提出公因式 $\frac{1}{\sigma_{\epsilon}}$：
$$
\frac{\partial L}{\partial x_i} = \frac{1}{\sigma_{\epsilon}} \left[ \frac{\partial L}{\partial \hat{x}_i} - \frac{1}{m}\sum_{j=1}^m \frac{\partial L}{\partial \hat{x}_j} - \frac{x_i - \mu}{\sigma_{\epsilon}^2} \cdot \frac{1}{m} \sum_{j=1}^m \frac{\partial L}{\partial \hat{x}_j}(x_j - \mu) \right]
$$
注意到 $\sigma_{\epsilon}^2 = \sigma^2 + \epsilon$，最终公式为：
$$
\frac{\partial L}{\partial x_i} = \frac{1}{\sqrt{\sigma^2 + \epsilon}} \left[ \frac{\partial L}{\partial \hat{x}_i} - \frac{1}{m}\sum_{j=1}^m \frac{\partial L}{\partial \hat{x}_j} - \frac{x_i - \mu}{\sigma^2 + \epsilon} \cdot \frac{1}{m} \sum_{j=1}^m \frac{\partial L}{\partial \hat{x}_j}(x_j - \mu) \right]
$$
如果包含可学习参数 $\gamma$ 和 $\beta$，只需将 $\frac{\partial L}{\partial \hat{x}_j}$ 替换为 $\frac{\partial L}{\partial y_j}\gamma_j$ 即可。

---
