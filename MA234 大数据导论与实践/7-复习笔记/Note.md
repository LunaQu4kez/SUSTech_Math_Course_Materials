

# MA234 Introduction to Big Data Notes



[TOC]

## 1. Introduction

### 机器学习 (Machine Learning) 的分类

- 监督学习 (supervised learning)：利用一组已知类别的样本调整分类器的参数，使其达到所要求性能
  - 分类 (classification)：输出是离散的
  - 回归 (regression)：输出是连续的

- 无监督学习 (unsupervised learning)：根据类别未知(没有被标记)的训练样本解决模式识别
  - 密度估计 (density estimation)
  - 聚类 (clustering)：比如 K-Means，SOM
  - 降维 (dimensionality reduction)
- 半监督学习 (semi-supervised learning)：存在缺失数据
  - 比如填补有缺失的图像等
- 强化学习 (reinforcement learning)
  - 对抗类游戏的人工智能



### 数据表示（PPT说的比较简洁清晰，直接放截图了）

<div align="center">
    <img src=".\\pic\\01_02.png" alt="" width="350">
    <img src=".\\pic\\01_03.png" alt="" width="350">
</div>




### 一些相关概念

<img src=".\\pic\\01_04.png" alt="" width="600" align="left">



### 风险最小化策略 (Risk Minimization Strategy)

<img src=".\\pic\\01_05.png" alt="" width="550" align="left">



### 模型评估 (Model Assessment)

假设我们预测出了$y = \hat{f}(\bold{x})$，那么有以下 3 种误差 (error)，其中 $L(y, f(\bold{x}))$ 代表损失函数（见上面）：

- 训练误差 (Training error)：$R_{emp}(\hat{f}) = \frac{1}{n} \sum_{i = 1}^{n} L(y_i, \hat{f}(\bold{x_i}))$，说明了该问题的学习难度
- 测试误差 (Test error)：$e_{test}(\hat{f}) = \frac{1}{m} \sum_{j = n+1}^{n+m} L(y_j, \hat{f}(\bold{x_j}))$，说明了问题的预测能力
  - error rate：$e_{test}$
  - accuracy：$r_{test}$
  - $e_{test} + r_{test} = 1$
- 泛化误差 (Generalization error)：$R_{exp}(\hat{f}) = E_p[L(y, \hat{f}(\bold{x}))] = \int_{\mathcal{X}\times\mathcal{X}} L(y, \hat{f}(\bold{x}))P(\bold{x},y)d\bold{x}dy$ ，刻画了学习算法的经验风险与期望风险之间偏差和收敛速度



### 过拟合 (Overfitting)

过多模型参数导致，对训练集更好，但对测试集更差

应对方法：

- 正则化 (Regularization)：$\min\limits_{f \in F} \frac{1}{n} \sum_{i = 1}^{n} L(y_i, f(\bold{x_i}))+\lambda J(f)$，加入惩罚项$J(f)$，模型参数越多，$J(f)$越大，选择 合适的 $\lambda$ 可同时将经验风险和模型复杂性降至最低
- 交叉验证 (Cross-validation, CV)：将训练集拆分为训练子集和验证子集，使用训练集重复训练不同的模型，使用验证集选择验证误差最小的模型
  - Simple CV：将数据随机拆分为两个子集
  - K-fold CV：将数据随机拆分为大小相同的 K 个不相交子集，将 K − 1 个子集的并集视为训练集，将另一个子集视为验证集，重复执行此操作并选择平均验证误差最小的模型
  - Leave-one-out CV：在上一种情况下取 K = n



## 2. 数据预处理 (Data Preprocessing)

**数据种类**：表格数据 (Tabular data) （矩阵、向量、对象、关系），图形数据 (Graphical data)（网络、图形），多媒体数据 (Multi-media data)（文本、图像、视频、音频）

**属性 (Attributes) 类型**：离散的、连续的



### 基本的统计概念

- 平均数 (Mean)

- 中位数 (Median)

- 最大值 (Maximum)，最小值 (Minimum)

- 分位数 (Quantile)：中位数的推广，$k^{th}$ q-分位数 $x_q$：$P[\bold{X} < x_q] \leq k/q$

  四分位距 (interquartile range)：$IQR=Q3(75\%) − Q1(25\%)$

- 方差 (Variance) $Var(\bold{X})$，标准差 (Standard deviation)
- 众数 (Mode)

- 箱形图 (Box Plot)：如下图示例

  <img src=".\\pic\\02_01.png" alt="" width="250" align="left">



### 距离 (Distance)

#### 有关布尔变量的距离

对于布尔变量，有 4 种距离

- symmetric distance $d(\bold{x_i},\bold{x_j}) = \frac{r+s}{q+r+s+t}$
- Rand index $Sim_{Rand}(\bold{x_i},\bold{x_j}) = \frac{q+t}{q+r+s+t}$
- non-symmetric distance $d(\bold{x_i},\bold{x_j}) = \frac{r+s}{q+r+s}$
- Jaccard index $Sim_{Jaccard}(\bold{x_i},\bold{x_j}) = \frac{q}{q+r+s}$

<img src=".\\pic\\02_02.png" alt="" width="180" align="left">

#### 闵可夫斯基 (Minkowski) 距离

$$
d(\bold{x_i},\bold{x_j}) = \sqrt[h]{\sum\limits_{k=1}^p \left| \bold{x_{ik}} - \bold{x_{jk}} \right|^h }
$$
上述距离被称作 $L_h$ 范数 (norm)

- 正定： $d(\bold{x_i} , \bold{x_j}) \ge 0$ 等号成立当且仅当 $i = j$ 

- 对称：$d(\bold{x_i} , \bold{x_j}) = d(\bold{x_j} , \bold{x_i})$
- 三角不等式：$d(\bold{x_i} , \bold{x_j}) \le d(\bold{x_i},\bold{x_k}) + d(\bold{x_k},\bold{x_j})$

| $h$      | 名称                            |
| -------- | ------------------------------- |
| 1        | 曼哈顿距离 (Manhattan distance) |
| 2        | 欧式距离 (Euclidean distance)   |
| $\infty$ | Supremum distance               |

#### 余弦相似度 (Cosine Similarity)

$$
cos(\bold{x_i},\bold{x_j}) = \frac{\bold{x_i}\cdot\bold{x_j}}{\|\bold{x_i}\|\|\bold{x_j}\|}
$$



### 数据缩放 (Data Scaling)

#### 数据缩放的原因

- 为更好的性能：比如 SVM 中的 RBF 和 Lasso/岭回归中的惩罚项，需假设均值为 0 方差为 1
- 规范化不同的维度：身高（1.75m）和体重（70kg）

#### 4种数据放缩

- Z-score scaling：$\bold{x_i^*} = \frac{\bold{x_i} - \hat{\mu}}{\hat{\sigma}}$，$\hat{\mu}$ 样本均值，$\hat{\sigma}$ 样本方差，**适用于最大值和最小值未知且数据分布良好**
- 0-1 scaling：$\bold{x_i^*} = \frac{\bold{x_i} - min_k\bold{x_k}}{max_k\bold{x_k} - min_k\bold{x_k}} \in [0,1]$，**适用于有界数据集，新加入数据要重新计算最大值最小值**
- Decimal scaling：$\bold{x_i^*} = \frac{\bold{x_i}}{10^k}$，**适用于适用于不同幅度的数据**
- Logistic scaling：$\bold{x_i^*} = \frac{1}{1+e^{-\bold{x_i}}}$，**适用于数据集中在原点附近，但会改变数据的分布**



### 数据离散化 (Data Discretization)

#### 离散化的原因

- 提高鲁棒性 (robustness)：通过将异常值放入某些区间来消除异常值
- 能够对算法提供更合理的解释
- 降低存储和计算消耗

#### 分类

- 无监督离散化 (Unsupervised discretization)
  - 等距离散化 (Equal-distance discretization)
  - 等频离散化 (Equal-frenquency discretization)
  - 基于聚类的离散化 (Clustering-based discretization)：进行分层聚类并得到分层结构（如使用 K-Means），并将同层中的样本放入相同的区间（比如家谱）
  - 基于 3σ 的离散化 (3σ-based discretization)：将样本分成 8 个间隔，需要先取对数
- 监督离散化 (Supervised discretization)
  - 基于信息增益的离散化 (Information Gain)：使用决策树进行分类
  - 卡方分箱离散化 (Chi Merge)：比较复杂一点，[参考链接](https://blog.csdn.net/fulk6667g78o8/article/details/120318205)



### 数据冗余 (Data Redundancy)

- 关联性强的变量 (Attribute)：比如 “出生日期” 和 “年龄”
- 通过关联分析确定数据冗余
  - 对于连续变量 A 和 B，计算相关系数 $\rho_{A,B}=\frac{\sum\limits_{i=1}^k (a_i-\overline{A})(b_i-\overline{B})}{k\hat{\sigma}_A\hat{\sigma}_B} \in [-1,1]$
  - 对于离散变量 A 和 B，计算$\chi^2$，越大说明关联性越小



### 数据缺失 (Missing Data)

#### 数据缺失的原因

-  Missing Completely At Random (MCAR)：缺失数据的出现是随机事件
- Missing At Random (MAR)：取决于一些控制变量，例如，在青少年调查中，年龄大于 20 的样本会被排除
- Missing Not At Random (MNAR)：例如，表现不佳的员工被解雇后缺少数据

#### 应对方法

- 简单的方法：删除样本（**适用于少量样本有数据缺失**），删除变量（**适用于某一变量缺失值较多**）
- 填补
  - 填补 0
  - 用数字类型的均值填充，用非数字类型的模数填充，**适用于MCAR**
    - 缺点：集中于均值而低估方差
    - 解决方案：填写不同的组
  - 用相似变量填充：引入自相关 (auto-correlation)
  - 用历史数据填充
  - K-Means 填充：使用完善的变量（无缺失值）计算数据的成对距离，然后用前 K 个最相似的完善数据的平均值填充缺失值，将自相关引入
  - 用期望最大化 (EM) 填充：引入隐藏变量并使用 MLE 估计缺失值
  - 随机填充 (Random filling)
    - Bayesian Bootstrap，Approximate Bayesian Bootstrap（略）
  - 基于模型的方法 (Model based methods)：将缺失变量视为 y，将其他变量视为 x，将没有缺失值的数据作为我们的训练集来训练分类或回归模型，将缺失值的数据作为测试集来预测缺失值
  - 插值填充 (Filling by Interpolation)

#### 哑变量 (Dummy Variables)

哑变量，也称为虚拟变量或名义变量，是一种用于表示分类变量的变量类型。哑变量通常取值为 0 或 1，用来反映某个变量的不同属性。在一个有 n 个分类属性的自变量中，通常需要选择一个分类作为参照，从而产生 n-1 个哑变量。



### 离群值 (Outlier)

数据点看似来自不同的分布，或噪声数据，通常采用无监督检测

#### 离群值检测

- 上下 α 分位数之外的样本（取较小的 α，通常是 1%）

- 从箱形图观察

- 局部异常值因子 (LOF)：是一种基于密度的方法

  计算每个点 $\bold{x}$ 的密度，将每个点 $\bold{x}$ 的密度与其相邻点的密度进行比较

#### 相关概念

<div align="center">
    <img src=".\\pic\\02_03.png" alt="" width="500">
    <img src=".\\pic\\02_04.png" alt="" width="450">
</div>





## 3. 分类1 (Classification)

分类是一种监督学习，简而言之是对样本 $\bold{x}$ 预测它的标签 $y$

训练阶段：给定数据集 $D = \{ (\bold{x},y) \}$，分割为 $D = D_{train}\cup D_{test}$，找一个能最佳的关联 $\bold{x_{train}}$ 和 $y_{train}$ 的函数 $y = f(\bold{x})$，然后测试这个函数能在多大程度上适配 $\bold{x_{test}}$ 和 $y_{test}$ 

预测阶段：将训练得到的函数用于没有标签的样本 $\bold{x_{pred}}$，得到预测值 $y_{pred} = f(\bold{x_{pred}})$



### kNN (k-Nearest Neighbour)

对于待预测的 $\bold{x}$，找和它相邻的 k 个点，统计它们的标签，数量最多的标签作为 $\bold{x}$ 的标签。

```
Compute d(x,x_j) for each (x_j,y_j) in D_train
Sort the distances in an ascending order, choose the first k samples (x_1,y_1),...,(x_k,y_k)
Make majority vote y_pred = Mode(y_1,...,y_k)
```

**特点**：是最简单的监督学习算法，同时进行训练和测试，低偏差（bias，预测值和真实值之间的误差）、高方差（variance，预测值之间的离散程度）

**优点：对异常值不敏感，易于实现和并行化，适用于大型训练集，适用于对数据的先验知识非常有限的情况**

缺点：需要调参 k，占用储存空间大，计算量大且密集

#### 调参k —— 交叉验证 M-fold Cross-validation

将数据集分为 M 折（通常取 M = 5 或 10），设 $\kappa:\{1,...,N\}\rightarrow \{1,...,M\}$ 是随即分区索引映射，那么预测误差的 CV 估计值为 $CV(\hat{f},k) = \frac{1}{N} \sum\limits_{i = 1}^{N} L(y_i, \hat{f}^{-\kappa_i}(\bold{x_i},k))$ 

#### 分析

**时间复杂度**：O(mndK)，其中 n 是训练样本的数量，m 是测试样本的数量，d 是维度，K 是最近邻参数

**误差**：假设样本是 i.i.d (独立同分布) 的，对于任何测试样本 $\bold{x}$ 和足够小的 $δ$，总存在一个训练样本 $\bold{z}\in B(\bold{x},δ)$ （$\bold{x}$ 的标签与 $\bold{z}$ 的标签相同），则 1NN 误差为
$$
\epsilon = \sum\limits_{c=1}^C p_c(x)(1-p_c(z)) \xrightarrow{\delta\rightarrow0}1-\sum\limits_{c=1}^C p^2_c(x)
$$


### 决策树 (Decision Tree)

简介略，如图

但要注意，尽量将整个数据集划分成的每个部分所含杂质尽量少，下面介绍 3 种不纯度度量

<img src=".\\pic\\03_01.png" alt="" width="300" align="center">

#### 不纯度 (Impurity) 度量 - GINI 指数 (GINI Index)

- 节点 (Node) t 的 Gini 指数：$Gini(t) = 1-\sum\limits_{c=1}^C (p(c|t))^2$，$p(c|t)$ 是节点 t 中 c 类数据的比例

- $Gini(t)$ 的最大值是 $1-\frac{1}{C}$，在 $p(c|t) = \frac{1}{C}$ 时取到；最小值是 0，在对于某些 $c$ 有 $p(c|t) = 1$ 时取到

- 一条分支 (Split) 的 Gini 指数：$Gini_{split}=\sum\limits_{k=1}^K\frac{n_k}{n}Gini(k)$，其中 $n_k$ 是子节点 k 的样本数量，$n = \sum\limits_{k=1}^Kn_k$

- 选择使得 $Gini(t) - Gini_{split}$ 最大的分支

<img src=".\\pic\\03_02.png" alt="" width="500" align="center">

#### 不纯度度量 - 信息增益 (Information Gain)

- 节点 t 的熵 (Entropy)：$H(t) = -\sum\limits_{c=1}^C p(c|t)\log_{2}{p(c|t)}$ 

- $H(t)$ 的最大值是 $\log_{2}{C}$，在 $p(c|t) = \frac{1}{C}$ 时取到；最小值是 0，在对于某些 $c$ 有 $p(c|t) = 1$ 时取到

- 信息增益：$InfoGain_{split}=H(t) - \sum\limits_{k=1}^K\frac{n_k}{n}H(k)$，其中 $n_k$ 是子节点 k 的样本数量，$n = \sum\limits_{k=1}^Kn_k$ 

- 引入信息增益比 (Introduce information gain ratio)：$SplitINFO=- \sum\limits_{k=1}^K\frac{n_k}{n}\log_2{\frac{n_k}{n}}$ ，$InfoGainRatio=\frac{InfoGain_{split}}{SplitINFO}$ （C4.5 算法）

- 选择使得 $InfoGain_{split}$ 最大的分支（ID3 算法）

- 缺点：容易生成过多的子节点导致过拟合

#### 不纯度度量 - 误分类误差 (Misclassification Error)

- 节点 t 的误分类误差：$Error(t) = 1-\max_c p(c|t)$ 
- 最大值是 $1-\frac{1}{C}$，在 $p(c|t) = \frac{1}{C}$ 时取到；最小值是 0，在对于某些 $c$ 有 $p(c|t) = 1$ 时取到



例：对于二分类问题，3 种不纯度度量如下图

<img src=".\\pic\\03_03.png" alt="" width="500" align="center">

#### 决策树相关算法

| 算法                                      | 属性类型       | 不纯度度量 | 分割的子节点数量 | 目标属性类型   |
| ----------------------------------------- | -------------- | ---------- | ---------------- | -------------- |
| Iterative Dichotomiser 3 (ID3)            | 离散型         | 信息增益   | k ≥ 2            | 离散型         |
| C4.5                                      | 离散型、连续型 | 信息增益率 | k ≥ 2            | 离散型         |
| C5.0                                      | 离散型、连续型 | 信息增益率 | k ≥ 2            | 离散型         |
| Classification and Regression Tree (CART) | 离散型、连续型 | Gini 指数  | k = 2            | 离散型、连续型 |

<img src=".\\pic\\03_04.png" alt="" width="600" align="center">

对于较为复杂的树，可以采用剪枝 (Pruning) 的方法减小其复杂程度

<img src=".\\pic\\03_05.png" alt="" width="600" align="center">

#### 优缺点

优点：

- 容易解释和可视化：广泛应用于金融、医疗健康、生物等领域
- 易于处理缺失值（视为新数据类型）
- 可以扩展到回归

缺点：

- 由于采用贪心算法，容易得到局部最小值
- 决策的边界过于简单：例如与轴平行的线



### 朴素贝叶斯 (Naive Bayes)

基于贝叶斯定理和样本的条件独立假设而进行分类的算法

#### 贝叶斯定理

$$
P(Y|X) = \frac{P(X|Y)P(Y)}{P(X)}
$$

$P(Y)$ 先验 (prior) 概率分布，$P(X|Y)$ 似然 (likelihood) 函数，$P(X)$ 论据 (evidence)，$P(Y|X)$ 后验 (posterior) 概率分布

#### Naive Bayes Method

机器学习的目的是估计 $P(Y|X)$

假设 $X=\{X_1,...,X_d\}$（ $X$ 是 $d$ 维的）

对于给定的 $X=x$，$P(X=x)$ 对于 $Y$ 是独立的，由贝叶斯定理：
$$
P(Y|X=x)\propto P(X=x|Y)P(Y)
$$
假设 $X_1,...,X_d$ 是独立的，对于给定的 $Y=c$：
$$
P(X=x|Y=c) =  \prod\limits_{i=1}^d P(X_i=x_i|Y=c)
$$
那么，朴素贝叶斯算法就是以下面这个方式求 $y$ 的预测值 $\hat{y}$ 
$$
\hat{y}=\arg\max\limits_{c}P(Y=c)\prod\limits_{i=1}^d P(X_i=x_i|Y=c)
$$
对于数据集 $D = \{ (\bold{x_1},y_1),...,(\bold{x_n},y_n) \}$ 需要估计 $P(Y=c)$ 和 $P(X_i=x_i|Y=c)$ ，使用极大似然估计 (MLE)

MLE 估计 $P(Y=c)$ ：$P(Y=c)=\frac{\sum\limits_{i=1}^n I(y_i=c)}{n}$ 

当 $X_i$ 是离散的，且值域为 $\{v_1,...,v_k\}$，$P(X_i=v_k|Y=c)=\frac{\sum\limits_{i=1}^n I(x_i=v_k,y_i=c)}{\sum\limits_{i=1}^n I(y_i=c)}$ 

当 $X_i$ 是连续的，假设服从 $N(\mu,\sigma^2)$，$P(X_i=x|Y=c)=\frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$，MLE 估计参数 $\mu$ 和 $\sigma$ 

#### 优缺点

**优点**

- 对于离群值和缺失值表现较为稳定
- 鲁棒性：用于不相关的变量，$P(X|Y)$ 与 $Y$ 无关，因此对后验概率没有影响
- 即使不满足条件独立性假设，也可能优于更复杂的替代方案

**缺点**

- 违反条件独立性假设时，朴素贝叶斯的性能可能会更差
- 很大程度上取决于参数估计值的准确性



### 分类算法的模型评估

#### 混淆矩阵 (Confusion Matrix)

二分类问题的混淆矩阵示例

<img src=".\\pic\\03_06.png" alt="" width="200" align="left">

TP：true positive，TN：true negative，FP：false positive，FN：false negative

$\text{Accuracy} = \frac{TP+TN}{TP+FP+FN+TN}$ ，当样本分布不平衡时不是一个很好的参考指标

$\text{Precision} = \frac{TP}{TP+FP}$ 

$\text{Recall} = \frac{TP}{TP+FN}$，在医学领域很重要

$\text{Specifity} = \frac{TN}{FP+TN}$，负样本的 $\text{Recall}$ 

F Score：$F_\beta = \frac{(1+\beta^2)\text{Precision}\times\text{Recall}}{\beta^2\times\text{Precision}+\text{Recall}}$ 

#### Receiver Operating Characteristic (ROC) and AUC

目的是解决不同的类的分布不均衡。

为连续的预测值设置不同的阈值 t，比如若 $P(Y=1|X=X_i)>t$，则 $\hat{y}_i=1$ 

对不同的 t 值计算 $TPR = \frac{TP}{TP+FN}$，$FPR = \frac{FP}{FP+TN}$ 并绘制 ROC 曲线，ROC 越大，说明表现越好

AUC：ROC 曲线下的面积，越大越好，> 0.75 说明性能极佳

<img src=".\\pic\\03_07.png" alt="" width="300" align="left">

#### Kappa Coefficient 

<div align="center">
    <img src=".\\pic\\03_08.png" alt="" width="450">
    <img src=".\\pic\\03_09.png" alt="" width="450">
</div>

#### 多分类问题

- ROC 和 AUC 无法生效
- 混淆矩阵变为 $C \times C$，每个格子表示在预测类 i 和真实类 j 相交处的样本数量
- 多分类转化为多个二分类问题





## 4. 回归 (Regression)

从自变量 x 中预测因变量 y：$y = f(\bold{x})$ 或 $y = E(y|\bold{x})$ 

### 线性回归 (Linear Regression)

对于 n 个样本的数据集 $\{(\bold{x_i}, y_i)\}_{i=1}^n$，每个 $\bold{x_i}$ 是 $p$ 维的，设 $\bold{y} = (y_1,...,y_n)^T$， $\bold{w} = (w_1,...,w_p)$ 是代求的参数，$\bold{X} = [\bold{1_n},(\bold{x_1},...,\bold{x_n})^T]\in \mathbb{R}^{n\times(p+1)}$，$\epsilon = (\epsilon_1,...,\epsilon_n)^T\sim \mathcal{N}(\bold{0},\sigma^2\bold{I_n})$ 为误差，则有
$$
\bold{y} = \bold{X}\bold{w}+\epsilon
$$

#### (平凡)最小二乘 [(Ordinary) Least Square，OLS]

最小化残差平方和 (residual sum-of-squares)：
$$
RSS(\bold{w})=\sum\limits_{i=1}^n(y_i-w_0-w_1x_1-...-w_px_p)^2=\|\bold{y}-\bold{X}\bold{w}\|_2^2
$$

$$
\nabla_\bold{w}RSS(\bold{\hat{w}})=0 \quad \Rightarrow \quad \bold{\hat{w}}=(\bold{X}^T\bold{X})^{-1}\bold{X}^T\bold{y}
$$

$$
\bold{\hat{y}}=\bold{X}(\bold{X}^T\bold{X})^{-1}\bold{X}^T\bold{y}=\bold{P}\bold{y}
$$



其中，$\bold{P}$ 是一个投影矩阵，即 $\bold{P}^2=\bold{P}$ 

#### 评价指标

参数 $R^2 = 1-\frac{SS_{res}}{SS_{tot}}=(\frac{SS_{reg}}{SS_{tot}}\text{ for linear regression})$ ，越大说明回归效果越好

总平方和 (total sum of squares)： $SS_{tot}=\sum\limits_{i=1}^n(y_i-\overline{y})^2$  

回归平方和 (regression sum of squares)： $SS_{reg}=\sum\limits_{i=1}^n(\hat{y_i}-\overline{y})^2$  

残差平方和 (residual sum of squares)： $SS_{res}=\sum\limits_{i=1}^n(y_i-\hat{y_i})^2$  

#### 多元共线性 (Multicolinearity)

若 $\bold{X}$ 的列几乎线性相关（即多元共线），那么 $det(\bold{X}^T\bold{X})\approx0$，则 $(\bold{X}^T\bold{X})^{-1}$ 会非常大，那么得到的 $\bold{\hat{w}}$ 将会不准确。

#### Bias-Variance Decomposition

$L^2$ 损失中泛化误差 (generalization error) 的偏置方差分解 (bias-variance decomposition)：
$$
E_{train}R_{exp}(\hat{f}(x))=E_{train}E_P[(y-\hat{f}(x))^2|x]=Var(\hat{f}(x))+Bias^2(\hat{f}(x))+\sigma^2
$$
其中 $P=P(y|x)$ 

偏差 (Bias)：$Bias^2(\hat{f}(x))=E_{train}\hat{f}(x)-f(x)$，是对该模型进行预测的平均精度 (accuracy)

方差 (Variance)：$Var(\hat{f}(x))=E_{train}(\hat{f}(x)-E_{train}\hat{f}(x))^2$ ，是由于不同数据集而导致的模型预测的可变性（稳定性）

<img src=".\\pic\\04_01.png" alt="" width="300" align="left">

对于 kNN 回归（将 kNN 分类的众数改为平均数）有：

- 对于较小的 k，过拟合，低偏差，高方差
- 对于较大的 k，不足拟合，高偏差，低方差

### 正则化 (Regularization)

在维度过高，列很多的时候，方差会很大。减小一些系数或将其设置为零可以减少过拟合。选择更少的变量同样可以达到更高的效果。

#### Best-subset selection

对于所有 $k\in \{0, 1,...,p\}$ ，大小为 $k$ 的子集 $S_k\subset \{0, 1,...,p\}$，使得 $RSS(\bold{w})=\sum\limits_{i=1}^n(y_i-w_0-\sum\limits_{j\in S_k}w_jx_{ij})^2$ 

#### 使用惩罚项 (Penalties) 正则化

$$
\sum\limits_{i=1}^n(y_i-w_0-w_1x_1-...-w_px_p)^2+\lambda\|\bold{w}\|_q^q=\|\bold{y}-\bold{Xw}\|_2^2+\lambda\|\bold{w}\|_q^q
$$

q = 2 时，被称为岭回归；q = 1 时，被称为 LASSO 回归

#### 岭回归 (Ridge Regression)

$$
\bold{\hat{w}}=\arg\min\limits_\bold{w}\|\bold{y}-\bold{Xw}\|_2^2+\lambda\|\bold{w}\|_2^2
$$

$\lambda\ge0$ 是可以通过交叉验证 (CV) 调参的参数

该问题等价于下面的优化问题：
$$
\bold{\hat{w}}=\arg\min\limits_\bold{w}\|\bold{y}-\bold{Xw}\|_2^2,\quad \text{subject to }\|\bold{w}\|_2^2 \le \mu
$$
$\mu\ge0$ 是一个参数，较大的 $\lambda$ 对应较小的 $\mu$ 

岭回归的解如下
$$
\hat{\bold{w}}^{ridge}=(\bold{X}^T\bold{X}+\lambda\bold{I}_{p+1})^{-1}\bold{X}^T\bold{y} \\
\hat{\bold{y}}^{ridge}=\bold{X}\hat{\bold{w}}^{ridge}
$$
也可以使用 SVD 先将矩阵 $\bold{X}$ 分解，得到 $\bold{X} = \bold{P}\bold{D}\bold{Q}$，其中 $\bold{P}\in\mathbb{R}^{n\times(p+1)}$，$\bold{Q}\in\mathbb{R}^{(p+1)\times(p+1)}$，都是正交矩阵（即 $\bold{P}^T\bold{P} = \bold{I}$，$\bold{Q}^T\bold{Q} = \bold{I}$）$\bold{D}$ 是对角矩阵，$\bold{D} = diag(v_1,...,v_{p+1})$，那么得到岭回归的解为
$$
\hat{\bold{y}}^{ridge}=\bold{P}diag(\frac{v_1}{v_1^2+\lambda},...,\frac{v_{p+1}}{v_{p+1}^2+\lambda})\bold{P}^T\bold{y}
$$
（可以对比一下 OLS (ordinary least square) 得到的解 $\hat{\bold{y}}^{OLS}=\bold{P}\bold{P}^T\bold{y}$ 

#### 从贝叶斯视角看岭回归

<img src=".\\pic\\04_02.png" alt="" width="600" align="left">

#### 岭轨迹 (Ridge Trace)

以 $\lambda$ 为自变量，$\hat{\bold{w}}^{ridge}(\lambda)$ 为因变量的函数图像被称为岭轨迹

当 $\lambda \in (0,0.5)$ 时，岭轨迹很不稳定，一般取 $\lambda=1$ 

下图是一些岭轨迹图像：

<img src=".\\pic\\04_03.png" alt="" width="300" align="left">

- 轨迹稳定、绝对值较小的系数对 $y$ 的影响不大，如 (a)
- 轨迹稳定、绝对值较大的系数对 $y$ 的影响较大，如 (b)
- 两个变量的岭轨迹不稳定，但和是稳定的，说明这两个变量有多重共线性，如 (d)
- 所有变量的岭轨迹稳定，说明使用 OLS 具有良好的性能，如 (f)

#### LASSO 回归

$$
 \bold{\hat{w}}=\arg\min\limits_\bold{w}\|\bold{y}-\bold{Xw}\|_2^2+\lambda\|\bold{w}\|_1
$$

该问题等价于下面的优化问题：
$$
\bold{\hat{w}}=\arg\min\limits_\bold{w}\|\bold{y}-\bold{Xw}\|_2^2,\quad \text{subject to }\|\bold{w}\|_1 \le \mu
$$
$\mu\ge0$ 是一个参数，较大的 $\lambda$ 对应较小的 $\mu$ 

LASSO 回归的解为：
$$
\hat{w}_i^{lasso}=(|\hat{w}_i^{OLS}|-\lambda)_+sign(\hat{w}_i^{OLS})
$$
$\hat{w}_i^{lasso}$ 被称为 $\hat{w}_i^{OLS}$ 的软阈值 (soft thresholding)，其中记号 $(a)_+ = max(a,0)$ 代表 $a$ 的正部

LASSO 轨迹与岭轨迹定义相同，这些路径是分段线性的，可能多次穿过 x 轴。在实际应用中，通常会通过交叉验证 (CV) 的方式选择参数 $\lambda$ 。

下面是一种解 LASSO 回归的方法：

<img src=".\\pic\\04_04.png" alt="" width="600">

 与 LASSO 相关的一些模型：

- Elastic net：$\hat{\bold{w}}=\arg\min\limits_\bold{w} \|\bold{y}-\bold{X}\bold{w}\|_2^2+\lambda_1\|\bold{w}\|_2^2+\lambda_2\|\bold{w}\|_1$ 
- Group LASSO：$\hat{\bold{w}}=\arg\min\limits_\bold{w} \|\bold{y}-\bold{X}\bold{w}\|_2^2+\sum\limits_{g=1}^G\lambda_g\|\bold{w}_g\|_2$，其中 $\bold{w}=(\bold{w}_1,...,\bold{w}_g)$ 是 $\bold{w}$ 的一个分组

#### 最大后验估计 [Max A Posterior (MAP) Estimation]

给定参数 $\theta$，$\bold{y}$ 的条件分布为 $P(\bold{y}|\theta)$，且参数 $\theta$ 有先验分布 $P(\theta)$ 

那么 $\theta$ 对于给定 $\bold{y}$ 的后验分布 $P(\theta|\bold{y})\propto P(\bold{y}|\theta)P(\theta)$ 

MAP 选择后验最大的参数估计：
$$
\hat{\theta}^{MAP}=\arg\max\limits_{\theta}P(\theta|\bold{y})=\arg\max\limits_{\theta}(\log P(\bold{y}|\theta)+ \log P(\theta))
$$
不同的对数先验导致不同的惩罚（正则化），但一般情况不是这样：有些惩罚可能不是概率分布的对数，其他一些惩罚取决于数据（先验独立于数据）



### 回归模型的评价

- Mean absolute error (MAE)：$MAE=\frac{1}{n}\sum\limits_{i=1}^n|y_i-\hat{y_i}|$ 

- Mean square error (MSE)：$MSE=\frac{1}{n}\sum\limits_{i=1}^n(y_i-\hat{y_i})^2$ 

- Root mean square error (RMSE)：$RMSE=\sqrt{MSE}$ 

- 相关系数 (Coefficient of Determination) $R^2 = 1-\frac{SS_{res}}{SS_{tot}}=(\frac{SS_{reg}}{SS_{tot}}\text{ for linear regression})$ ，越大说明回归效果越好

  总平方和 (total sum of squares)： $SS_{tot}=\sum\limits_{i=1}^n(y_i-\overline{y})^2$  

  回归平方和 (regression sum of squares)： $SS_{reg}=\sum\limits_{i=1}^n(\hat{y_i}-\overline{y})^2$  

  残差平方和 (residual sum of squares)： $SS_{res}=\sum\limits_{i=1}^n(y_i-\hat{y_i})^2$  

- 校正*系数*的决定系数 (Adjusted Coefficient of Determination)：$R_{adj}^2=1-\frac{(1-R^2)(n-1)}{n-p-1}$，其中 $n$ 代表样本数量，$p$ 代表样本的维数，越大说明回归效果越好

  当加入重要样本时，$R_{adj}^2$ 变大而 $SS_{res}$ 变小；当加入不重要样本时，$R_{adj}^2$ 可能变小而 $SS_{res}$ 可能变大

  实际上，$1-R_{adj}^2=\frac{\hat{\sigma}^2}{S^2}$，其中 $\hat{\sigma}^2=\frac{1}{n-p-1}\sum\limits_{i=1}^n(y_i-\hat{y_i})^2$，$S^2=\frac{1}{n-1}\sum\limits_{i=1}^n(y_i-\overline{y})^2$ （若 $\bold{w}=0$，则有 $(n-p-1)\frac{\hat{\sigma}^2}{\sigma^2}\sim \chi^2_{n-p-1}$ 和 $(n-1)\frac{S^2}{\sigma^2}\sim \chi^2_{n-1}$）





## 5. 分类2

### 逻辑回归 (Logistic Regression)

对于二分类问题，逻辑回归是
$$
P(y=1|\bold{X}=\bold{x})=\frac{\exp(\bold{w}^T\bold{x})}{1+\exp(\bold{w}^T\bold{x})} \\
P(y=0|\bold{X}=\bold{x})=\frac{1}{1+\exp(\bold{w}^T\bold{x})}
$$
假设 $P(y=k|\bold{X}=\bold{x})=p_k(\bold{x};\bold{w}) \quad k=0,1$，似然函数为 $L(\bold{w})=\prod\limits_{i=1}^n p_{y_i}(\bold{x_i};\bold{w})$ 

MLE 估计 $\bold{w}$：$\hat{\bold{w}} = \arg\max\limits_{\bold{w}}L(\bold{w})$ 

最后使用 Newton-Raphson 方法解 $\nabla_{\bold{w}} \log L(\bold{w}) = 0$ 



### 线性判别分析 [Linear Discriminant Analysis (LDA)]

假设 $π_k = P(Y = k)$ 是先验概率，$f_k(\bold{x}) = P(\bold{X} = \bold{x}|Y = k)$ 是 $Y=k$ 这一类的密度函数

由贝叶斯定理：$P(Y|\bold{X} = \bold{x})\propto f_k(\bold{x}) π_k$ 

假设 $f_k(\bold{x})$ 是多元高斯分布，那么
$$
f_k(\bold{x})=\frac{1}{(2\pi)^{p/2}|\bold{\Sigma}_k|^{1/2}}e^{-\frac{1}{2}(\bold{x}-\mu_k)^T\bold{\Sigma}_k^{-1}(\bold{x}-\mu_k)}
$$
协方差矩阵是统一的，$\bold{\Sigma}_k=\bold{\Sigma}$ ，从对数角度来看，也就是
$$
\log\frac{P(Y=k|\bold{X}=\bold{x})}{P(Y=l|\bold{X}=\bold{x})}=\log\frac{\pi_k}{\pi_l}-\frac{1}{2}(\mu_k+\mu_l)^T\bold{\Sigma}^{-1}(\mu_k-\mu_l)+\bold{x}^T\bold{\Sigma}^{-1}(\mu_k-\mu_l)
$$
记 $\delta_k(\bold{x})= \log\pi_k-\frac{1}{2}\mu_k^T\bold{\Sigma}^{-1}\mu_k+\bold{x}^T\bold{\Sigma}^{-1}\mu_k$，那么 $\log\frac{P(Y=k|\bold{X}=\bold{x})}{P(Y=l|\bold{X}=\bold{x})}=\delta_k(\bold{x})-\delta_l(\bold{x})$ 

分类依据：$k^*=\arg\max_k\delta_k(\bold{x})$ 

对未知信息的样本的参数估计：

- $\hat{\pi_k}=N_k/N$，其中 $N = \sum\limits_{k=1}^K N_k$ 
- $\mu_k=\frac{1}{N_k}\sum\limits_{y_i=k}\bold{x_i}$ 
- $\bold{\hat{\Sigma}}=\frac{1}{N-K}\sum\limits_{k=1}^K \sum\limits_{y_i=k}(\bold{x_i}-\hat{\mu_k})(\bold{x_i}-\hat{\mu_k})^T$ 

对于二分类问题，LDA 将样本分类为类别 2 如果满足下列不等式：
$$
(\bold{x}-\frac{\hat{\mu}_1+\hat{\mu}_2}{2})^T\bold{\Sigma}^{-1}(\hat{\mu}_1-\hat{\mu}_2)+\log\frac{\hat{\pi}_2}{\hat{\pi}_1}>0
$$
判别的方向为 $\beta = \bold{\Sigma}^{-1}(\hat{\mu}_1-\hat{\mu}_2)$ ，贝叶斯错误分类比例为 $1-\Phi(\beta^T(\mu_2-\mu_1)/(\beta^T\bold{\Sigma}\beta)^{\frac{1}{2}})$，其中 $\Phi(x)$ 是正态分布的累计函数

下面是一个其他的判别分析：

<img src=".\\pic\\05_01.png" alt="" width="600">



### 神经网络 (Neural Network)

#### 深度学习 (Deep Learning)

深度学习是机器学习的一个子领域，试图模仿人类大脑使用神经元的工作

深度学习专注于使用几个隐藏层来构建人工神经网络（Artificial Neural Networks，ANN）

有各种各样的深度学习网络，如多层感知器（Multilayer Perceptron，MLP）、自动编码器（Autoencoders，AE）、卷积神经网络（Convolution Neural Network，CNN）、递归神经网络（Recurrent Neural Network，RNN）

近年来，软件硬件及大数据飞速发展，为深度学习的发展提供了良好的条件。还有研究表明，数据量更大时，深度学习的效果比传统机器学习算法更优

#### 感知器：正向传播 (The Perceptron : Forward Propagation)

<img src=".\\pic\\05_02.png" alt="" width="300">
$$
\hat{y}=g(w_0+\bold{X}^T\bold{W}), \quad where \quad \bold{X}= \begin{bmatrix}x_1 \\...\\x_m \end{bmatrix}, \bold{W}= \begin{bmatrix}w_1 \\...\\w_m \end{bmatrix}
$$
常见的激活函数 (Activation functions)

<img src=".\\pic\\05_03.png" alt="" width="600">

单层神经网络举例：

<img src=".\\pic\\05_04.png" alt="" width="350">
$$
z_i=w_{0,i}^{(1)}+\sum\limits_{j=1}^mx_jw_{j,i}^{(1)}, \quad \hat{y_i}=g(w_{0,i}^{(2)}+\sum\limits_{j=1}^{d_1}z_jw_{j,i}^{(2)})
$$

#### 神经网络算法

记 $C(\bold{W})=\frac{1}{n}\sum\limits_{i=1}^n \mathcal{L}(f(x^{(i)};\bold{W}),y^{(i)})$，需要求损失函数最小的参数 $\bold{W^*}=\arg\min\limits_{\bold{W}}C(\bold{W})$，其中 $\bold{W}=\{\bold{W^{(0)}},\bold{W^{(1)}},...\}$ 

使用梯度下降法 (Gradient Decent) 优化参数 $\bold{W}$，计算 $\frac{\partial C}{\partial \bold{W}}$ 

符号说明：

- $w^l_{jk}$ 是从第 $(l−1)$ 层第 $k$ 个神经元到第 $l$ 层第 $j$ 个神经元的权重
- $b^l_j = w_{j0}^l$ 是第 $l$ 层第 $j$ 个神经元的偏差 (bias)
- $a^l_j$ 表示第 $l$ 层第 $j$ 个神经元 $z_j^l$ 的激活，即 $a_j^l=g(z_j^l)=g\left(\sum\limits_k w_{jk}^la_k^{l-1}+b_j^l \right)$ 

4 个基本结论及推导：

<img src=".\\pic\\05_05.png" alt="" width="500">

<div align="center">
    <img src=".\\pic\\05_06.png" alt="" width="280">
    <img src=".\\pic\\05_07.png" alt="" width="280">
</div>

<div align="center">
    <img src=".\\pic\\05_08.png" alt="" width="280">
    <img src=".\\pic\\05_09.png" alt="" width="280">
</div>

#### 反向传播过程 (Back Propagation Procedure)

1. 输入 $x$：为输入层设置相应的激活值 $a^1$
2. 向前反馈：对于每个 $l=2,3,...,L$，计算 $z^l=w^la^{l-1}+b^l$ 和 $a^l=\sigma(z^l)$ 
3. 输出误差 $\delta^L$：计算 $\delta^L=\nabla_aC \bigodot\sigma'(z^l)$ 
4. 反向传播误差：对于每个 $l=L-1,L-2,...,2$，计算 $\delta^l=((w^{l+1})^T\delta^{l+1})\bigodot\sigma'(z^l)$ 
5. 输出：函数 $C$ 的输出梯度为 $\frac{\partial C}{\partial w_{jk}^l}=a_k^{l-1}\delta_k^l$ 和 $\frac{\partial C}{\partial b_j^l}=\delta_k^l$ 

#### 梯度下降算法 (Gradient Descent)

1. 随机初始化权值 $\sim\mathcal{N}(0,\sigma^2)$ 
2. 循环直到结果收敛
3. ​        计算梯度 $\frac{\partial J(\bold{W})}{\partial\bold{W}}$ 
4. ​        更新权值 $\bold{W} \leftarrow \bold{W}-\eta\frac{\partial J(\bold{W})}{\partial\bold{W}}$ 
5. 返回权值结果



### 支持向量机 (Support Vector Machine, SVM)

使用超平面划分数据以达到分类效果，最大化边距 (Margin)

#### 线性 SVM

训练数据：$\{(\bold{x_1}, y_1),...,(\bold{x_n},y_n)\}$，$y_i\in\{-1,+1\}$ 

超平面：$S=\bold{w}^T\bold{x}+b$ 

决策函数：$f(\bold{x})=sign(\bold{w}^T\bold{x}+b)$ 

点与超平面间的距离：$r_i=\frac{y_i(\bold{w}^T\bold{x}+b)}{\|\bold{w}\|_2}$ 

数据集与超平面间的边距：$\min\limits_i r_i$ 

最大化边距：$\max\limits_{\bold{w},b}\min\limits_{i} \frac{y_i(\bold{w}^T\bold{x}+b)}{\|\bold{w}\|_2}$ 

化为一下约束规划问题：
$$
\min\limits_{\bold{w},b}\frac{1}{2}\|\bold{w}\|_2^2, \quad s.t. \, y_i(\bold{w}^T\bold{x_i}+b) \ge1,i=1,...,n
$$
 这是一个具有线性约束的二次规划 (quadratical programming) 问题，计算复杂度为 $O(p^3)$，其中 $p$ 为维数

##### 使用拉格朗日乘子法 (Method of Lagrange Multipliers) 解决

假设 $\alpha_i \ge 0$ 是约束条件 $y_i(\bold{w}^T\bold{x_i}+b) \ge1$ 拉格朗日乘子因数，那么拉格朗日函数如下：
$$
L(\bold{w},b,\alpha)=\frac{1}{2}\|\bold{w}\|_2^2-\sum\limits_{i=1}^n\alpha_i[y_i(\bold{w}^T\bold{x_i}+b) -1]
$$
那么，
$$
\max\limits_{\alpha}L(\bold{w},b,\alpha)=
\begin{cases}
\frac{1}{2}\|\bold{w}\|_2^2, & y_i(\bold{w}^T\bold{x_i}+b) -1\ge0 \\
+\infty, & y_i(\bold{w}^T\bold{x_i}+b) -1< 0
\end{cases}
$$
那么问题转化为求 $\min\limits_{\bold{w},b}\max\limits_{\alpha}L(\bold{w},b,\alpha)$，对偶问题为 $\max\limits_{\alpha}\min\limits_{\bold{w},b}L(\bold{w},b,\alpha)$ 

先解 $\min\limits_{\bold{w},b}L(\bold{w},b,\alpha)$：
$$
\nabla_{\bold{w}}L=0 \Rightarrow \bold{w^*}=\sum\limits_i\alpha_iy_i\bold{x}_i \\
\frac{\partial L}{\partial b}=0 \Rightarrow \sum\limits_i\alpha_iy_i =0
$$
代入 $L$ 中：$L(\bold{w^*},b^*,\alpha)=\sum\limits_i\alpha_i-\frac{1}{2}\sum\limits_i\sum\limits_j\alpha_i\alpha_jy_iy_j(\bold{x}_i^T\bold{x}_j)$ 

问题转化为：
$$
\max\limits_{\alpha}\left[\frac{1}{2}\sum\limits_i\sum\limits_j\alpha_i\alpha_jy_iy_j(\bold{x}_i^T\bold{x}_j)-\sum\limits_i\alpha_i\right]\quad
s.t. \, \alpha_i\ge0,\sum\limits_i\alpha_iy_i =0,i=1,...,n
$$

##### KKT 条件

$$
\begin{cases}
\alpha_i^* \ge0 \\
y_i((\bold{w^*})^T\bold{x_i}+b^*) -1\ge0 \\
\alpha_i^* \left[ y_i((\bold{w^*})^T\bold{x_i}+b^*) -1 \right] = 0
\end{cases}
$$

支持向量的指标集 $S=\{i|\alpha_i>0\}$ 

$b=y_s-\bold{w}^T\bold{x}_s=y_s-\sum\limits_{i\in S}\alpha_iy_i\bold{x}_i^T\bold{x}_s$ 

更稳定的解：$b=\frac{1}{|S|}\sum\limits_{s\in S}\left(y_s-\sum\limits_{i\in S}\alpha_iy_i\bold{x}_i^T\bold{x}_s\right)$ 

##### SMO (Sequential Minimal Optimization) 算法

1. 选择离 KKT 条件最远的一对 $\alpha_i$ 和 $\alpha_j$ 
2. 假设其他参数固定不变，求解 $\alpha_i$ 和 $\alpha_j$ 
3. 更新 $\alpha_i$ 和 $\alpha_j$，返回第 1 步
4. 直到参数收敛，退出

计算复杂度为 $O(n^3)$ 

##### 软边距

当数据集不线性可分，那么可以引入一些松弛变量 $\xi_i\ge0$ ，将约束条件变为 $y_i(\bold{w}^T\bold{x_i}+b) \ge1-\xi_i$ 

那么优化问题为：
$$
\min\limits_{\bold{w},b}\frac{1}{2}\|\bold{w}\|_2^2+C\sum\limits_{i=1}^n\xi_i, \quad s.t. \, y_i(\bold{w}^T\bold{x_i}+b) \ge1-\xi_i,\xi_i\ge0,i=1,...,n
$$
对偶问题为：
$$
\max\limits_{\alpha}\left[\frac{1}{2}\sum\limits_i\sum\limits_j\alpha_i\alpha_jy_iy_j(\bold{x}_i^T\bold{x}_j)-\sum\limits_i\alpha_i\right]\quad
s.t. \, 0\le\alpha_i\le C,\sum\limits_i\alpha_iy_i =0,i=1,...,n
$$


#### 非线性 SVM

非线性 SVM 采用核函数 (Kernel function) 来替代 $\bold{x}_i$ 和 $\bold{x}_j$ 的内积，优化问题的对偶问题为：
$$
\max\limits_{\alpha}\left[\frac{1}{2}\sum\limits_i\sum\limits_j\alpha_i\alpha_jy_iy_jK(\bold{x}_i,\bold{x}_j)-\sum\limits_i\alpha_i\right]
$$
一般选用的核函数有以下几种

| Kernel     | Definition                                           | Parameters              |
| ---------- | ---------------------------------------------------- | ----------------------- |
| Polynomial | $(\bold{x}_1^T\bold{x}_2+1)^d$                       | $d$ 是正整数            |
| Gaussian   | $e^{-\frac{\|\bold{x}_1-\bold{x}_2\|^2}{2\delta^2}}$ | $\delta > 0$            |
| Laplacian  | $e^{-\frac{\|\bold{x}_1-\bold{x}_2\|}{2\delta^2}}$   | $\delta > 0$            |
| Fisher     | $tanh(\beta\bold{x}_1^T\bold{x}_2+\theta)$           | $\beta > 0, \theta < 0$ |



#### SVM 的优缺点

**优点**

- 在图像识别方面表现很好
- 易于使用核函数处理维度高的数据集
- 鲁棒性好，并易于推广到新数据集

**缺点**

- 对超高维数据处理能力不好
- 当样本量较大时，非线性 SVM 的计算效率较低
- 没有概率的参与所以可解释性较差





## 6. 集成学习 (Ensemble Learning)

多个弱学习模型（basic learner，可能是异源的 (heterogenous) ）可以提高学习效果

减少误分类率：假设一个分类器的误分类率为 $p$，选用 $N$ 个独立同分布的分类器，它们投票后的误分类率 $\sum\limits_{k>N/2} (^N_k)p^k(1-p)^{N-k}$ ，当 $N=5$，$p=0.1$ 时，误分类率小于 1%



### 集成学习的分类

- Bagging (引导聚合 (bootstrap aggregation) 的缩写)
  - 随机抽样：生成独立的模型，并对回归进行平均（对分类或回归进行多数投票）
  
  - 算法：
  
    输入：数据集 $D = \{(\bold{x_1},y_1),...,(\bold{x_n},y_n)\}$ 
  
    输出：叠加模型 $\hat{f}_{bag}=(x)$ 
  
    ```python
    for m = 1 to M:
    	Sample from D with replacement to obtain D_m
    	Train a model f_m(x) from the dataset Dm : for classification, f_m(x) returns a K-class 0-1 		vector ek ; for regression, it is just a value
    Compute bagging estimate f_bag(x) = 1/M \sum_{m=1}^{M} f_m(x) : for classification, make majority vote G_bag(x) = argmax_k f_k(x); for regression, just return the average value
    ```
  
  - 误差分析
  
    假设 $\{\hat{f}_m(x)\}_{m=1}^M$ 的方差均为 $\sigma^2(x)$ ，每一对之间的相关性为 $\rho(x)$ 
    $$
    Var(\hat{f}_{bag}(x))= \frac{1}{M^2}\left(\sum\limits_{m=1}^MVar(\hat{f}_m(x)) + \sum\limits_{t \ne m}Cov(\hat{f}_t(x)\hat{f}_m(x)) \right) = \rho(x)\sigma^2(x)+\frac{1-\rho(x)}{M}\sigma^2(x)
    $$
    方差小于原来的 $\sigma^2(x)$ 
  
- Boosting
  - 顺序训练：根据以往模型的误差对后续模型进行训练
  - 减少偏差
  - 比如 AdaBoost，GBDT
  
  <img src=".\\pic\\06_02.png" alt="" width="200">



### 随机森林 (Random Forest)

#### 决策树的缺点

- 受限于局部最优：贪婪算法使它停止在局部最优，因为它在每棵树的分裂中寻求最大的信息增益
- 决策边界：在每个分割中仅使用一个特征，决策边界平行于坐标轴
- 可描述性不好，稳定性不好

#### 随机森林算法

<img src=".\\pic\\06_01.png" alt="" width="500">

#### 随机森林的模型评估

Out-of-bag (OOB) errors 袋外样本误差：

对于每个观察结果 $(x_i,y_i)$，找到将其作为 OOB 样本的树：$\{\hat{T}_m(\bold{x}):(\bold{x}_i,y) \notin D_m\}$ 

使用这些树对这些观察结果进行分类，并将多数投票作为该观察结果的标签：$\hat{f}_{oob}(\bold{x}_i)=\arg\max\limits_{y\in\mathcal{Y}}\sum\limits_{m=1}^MI(\hat{f}_{m}(\bold{x}_i)=y)I(\bold{x}_i \notin D_m)$ 

计算错误分类样本的数量，并以该数量与样本总数的比值作为OOB误差：$Err_{oob}=\frac{1}{N}\sum\limits_{i=1}^N I(\hat{f}_{oob}(\bold{x}_i)\neq y_i)$ 

#### 随机森林的 Feature Importance

[链接](https://blog.csdn.net/weixin_34250709/article/details/93550808) 

#### 随机森林的优缺点

**优点**

- Bagging 或随机森林（RF）适用于具有高方差但低偏差的模型
- 对于非线性估计更好
- RF 适用于非常高维的数据，并且不需要做特征选择，因为 RF 赋予了 Feature Importance

- 易于进行并行计算

**缺点**

- 当样本规模大、噪声大或数据维度较低时容易过拟合
- 与单棵树相比，计算较慢
- 难以解释



### AdaBoost

#### 使用 Boost 拟合叠加模型

对于叠加模型 (Additive Model)：$f(x)=\sum\limits_{m=1}^M\beta_mb(x;\gamma_m)$ 

参数选择：$\min\limits_{\{\beta_m,\gamma_m\}}\sum\limits_{i=1}^NL(y_i,f(x))$ 

损失函数：平方误差或 likelihood-based loss

算法：

<img src=".\\pic\\06_03.png" alt="" width="500">

#### AdaBoost 算法

指数损失：$L(y,f(x))=exp(-yf(x))$ 

取 $b(x;\gamma_m)$ 为 $G(x)$ 

假设 $w_i^{(m)} = exp(−y_if_{m−1}(x_i))$ ，上述算法的步骤 2.1 为
$$
(\beta_m,G_m)=\arg\min\limits_{\beta,G}\sum\limits_{m=1}^M w_i^{(m)}exp(-\beta_my_iG(x_i))=\arg\min\limits_{\beta,G}\left[ \sum\limits_{y_i \neq G(x_i)} w_i^{(m)} (e^\beta-e^{-\beta})+e^{-\beta}\sum\limits_{m=1}^Mw_i^{(m)}\right]
$$
$G_m=\arg\min\limits_G \sum\limits_{i=1}^nw_i^{(m)}I(y_i \neq G(x_i))$ 

$\beta_m=\arg\min\limits_\beta [\epsilon_m(e^{\beta}-e^{-\beta})+e^{-\beta}]=\frac{1}{2}\log\frac{1-\epsilon_m}{\epsilon_m}$ ，其中 $\epsilon_m=\frac{\sum\limits_{i=1}^nw_i^{(m)}I(y_i \neq G(x_i))}{\sum\limits_{i=1}^nw_i^{(m)}}$ 为 weighted error rate

<img src=".\\pic\\06_04.png" alt="" width="500">

弱分类器的权重：分类器越好，其权重就越大

样本权重：每一步后重新加权，增加错误分类样本的权重

损失函数的选择：对于分类， exponential loss 也就是 $exp(-yf(x))$ 和  binomial negative log-likelihood (deviance) loss $\log(1 + exp(−2yf))$ 一样好；对于回归， squared error loss 不好，exponential loss 好，binomial deviance 更好

#### AdaBoost 优缺点

**优点**

- 与弱分类器相比，AdaBoost 提高了分类性能
- 弱分类器有许多选择：树、支持向量机、kNN等
- 只有一个需要调的参数 $M$： 弱分类器的数量
- 防止单一弱分类器（如复杂决策树）造成的过拟合

**缺点**

- 可解释性弱
- 在使用很差的弱分类器时容易过拟合
- 对异常值敏感
- 不易并行计算



### Gradient Boosting Decision Tree (GBDT)

#### Boosting Tree

使用分类树或回归树作为基础学习单元

$f_M(x)=\sum\limits_{m=1}^MT(x;\Theta_m)$，其中 $T(x;\Theta)=\sum\limits_{j=1}^J\gamma_jI(x\in R_j)$ ，参数集 $\Theta=\{R_j,\gamma_j\}_{j=1}^J$ 

对参数进行估计是一个组合优化问题：$\hat{\Theta}=\arg\min\limits_{\Theta}\sum\limits_{j=1}^J\sum\limits_{x_i\in R_j}L(y_i,\gamma_j)$ 

如果使用前向分步算法 (Forward Stagewise Algorithm)，$\hat{\Theta}_m=\arg\min\limits_{\Theta_m}\sum\limits_{i=1}^N L(y_i,f_{m-1}(x_i)+T(x_i;\Theta_m))$  

- $\hat{\gamma}_{jm}=\arg\min\limits_{\gamma_{jm}}\sum\limits_{x_i\in R_{jm}}L(y_i,f_{m-1}(x_i)+\gamma_{jm})$ 
- 但是计算 $R_{jm}$ 比在单颗树计算困难很多

#### GBDT

监督学习等同于优化问题：$\min\limits_{f}L(f)=\min\limits_f\sum\limits_{i=1}^N L(y_i,f(x_i))$ 

采用数值优化方法：$\bold{\hat{f}}=\arg\min\limits_{\bold{f}}L(\bold{f})$，其中 $\bold{f}=\{f(x_1),...,f(x_N)\}$ 

$\bold{f_M}=\sum\limits_{m=1}^M \bold{h_m}$，$\bold{f_0}=\bold{h_0}$ 是随机初始值

$\bold{f}_m=\bold{f}_{m-1}-\rho_m\bold{g}_m$ ，$\bold{h}_m=-\rho_m\bold{g}_m$ ，$g_{im}=\left[ \frac{\partial L(y_i,f(x_i))}{\partial f(x_i)} \right]_{f(x_i)=f_{m-1}(x_i)}$ 

GBDT 要找一个树 $T(x;\Theta_m)$ 满足 $\tilde{\Theta}_m=\arg\min\limits_{\Theta_m}\sum\limits_{i=1}^N (-g_{im}-T(x;\Theta_m))^2$ 

<img src=".\\pic\\06_05.png" alt="" width="500">

正则化的方法：

- 收缩 Shrinkage：步骤 2.4 变为 $f_m(x)=f_{m-1}(x)+\nu\sum\limits_{j=1}^{J_m}\gamma_{jm}I(x_i\in R_{jm})$ 
- 二次抽样 Subsampling：在每次迭代中，采样训练集的占比为 $η$ 的一部分，并使用子样本生成下一个树

#### Feature importance

<img src=".\\pic\\06_06.png" alt="" width="500">

#### GBDT 优缺点

**优点**

- 适用于所有回归问题
- 更适用于两分类，可适用于多分类问题（不建议）
- 具有多种非线性性，强表征性

**缺点**

- 顺序流程，不易于并行计算
- 计算复杂度高，不适合用于具有稀疏特征的高维问题



### XGBoost

Cost Function：$F(\Theta_m)=\sum\limits_{i=1}^N L(y_i,f_{m-1}(x_i)+T(x_i;\Theta_m))+R(\Theta_m)$ 

二阶泰勒展开得到 $F(\Theta_m)\approx \sum\limits_{i=1}^N\left[ L(y_i,f_{m-1}(x_i))+g_i^{(m)}T(x_i;\Theta_m)+\frac{1}{2}h_{ii}^{(m)}T(x_i;\Theta_m)^2\right] + R(\Theta_m)$ ，其中 $g_{i}^{(m)}=\left[ \frac{\partial L(y_i,f(x_i))}{\partial f(x_i)} \right]_{f(x_i)=f_{m-1}(x_i)}$ 是损失函数的梯度，$h_{ii}^{(m)}=\left[ \frac{\partial^2 L(y_i,f(x_i))}{\partial f(x_i)^2} \right]_{f(x_i)=f_{m-1}(x_i)}$ 是 损失函数的 Haisen 矩阵的对角线元素（非对角线全为0）

**惩罚项的选择**，以回归树为例

 设 $J_m$ 为叶子节点的数量（分区中的矩形数量），$γ_{jm}$ 为叶子节点 (区域) $R_{jm}$ 中的近似常数 (权重 $w$)

树的复杂度为 $\{\gamma_{jm}\}$ 的 $L^0$ 范数与 $L^2$ 范数的和：$R(\Theta_m)=\frac{1}{2}\lambda\sum\limits_{j=1}^{J_m}\gamma_{jm}+\mu J_m$ 

对 cost function 进行求解：

$F(\Theta_m)\approx \sum\limits_{i=1}^N L(y_i,f_{m-1}(x_i))+ \sum\limits_{j=1}^{J_m}\left[ (\sum\limits_{x_i\in R_{jm}}g_i^{(m)} )\gamma_{jm}+\frac{1}{2}(\sum\limits_{x_i\in R_{jm}}h_{ii}^{(m)}+\lambda)\gamma_{jm}^2 \right]+\mu J_m$

$F(\Theta_m)\approx\sum\limits_{j=1}^{J_m}[G_j^{(m)}\gamma_{jm}+\frac{1}{2}(H_j^{(m)}+\lambda)\gamma_{jm}^2]+\mu J_m + \text{constant}$ ，其中 $G_j^{(m)}=\sum\limits_{x_i\in R_{jm}}g_i^{(m)}$ ，$H_j^{(m)}=\sum\limits_{x_i\in R_{jm}}h_{ii}^{(m)}$ 

对 $\gamma_{jm}$ 求导可得：$\hat{\gamma}_{jm}=-\frac{G_j^{(m)}}{H_j^{(m)}+\lambda}$ 

化简：$F(\Theta_m)=-\frac{1}{2}\sum\limits_{j=1}^{J_m}\frac{(G_j^{(m)})^2}{H_j^{(m)}+\lambda}+\mu J_m+\text{constant}$ 

忽略常数项，我们得到了结构分数 (Structure Score)：$SS=-\frac{1}{2}\sum\limits_{j=1}^{J_m}\frac{(G_j^{(m)})^2}{H_j^{(m)}+\lambda}+\mu J_m$ ，它类似于信息增益：最小化结构分数会得到最好的树

损失函数：

- Square loss $L(y,f)=(y-f)^2$ ：$g_i^{(m)}=2(f_i-y_i)=2\times \text{residue}$ ，$h_{ii}^{(m)}=2$ 
- Logistic loss $L(y,f)=y\ln(1+e^{-f})+(1-y)\ln(1+e^f)$ ：$g_i^{(m)}=-y_i(1-\frac{1}{1+e^{-f_{m-1}(x_i)}}+(1-y_i)\frac{1}{1+e^{-f_{m-1}(x_i)}})=\text{Pred}-\text{Label}$，$h_{ii}^{(m)}=\frac{e^{-f_{m-1}(x_i)}}{(1+e^{-f_{m-1}(x_i)})^2}=\text{Pred}\times(1-\text{Pred})$  

#### 节点分割：贪心算法

当将一个节点分割为左 (L) 和右 (R) 子节点时，最大化 $Gain=\frac{1}{2}[\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}-\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}]-\gamma$ 

<img src=".\\pic\\06_07.png" alt="" width="500">



### 集成学习总结

- 集成方法具有单个模型的可叠加能力，具有更好的性能
- 易于推广到新的数据
- 噪声强时，容易过拟合（对噪声敏感）
- 计算密集型





## 7. 聚类 (Clustering)

也称为数据分割，将对象集合分组为子集或集群，得到的结果为每个集群中的对象比不同集群中的对象更相似

聚类不同于分类的点在于，它是**无监督学习**，没有输出或标签

聚类的最终目标：

- 在集群内样本相似性更高
- 在集群之间样本相似性更低

Cost Function：与输出无关，但与相似性有关

2 种类型的输入：

- n × n 相似度（不相似度）矩阵
  - 只有样本对之间的距离
  - 可能会丢失一些数据信息
- 具有 $d$ 个特征的原始数据 $X\in \mathbb{R}^{n \times d}$ 

聚类的过程：

- 数据预处理，特别是标准化预处理
- 得到相似矩阵
- 选择要运用聚类算法
- 确定集群的最佳数量

聚类算法：

- 分区聚类 (Partitional clustering)
  - K-means
  - K-Medoids
  - Spectral clustering
  - DBSCAN
- 分层聚类 (Hierarchical clustering)



### K-Means

将 n 个样本分组为 k 个集群，使每个样本属于最近的集群

数据集表示为 $\{x_i\}_{i=1}^n$，其中 $x_i\in\mathbb{R}^d$ 

第 $k$ 个集群 $C_k$ 的质心为 $c_k$，其中 $k=1,2,...,K$ 

K-Means 的大致思路是，$x_i$ 属于集群 $k$ 如果 $d(x_i,c_k) < d(x_i,c_m)$, $m \neq k$ ，其中 $d(x_i,x_j)$ 是不相似度函数，使质心位置良好，从而使每个样本到其质心之间的平均距离尽可能小

#### 转化为优化问题

聚类 $C:\{1,2,...,n\} \rightarrow \{1,2,...,k\}$ 
$$
T=\frac{1}{2}\sum\limits_{i=1}^n\sum\limits_{j=1}^n d(x_i,x_j)=\frac{1}{2}\sum\limits_{k=1}^K\sum\limits_{C(i)=k}\left( \sum\limits_{C(j)=k}d_{ij} + \sum\limits_{C(j)\neq k}d_{ij} \right)=W(C)+B(C)
$$
Loss Function：

- 聚类内点损失 $W(C)=\sum\limits_{k=1}^K\sum\limits_{C(i)=k}\sum\limits_{C(j)=k}d_{ij}$ 
- 聚类间点损失 $B(C)=\sum\limits_{k=1}^K\sum\limits_{C(i)=k}\sum\limits_{C(j)\neq k}d_{ij}$ 

最小化 $W(C)$ 相当于最大化 $B(C)$ 

#### 不相似度

相似矩阵：n × n 的相似矩阵

点对之间的不相似度：

- $d(x_i,x_j)=\sum\limits_{k=1}^pd_k(x_{ik},x_{jk})$ ，$d_k$ 可以取平方距离，绝对值距离等

- 加权平均：$d(x_i,x_j)=\sum\limits_{k=1}^pw_kd_k(x_{ik},x_{jk})$，其中 $\sum\limits_{k=1}^pw_k=1$ 

  设置 $w_k \sim 1/d_k$ ，$\overline{d_k}=\frac{1}{n^2}\sum\limits_{i=1}^n \sum\limits_{j=1}^n d_k(x_{ik},x_{jk})=2\hat{Var}(X_k)$ 将对所有列产生同等影响

- 基于相关性：$d(x_i,x_j) \propto 1-\rho(x_i,x_j)$ 

#### K-Means (as Central Voronoi Tessellation)

最小化 $W(C)$ 通常是不可行的，因为这是一个贪婪的算法，只适用于小的数据集，因此会将其转化为优化问题

取平方不相似度，即 $W(C)=\sum\limits_{k=1}^Kn_k \sum\limits_{c(i)=k}\|x_i-\overline{x}_k\|^2$ ，其中 $n_k$ 表示集群 k 的样本数量，$\overline{x}_k=\frac{1}{n_k}\sum\limits_{C(j)=k}x_j=\arg\min\limits_{m_k}\sum\limits_{C(j)=k}\|x_j-m_k\|^2$ 

那么，转化的优化问题为
$$
\min\limits_{C}W(C) \Leftrightarrow \min\limits_{C,m_k}\sum\limits_{k=1}^Kn_k\sum\limits_{C(i)=k}\|x_i-m_k\|^2
$$
轮换着迭代：

- 给定 $C$，解 $m_k$，$\Rightarrow \,\, m^*_k=\overline{x}_k$ 
- 给定 $m_k$，解 $C$，$\Rightarrow \,\, C(i)=\arg\min\limits_{1\leq k\leq K}\|x_i-m_k\|^2$ 

K-Means 的迭代结束条件是所有集群的质心都不再变化

各集群初始质心的选择：

- 随机猜测，选取猜测到的最小的 $W(C)$ 
- 基于其他聚类方法猜测 K-Means 的初始值

#### K 的选取

- 最小化贝叶斯信息准则 [Minimizing Bayesian Information Criterion (BIC)]

   $\text{BIC}(\mathcal{M}|\bold{X})=-2\log P(\bold{X}|\hat{\Theta},\mathcal{M})+p\log n$，其中 $\mathcal{M}$ 是模型本身，$\hat{\Theta}$ 是模型参数的最小似然估计，$P(\bold{X}|\hat{\Theta})$ 是似然函数，$p$ 是模型的参数个数，是控制对数似然和模型复杂度之间的权衡

- 最小描述长度 [Minimum Description Length (MDL)]

  从较大的 $K$ 开始，减小 $K$ 直到 $-\log P(\bold{X}|\hat{\Theta},\mathcal{M})-\log P(\Theta|\mathcal{M})$ 达到最小值，类似于最大后验估计 (MAP) 

- 基于高斯分布的假设

  从 $K = 1$ 开始，增加 $K$，直到每个簇中的点遵循高斯分布

#### K-Means 优缺点

**优点**

- 直观，易于实现
- 时间复杂度较低，为 $O(tnpK)$，其中 $t$ 为迭代次数

**缺点**

- 需要调参 $K$
- 强烈依赖于各集群质心的初始值
- 容易陷入局部最优
- 假设数据是球形的（数据经过了归一化预处理），难处理非球形数据
- 对异常值敏感

#### K-Means 的变体

##### Bisecting K-means

用于对各集群初始中心的选择的猜测，核心思想是依次将最坏的集群划分为两个子集群

算法过程：

1. 初始化将所有数据放在一个集群中

2. 循环：

   2.1 选择使集群内点最大分散 $\sum\limits_{C(i)=k}\sum\limits_{C(j)=k}\|x_i-x_j\|^2$ 的集群 $k$ 

   2.2 使用 2-means 将集群 $k$ 划分为两个子集群，随机初始猜测两个中心

   2.3 重复步骤 2.2 $p$ 次，选择能最小化集群点分散的最佳集群对

3. 有 $K$ 个集群时停止，或在获得满意集群结果的时候停止

#####  K-medoids

为克服异常值的影响而发明，可以处理更一般类型的数据，核心思想为每个集群的中心点仅限于分配给该集群的观测值之一 ~~（不是人话啊）~~ 

轮换迭代：

- 给定 $C$，解 $m_k=x_{i_k^*}$ 使得集群中点的分散指标最小：$i_k^*=\arg\min\limits_{\{i:C(i)=k\}} \sum\limits_{C(j)=k}d(x_i,x_j)$ 
- 给定 $m_k$，求解 $C$：$C(i)=\arg\min\limits_{1\leq k\leq K}d(x_i,m_k)$ 

**比 K-Means 健壮性更强** 

在轮换迭代的第 1 个步骤中，时间复杂度为 $O(n_k^2)$，比 K-Means 的 $O(n_k)$ 更高

##### 其他变种

- K-Median：使用曼哈顿距离（$L^1$-距离）代替；中心不是平均值，而是中位数
- K-Means++：选择彼此远离的初始中心
- Rough-set-based K-means：每个样本都可以被分配到多个集群中



### 分层群聚 (Hierarchical Clustering)

在不同的层次结构中进行聚类，生成树状结构

两种方式：

- 凝聚聚类 Agglomerate clustering：自下而上
- 分散聚类 Divisive clustering：自上而下

不足之处是，一旦合并或分割，该操作就不能被修改

#### 凝聚聚类

给定 $n$ 个样本和不相似度矩阵，执行以下步骤

1. 让每个观测结果都代表一个单例集群
2. 将两个最近的集群合并为一个集群
3. 计算新的不相似度矩阵（两个集群之间的差异）
4. 重复步骤 2 和步骤 3，直到将所有样本都合并到一个集群中

计算组间不相似度的三种方法：

- Single linkage：最大相似或最不相似 $d_{SL}(C_i,C_j)=\min\limits_{x\in C_i,y\in C_j} d(x,y)$ 
- Complete linkage：最小相似性或最大差异性 $d_{CL}(C_i,C_j)=\max\limits_{x\in C_i,y\in C_j} d(x,y)$ 
- Average linkage：平均相似度或不相似度 $d_{AL}(C_i,C_j)=\frac{1}{|C_i||C_j|} \sum\limits_{x\in C_i, y\in C_j} d(x,y)$ 

凝聚聚类算法：

<img src=".\\pic\\07_01.png" alt="" width="500">



#### 分散聚类

分散聚类算法

<img src=".\\pic\\07_02.png" alt="" width="500">

#### 分层聚类优缺点

**优点**

- 分层聚类可以一次性计算出整个聚类过程的树状结构
- SL 和 CL 对异常值很敏感，而 AL 则给出了一个妥协

**缺点**

- 计算密集
- 一旦一个样本被错误地分组到一个分支中，无论如何对树设置参数或阈值，它永远会在与该分支对应的集群中



### DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

分层聚类和 K-means 聚类的局限性：倾向于发现凸集群

基于密度的聚类：寻找由低密度区域分隔的高密度区域，可以发现任何形状的集群

#### 概念

- Core point：在其 $\epsilon$-邻域内的样本点数量 ≥ MinPts
- Boundary point：在某些核心点的 $\epsilon$-邻域的边界上，且它本身不是核心点
- Noise point：不是核心点也不是边界点，位于稀疏区域
-  $\epsilon$-邻域：$N_{\epsilon}(x_i)=\{x_j\in D|d(x_i,x_j)\leq\epsilon\}$ 
- Directly density-reachable：若 $x_j\in N_{\epsilon}(x_i)$，且 $x_i$ 是核心点，则称 $x_j$ 从 $x_i$ 直接密度可达
- Density-reachable：对于 $x_i$ 和 $x_j$，若存在 $p_1,p_2,...,p_m$ 使得 $p_1=x_i,p_m=x_j$，$p_{k+1}$ 从 $p_k$ 直接密度可达，那么称 $x_j$ 从 $x_i$ 密度可达
- Density-connected：若存在 $p$ 使得 $x_i$ 和 $x_j$ 都从 $p$ 密度可达，那么称 $x_i$ 和 $x_j$ 密度连接

 #### DBSCAN算法

<img src=".\\pic\\07_03.png" alt="" width="500">

算法的时间复杂度为 $O(nt)$，$t$ 为在 $\epsilon$-邻域内搜寻目标的时间，最坏情况下为 $O(n^2)$ 

在低维数据集中，可以使用 KD-Tree 优化到 $O(n\log n)$ 

#### DBSCAN 优缺点

**优点** 

- 能快速地实现聚类
- 能更好的处理噪音点
- 对任何形状的集群都有效

**缺点** 

- 需占用大量内存
- 当密度分布不均匀且集群间距较大时，性能不佳

#### DBSCAN vs K-Means

| DBSCAN                                                 | K-Means                          |
| ------------------------------------------------------ | -------------------------------- |
| 聚类结果并不是对原始数据集的完全划分（噪声点被排除了） | 聚类结果是对原始数据集的完整划分 |
| 可以处理任何形状和大小的集群吗                         | 聚类的集群几乎是球形的           |
| 可处理噪声点和离群值                                   | 对离群值敏感                     |
| 对密度的界定很重要                                     | 集群中心的选择很重要             |
| 在处理高维数据时效率低下                               | 在处理高维数据时效率较好         |
| 对样本分布没有隐含的假设                               | 这些样本隐式地遵循高斯分布的假设 |



### 最大期望算法 [Expectation-Maximization (EM) Algorithm]

省略，考试对这一部分不做要求



### 谱聚类 (Spectral Clustering)

**相似性图** 

- $\epsilon$-邻域图： $v_i$ 和 $v_j$ 相连如果 $d(v_i,v_j)<\epsilon$ ，无权图，$\epsilon \sim (\log n/n)^p$，需要调参 $\epsilon$ 
- k 近邻图：如果 $v_j$ 是 $v_i$ 的 k 近邻之一，则连接 $v_i$ 到 $v_j$ ，有向图；如果 $v_i$ 和 $v_j$ 是彼此的 k 近邻之间，则连接 $v_i$ 和 $v_j$ ，相互 k 最近邻图，无向；$k\sim \log n$ 
- 全连通图：将所有具有正相似性的点连接起来；建模局部邻域关系；高斯相似度函数 $s(x_i,x_j) = exp(−\|x_i−x_j\|^ 2/(2σ^2))$，其中 $σ$ 控制邻域的宽度；邻接矩阵不是稀疏的；$σ\sim \epsilon$ 

**拉普拉斯图**

- 非归一化拉普拉斯图：$L=D-W$ 
  - 有 $\bold{1}$ 作为特征向量对应 0 特征值
  - 对称和正定：$\bold{f}^T\bold{L}\bold{f}=\frac{1}{2}\sum_{i,j}w_{ij}(f_i-f_j)^2$ 
  - 非负的、实值的特征值：$0=\lambda_1\le \lambda_2\le...\le\lambda_n$ 
  - 特征值 0 的特征空间由向量 $\bold{1}_{A_1}$，…，$\bold{1}_{A_k}$ 张成，其中 $A_1$，...，$A_k$ 是图中的 $k$ 个连通分量
- 归一化拉普拉斯图
  - 对称拉普拉斯函数：$L_{sym}=D^{-1/2}LD^{-1/2}$ 
  - 随机游走拉普拉斯函数：$L_{rm}=D^{-1}L$ 
  - 两者都有与 $L$ 相似的性质

#### 谱聚类

**Graph cut**：将 $G$ 分为 $K$ 组 $A_1$，…，$A_K$，其中 $A_i⊂V$，这相当于最小化图切割函数 $cut(A_1,…,A_K)=\frac{1}{2}\sum\limits_{k=1}^KW(A_k,\overline{A_k})$，其中 $W(A,B)=\sum_{i∈A,j∈B}w_{ij}$。简单的解由一个单例及其补组成。

**RatioCut**：$RatioCut(A_1,…,A_K)=\frac{1}{2}\sum\limits_{k=1}^K\frac{W(A_k,\overline{A_k})}{|A_k|}$，其中 $|A|$ 是 $A$ 中的点的数量

**Normalized cut**：$Ncut(A_1,…,A_K)=\frac{1}{2}\sum\limits_{k=1}^K\frac{W(A_k,\overline{A_k})}{vol(A_k)}$，其中 $vol(A_k)=\sum_{i\in A}d_i$，这个计算是 NP 困难的

<div align="center">
    <img src=".\\pic\\07_05.png" alt="" width="350">
    <img src=".\\pic\\07_06.png" alt="" width="350">
</div>

谱聚类算法：

<div align="center">
    <img src=".\\pic\\07_07.png" alt="" width="450">
</div>



### 聚类模型评估

外部指标：通过真实标签或比较两个集群得到，比如 Purity，Jaccard coefficient and Rand index，Mutual information

内部指标：不通过外部信息得到，基于集群内相似性和集群间距离，比如 Davies-Bouldin index (DBI)，Silhouette coefficient (SC)

##### Purity

$n_{ij}$ 表示标签是 $j$ 但聚类到集群 $i$ 的样本个数

$n_i=\sum\limits_{j=1}^C n_{ij}$ 是集群 $i$ 的总样本数

$p_{ij}=n_{ij}/n_i$ 是在集群 $i$ 中的概率分布

集群 $i$ 的纯度为 $p_i=\max\limits_j p_{ij}$ 

总纯度定义为 $\sum\limits_i \frac{n_i}{n}p_i$ 

##### Confusion Matrix

|                     | Same Cluster              | Different Cluster         |
| ------------------- | ------------------------- | ------------------------- |
| **Same Class**      | SS (True Positive or TP)  | DS (False Negative or FN) |
| **Different Class** | SD (False Positive or FP) | DD (True Negative or TN)  |

##### Jaccard Coefficient

$$
JC=\frac{SS}{SS+SD+DS}\in [0,1]
$$

##### Rand index

$$
RI=\frac{SS+DD}{SS+SD+DS+DD}\in[0,1]
$$

##### Mutual Information (简略)

[维基链接](https://en.wikipedia.org/wiki/Mutual_information)

<img src=".\\pic\\07_04.png" alt="" width="500">

##### Davies-Bouldin Index

DBI 测量集群内发散性和集群间距离
$$
DBI=\frac{1}{k}\sum\limits_{i=1}^k\max\limits_{j\neq i}\left( \frac{div(c_i)+div(c_j)}{d(\mu_i,\mu_j)} \right)
$$
其中 $div(c_i)$ 表示集群 $c_i$ 内样本的平均距离，$\mu_i$ 是 $c_i$ 的质心

DBI 越小说明聚类效果越好

##### Silhouette Coefficient

$$
SC=\frac{b_i-a_i}{\max(a_i,b_i)}\in[-1,1]
$$

其中，$a_i$ 是第 $i$ 个样本与同一集群中所有其他样本之间的平均距离，$b_i$ 是从第 $i$ 个样本到其他集群的最小距离

SC 越大，说明聚类效果越好





## 8. 降维 Dimensionality Reduction

$f:\mathcal{X} \rightarrow \mathcal{Y}$，其中 $dim\mathcal{X} > dim\mathcal{Y}$ 

降维的原因：

- 在实际应用中，样本的数量是有限的
- 由于高维数据的稀疏性，很容易进行过拟合
- 很难训练一个好的模型来对边界数据 (在高维更多) 进行分类

降维能做什么：

- 数据压缩 (Data compression)
- 去噪声
- 通过映射和特征选择来进行特征提取 (比如 LASSO)
- 降低空间和时间的复杂性，从而需要更少的参数和更小的计算能力
- 可视化

分类：

- 线性降维
  - Principal component analysis (PCA)
  - Linear discriminant analysis (LDA)
  - Independent component analysis (ICA)
- 非线性降维
  - Kernel based methods (Kernel PCA)
  - Manifold learning (ISOMAP, Locally Linear Embedding (LLE), Multidimensional scaling (MDS), t-SNE)



### PCA

方差：$Var(X)=\mathbb{E}(X-\mathbb{E}X)^2$ 

样本方差：$S^2=\frac{1}{n-1}\sum\limits_{i=1}^n(x_i-\overline{x})^2$ 

相关系数：$Cov(X,Y)=\mathbb{E}(X-\mathbb{E}X)(Y-\mathbb{E}Y)$ 

样本相关系数：$C=\frac{1}{n-1}\sum\limits_{i=1}^n(x_i-\overline{x})(y_i-\overline{y})$ 

若 $X=(\bold{x_1},...,\bold{x_n})^T\in\mathbb{R}^{n\times p}$ 是样本矩阵，$C=\frac{1}{n-1}(X-\bold{1_n}\bold{\overline{x}}^T)^T(X-\bold{1_n}\bold{\overline{x}}^T)=\frac{1}{n-1}(X-\frac{1}{n}\bold{1_n}\bold{1_n}^TX)^T(X-\frac{1}{n}\bold{1_n}\bold{1_n}^TX)=\frac{1}{n-1}X^TJX$ ，其中 $J=\bold{I_n}-\frac{1}{n}\bold{1_n}\bold{1_n}^T$ 是一个秩为 $n-1$ 的投影矩阵

#### PCA

- PCA 通过使用正交变换，将一组强相关变量转换为另一组（通常要小得多）的弱相关变量
- 这些新的变量被称为主成分 (principal components)
- 新的变量集是原始变量的线性组合，其方差信息被尽可能地继承
- 是无监督学习

**可视化解释**

假设一组二维数据遵循高斯分布 (但不限于高斯分布！)，通过采用较大方差（数据变异性较大）的方向，成功地降低到一维

主轴的方向比其他方向包含更多的信息，因为较小的方差表明变量包含的信息几乎相同

<div align="center">
    <img src=".\\pic\\08_01.png" alt="" width="350">
</div>

#### 样本协方差矩阵的特征分解

$\{e_i\}_{i=1}^p$ 是欧几里得空间的标准基，想找到另一组正交基 $\{\tilde{e_i}\}_{i=1}^p$ 使得随机向量 $v = \sum\limits_{i=1}^px_ie_i$ 可以在新的基中被表示 $v = \sum\limits_{i=1}^p\tilde{x_i}\tilde{e_i}$，且 $Var(\tilde{x_1})\ge...\ge Var(\tilde{x_n})$ 且对于 $i \neq j$ 有 $Cov({\tilde{x_i},\tilde{x_j}})\approx0$ 

通过线性代数，由线性变换得到坐标变换 $(\tilde{e_1},...,\tilde{e_p})=(e_1,...,e_p)W$，其中 $W\in\mathbb{R}^{p\times p}$ ，并对分量系数进行了相应的变换 $x=W\tilde{x}$ 

假设我们有 $n$ 个中心化的样本 $\{x_i\}_{i=1}^n$ 且 $\frac{1}{n}\sum\limits_{i=1}^nx_i=\bold{0_p}$ 
$$
X^T=(x_1,...,x_n)=W(\tilde{x_1},...,\tilde{x_n})=W\tilde{X}^T \\
Cov(X)=\frac{1}{n-1}X^TX \\
Cov(\tilde{X})=\frac{1}{n-1}\tilde{X}^T\tilde{X}=\frac{1}{n-1}W^TX^TXW=W^TCov(X)W
$$
它的对角线是 $Var(\tilde{x_1}),...,Var(\tilde{x_p})$ ，非对角线是 $\tilde{x_i}$ 和 $\tilde{x_j}$ 协方差

需要 $Cov(\tilde{X})$ 几乎是对角线的，且对角线项递减，相当于进行特征分解：$Cov(X)=Odiag(\lambda_1,...,\lambda_p)O^T$，$O\in\mathbb{R}^{p\times p}$ 是正交矩阵且 $\lambda_1\ge...\ge\lambda_p\ge0$ 并让 $W = O$ 

**解释：**

- 转换后的变量中的方差：$Var(\tilde{x_i})=\lambda_i$，$Cov(X)$ 的特征值
- 新的基，$W = O$ 的每一列
- 百分比 $\frac{\lambda_i}{\sum\lambda_i}$ 代表新变量 $\tilde{x_i}$ 的重要性
- 对于任意向量 $x\in\mathbb{R}^p$ ，对应的 $r$ 个主成分是这样表示的 $w_1^Tx,...,w_r^Tx$ 

**从 Best Reconstruction 的角度**

<div align="center">
    <img src=".\\pic\\08_02.png" alt="" width="500">
</div>

#### PCA 算法

<div align="center">
    <img src=".\\pic\\08_03.png" alt="" width="500">
</div>

应用实例：

<div align="center">
    <img src=".\\pic\\08_04.png" alt="" width="350">
    <img src=".\\pic\\08_05.png" alt="" width="350">
</div>


### LDA

LDA 属于监督学习，在标签的基础上进行线性投影，以最大限度地提高低维类间点分散性（可变性）

每个类中的样本数量为 $n_k$，总样本数量为 $n$ 

第 $k$ 类样本的平均值为 $\mu_k=\frac{1}{n_k}\sum\limits_{i:x_i\in C_k}x_i$，所有样本的平均值为 $\mu$ 

在投影之前，类间点散度为 $S_b=\sum\limits_{k=1}^K\frac{n_k}{n}(\mu_k-\mu)(\mu_k-\mu)^T$ ，投影后，投影矩阵为 $W_r\in\mathbb{R}^{p\times r}$ ，类间点散度为 $\tilde{S_b}=W_r^TS_bW_r$ 

在投影之前，每个类 $C_k$ 的类内点散度 (方差) 为 $S_k=\frac{1}{n_k}\sum\limits_{i:x_i\in C_k}(x_i-\mu_k)(x_i-\mu_k)^T$ ，因此，类内点总散度是 $S_w=\sum\limits_{k=1}^K\frac{n_k}{n}S_k$ 

投影后，每个类 $C_k$ 的类内点散度 (方差) 为 $\tilde{S_k}=W_r^TS_kW_r$ ，类内点总散度是 $\tilde{S_w}=W_r^TS_kW_r$  

LDA 转化为优化问题即为，要找到一个 $W_r$，使得类间点散度 $\tilde{S_b}$ 最大而类内点散度 $\tilde{S_w}$ 最小，即
$$
\max\limits_{w}J(w)=\frac{w^TS_bw}{w^TS_ww}
$$
这和以下等价
$$
\max\limits_{w}J_b(w)=w^TS_bw,\ \ \ \text{subject}\ \text{to}\ w^TS_ww=1
$$
使用拉格朗日乘子法，定义 $L(w,\lambda)=w^TS_bw-\lambda(w^TS_ww-1)$ 
$$
\nabla_wL=2S_b-2\lambda S_ww=0\ \ \ \ \Rightarrow\ \ \ \ S_w^{-1}S_bw=\lambda w
$$
最优解的方向是 $S_w^{-1}S_b$ 的特征向量

应用实例：

<div align="center">
    <img src=".\\pic\\08_06.png" alt="" width="350">
    <img src=".\\pic\\08_07.png" alt="" width="350">
</div>



### PCA 与 LDA 对比

- PCA
  - 从样本协方差矩阵出发，找到方差最大的方向
  - 无监督学习，作为训练前的步骤，必须与其他学习方法相结合
- LDA
  - 利用标签，找到投影，使之后分类变得更加明显
  - 监督学习，可以用作分类或与其他学习方法相结合



### 非线性降维

考试不涉及，因此省略了 QAQ
