# 随机过程理论

设目标等级为 $n$ , 给定转移矩阵 $A_{(n+1)\times (n+1)}$ (下标从 $0$ 开始), 第 $i$ 行 $j$ 列元素表示强化一次 +i 强化等级的物品后, 得到 +j 等级的物品.

显然, 每行元素和为 $1$ . 注意 $A_{n,n}=1$ , 即+n强化等级为吸收态。

若初始强化等级为+0, 经过恰好 $k$ 次强化后, 得到各个强化等级的概率分布为:

$$
(1, 0, 0, \cdots , 0, 0)_{1\times(n+1)}\times A^k.
$$

于是在经过k次强化后，得到目标强化等级物品的概率为 $(A^k)_{0,n}$ .

# 未证明的猜想

由无记忆性, 在给定强化次数以内, 得到成品的概率近似服从指数分布(几何分布).

因此, 按照期望消耗的数量准备材料, 最终能够得到成品的概率约为 $1-\frac 1 e \approx 0.632$ .


