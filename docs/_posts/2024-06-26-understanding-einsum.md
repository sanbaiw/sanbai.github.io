---
layout: post
title:  "理解 einsum ✖️➕"
date: 2024-06-01 14:04:20 +0800
categories: pytorch
---
## einsum
Einstein summation 是爱因斯坦发明的一种矩阵运算标记, 旨在简化 tensor 运算表达式的书写.
![pic-einsum-meme](/assets/images/pic-einsum-meme.png)

比如两个 tensor A, B 的乘法,可以表示成 `ij, jk -> ik`, 式形统一简洁直观. 起初我了解到的规则是:
> 1. 输入中重复的字母代表这些 dim 相乘;
> 2. 输出中省略的字母代表这在这些 dim 上求和;

起初一看很 make sense, 但是这个规则很难解释其他场景, 比如 `ii -> i`.
我们可以尝试另一种视角, 从代码的角度理解 einsum.

## index 其实是 iterator
把输入中的字母 (index) 看作 Python 的 iterator: 
> 1.`i` 迭代器返回 row (dim=0) 的 index, `k` 返回 col 的 index;
> 2. 相同字母代表迭代器返回的数值相同

接下来看几个例子
#### - ij, jk -> ik: 矩阵乘法
> `i` 在 A 的行上迭代, `j` 在 A 的列上迭代
> `j, k` 分别在 B 的行列迭代
> `->` 代表输出
> `i, k` 是结果 C 的行和列迭代器
C 中的每个位置 `(i, k)` 代表 A 的 `i` 行和 B `k` 列点积

用伪代码表示
```python
for i in A.rows:
    for k in B.cols:
        for j in A.rows[i], B.cols[k]:
            C[i, k] += A[i, j] * B[j, k]
```
结果是 A, B 矩阵乘法

#### - ij, ij -> ij: 元素相乘
```python
for i in A.rows, B.rows:
    for j in A.cols, B.cols:
        C[i, j] = A[i, j] * B[i, j]
```
结果是 A 与 B 的元素乘法 (elementwise - multiplication)


#### - ik, jk -> ij: 逐行点积
```python
for i in A.rows:
    for j in B.rows:
        for k in A.rows[i], B.rows[j]:
            C[i, j] += A[i, k] * B[j, k]
```
结果是 A 与 B 逐行点积


## 更多例子
einsum 可以用于单个 tensor
#### - ii -> i: 对角线上的元素
两个 i 是作用在 A 行和列的相同迭代器
```python
for i in A.rows:
    C[i] = A.rows[i][i]
```
所以结果 C 的每个元素是 A 中行列 index 相同的元素: A 对角线上的元素

类似的 `ii ->`: 对角线上的元素再求和, 结果是 A 的迹 (trace)

#### - ij -> i: 逐行求和
```python
for i in A.rows:
    for j in A.rows[i]:
        C[i] += A[i, j]
```
A 逐行求和

类似的 `ij ->` 所有元素求和

#### - ik, jk -> ijk: 逐行相乘
A 中的每一行和 B 中的每一行元素相乘, 输出一个 rank 3 tensor
```python
for i A.rows:
    for j in B.rows:
        for k in A.rows[i], B.rows[j]:
            C[i, j] = A[i, k] * B[j, k] # 是一个 vecotr 
```