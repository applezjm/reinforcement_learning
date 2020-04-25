double_dqn
===
原论文地址：https://arxiv.org/pdf/1509.06461v3.pdf
___
### 算法主要思想
#### 1.在target_net上每次action选取都用argmax方法，会导致学习的over-estimation。
> over-estimation原因可以参考 https://zhuanlan.zhihu.com/p/97853300  
> over-estimation确实普遍存在，且对最终效果有负面影响。对于未采样的点，学习的可能很差，而argmax则放大了这个差的效果。
#### 2.将target_net中的对max_action的选取替换成eval_net中的max_action,double_dqn是无偏估计（待看）
