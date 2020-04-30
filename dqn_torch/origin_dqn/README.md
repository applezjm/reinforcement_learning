origin_dqn
===
原论文地址：https://arxiv.org/abs/1312.5602
___
### 算法主要思想
#### 1.model-free off-policy(采样和评估策略不一致)algorithm
#### 2.用NN去拟合Q(s,a)值，对比sarsa不需要手工选取特征
#### 3.增加replay buffer,打破数据连续性，减少学习的方差，防止陷入局部最优。
#### 4.论文提出两个问题
> 数据稀疏、嘈杂、延迟（数千步后才有结果）  
> 经验有重要性大小  
#### 5.原论文算法迭代流程：
![image](https://github.com/applezjm/reinforcement_learning/blob/master/origin_dqn/image.png)
