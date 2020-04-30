dueling_dqn
===
原文地址：https://arxiv.org/abs/1511.06581
___
### 网络结构创新：将Q_net拆分成V_net + A_net
#### Q(s,a) = V(s) + A(s,a)  ，其中A(s,a)=A(s,a)-meanA(s,a)
##### 拆分成两个网络的理由  
原文中提到 “the dueling architecture can learn which states are (or are not) valuable, without having to learn the effect
of each action for each state. This is particularly useful in states where its actions do not affect the environment in
any relevant way.”  
对于大部分state来说，action选取无关重要。  
本来Q(s,a)对于每个<s,a>更新，更新速度较慢。而对于V(s)的更新可以对Q(s,a)中的s更新，更新速度较快。    
##### 用A(s,a)=A(s,a)-meanA(s,a)的理由  
假如只用原始的进行更新，会没有“unidentifiable”，也就是无法得到唯一的V,A。加了以后V就代表Q（s，a*）,具有实际含义。 详见：https://zhuanlan.zhihu.com/p/66069958   
增加-mean操作后，相当于增加了一个restriction，会加速网络收敛速度。见：https://www.zhihu.com/question/347276055
  
![image](https://github.com/applezjm/reinforcement_learning/blob/master/dueling_dqn/image.png)

