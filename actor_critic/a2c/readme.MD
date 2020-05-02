(Asynchronous) Advantage Actor Critic
===

## Policy Gradient
参考：https://zhuanlan.zhihu.com/p/75174892  
##### J = E(G * ∑log(p(a|s,theta)))
其中最原始的J中的G是时间0-t累积reward（定值），∑符号在G外，由此计算出来的策略梯度不存在偏差，但是由于需要累积多步的回报，因此方差会很大。  
因此用Q(s,a)替代，并把∑挪到外面，方便迭代。其优点是方差小，但是这三种方法中都用到了逼近方法，因此计算出来的策略梯度都存在偏差。  
参考：https://zhuanlan.zhihu.com/p/62100741

## Advantage Actor Critic
##### 用value的TD_ERROR=r+V(S(t+1))-v(St)替代G  
可以减去baseline的理由参考：https://zhuanlan.zhihu.com/p/98506549

##### loss = loss_policy + loss_value + loss_entropy
loss_policy:actor部分loss  
loss_value:critic部分loss
loss_entropy:actor输出prob的entropy，增大熵从而增加动作的随机性以便exploration

## Asynchronous Advantage Actor Critic
两个优势：减少收敛时间，增加鲁棒性（待商榷）  
原论文中提到Asynchronous也可用于dqn系算法，直接替代experience-replay（多个actor已经可以保证数据的不相关性和非连续）
