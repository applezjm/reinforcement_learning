(Asynchronous) Advantage Actor Critic
===

## Policy Gradient
参考：https://zhuanlan.zhihu.com/p/75174892  
##### J = E(G * ∑log(p(a|s,theta)))
其中最原始的J中的G是时间0-t累积reward（定值），∑符号在G外，由此计算出来的策略梯度不存在偏差，但是由于需要累积多步的回报，因此方差会很大。  
因此用Q(s,a)替代，并把∑挪到外面，方便迭代。其优点是方差小，但是这三种方法中都用到了逼近方法，因此计算出来的策略梯度都存在偏差。  
参考：https://zhuanlan.zhihu.com/p/62100741
