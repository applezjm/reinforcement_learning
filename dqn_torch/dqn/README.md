dqn
===
原论文地址：https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf
___
### 算法主要思想
#### 1.主要方法与origin_dqn大同小异。
#### 2.由于origin_dqn中评估和采样的网络为一个，等于每次更新eval的时候target也会更新，非常容易导致参数不收敛。因此dqn中将eval_net和target_net拆分成两个，target_net固定不动，每C步用eval_net更新一次。
![image](https://github.com/applezjm/reinforcement_learning/blob/master/dqn/dqn_algorithm.png)
