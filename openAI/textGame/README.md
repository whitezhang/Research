# TextGame Report

### Q-Learning VS. SARSA
- Q-Learning: Q[s,a] ←Q[s,a] + α(r+ γmax Q[s',a'] - Q[s,a])
- SARSA: Q[s,a] ←Q[s,a] + α(r+ γ Q[s',a'] - Q[s,a])

Off-policy Q-Learning learns action values relative to greedy policy, while on-policy SARSA does it relative to the policy it follows.

#### Experiments
From the experiments, SARSA is not guaranteed to converge. The Q-Value computed by SARSA can not always give a path leading to the target point, which is not stable. By studying the specific case, the loop exists in some Q-Value tables. This is caused by uneven random action. In other words, the average weight of Q-Value is not guaranteed in SARSA progress. (This is only a guess)

- Wrong(From stackoverflow)??? Q-Learning tends to converge a little slower, but has the capability to continue learning while changing policies. Also, Q-Learning is not guaranteed to converge when combined with linear approximation.

The value function is different as follows:
```
Q-Learning: Q(st+1, at+1) = max Q(st+1, a)
SARSA: Q(st+1, at+1) = e·mean Q(st+1, a) + (1-e)max Q(st+1, a)
```
