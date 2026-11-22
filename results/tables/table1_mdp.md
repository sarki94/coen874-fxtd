| Element                     | Specification                                                                           |
|:----------------------------|:----------------------------------------------------------------------------------------|
| Environment ID              | FrozenLake-v1 (map_name=8x8, is_slippery=True)                                          |
| Gymnasium version           | 1.4.0                                                                                   |
| States S                    | 64 discrete cells, s = 8*row + col; start s0 = 0; 10 holes; goal s = 63                 |
| Actions A                   | 4: 0 left, 1 down, 2 right, 3 up                                                        |
| Transitions P(s'|s,a)       | intended direction 1/3, each perpendicular direction 1/3; walls keep the agent in place |
| Reward R(s,a,s')            | +1 on entering the goal, 0 otherwise                                                    |
| Discount gamma              | 0.99                                                                                    |
| Termination                 | entering a hole or the goal (no bootstrap)                                              |
| Truncation                  | 100 steps (registered TimeLimit; bootstraps)                                            |
| DP optimum V*(s0)           | 0.4146 (value iteration)                                                                |
| Optimal-policy success rate | 0.664 (1,000 greedy rollouts, 100-step limit)                                           |
| Random-policy V(s0)         | 0.0011                                                                                  |