import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('dark_background')

k = 10
q = np.random.normal(size = k)
bandit = lambda A: np.random.normal(q[A], 1)
steps = 10000
epsilon = 0.1
alpha = 0.1

Q0 = np.zeros(k)
N0 = np.zeros(k)
Q1 = np.zeros(k)

mean_rewards_0 = [0]
mean_rewards_1 = [0]

optimal_action_taken_0 = [0]
optimal_action_taken_1 = [0]

for i in range(steps):
  A0 = np.argmax(Q0) if np.random.random() < 1 - epsilon else np.random.randint(k)
  A1 = np.argmax(Q1) if np.random.random() < 1 - epsilon else np.random.randint(k)
  R0 = bandit(A0)
  R1 = bandit(A1)

  N0[A0] += 1
  Q0[A0] += (R0 - Q0[A0]) / N0[A0]
  Q1[A1] += alpha * (R1 - Q1[A1])

  mean_rewards_0.append(mean_rewards_0[-1] + (R0 - mean_rewards_0[-1]) / (i + 1))
  mean_rewards_1.append(mean_rewards_1[-1] + (R1 - mean_rewards_1[-1]) / (i + 1))

  optimal = np.argmax(q)
  optimal_action_taken_0.append(optimal_action_taken_0[-1] + (int(A0 == optimal) - optimal_action_taken_0[-1]) / (i + 1))
  optimal_action_taken_1.append(optimal_action_taken_1[-1] + (int(A1 == optimal) - optimal_action_taken_1[-1]) / (i + 1))

  q += np.random.normal(0, 0.01)

fig = plt.figure(figsize = (15, 7))
plt.subplot(1, 2, 1)
plt.plot(mean_rewards_0, label = 'Incrementally computed sample averages')
plt.plot(mean_rewards_1, label = 'Constant step-size parameter')
plt.xlim(0, steps)
plt.xlabel('Steps')
plt.ylabel('Average reward')
plt.xscale('symlog')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(optimal_action_taken_0, label = 'Incrementally computed sample averages')
plt.plot(optimal_action_taken_1, label = 'Constant step-size parameter')
plt.xlim(0, steps)
plt.ylim(0, 1)
plt.xlabel('Steps')
plt.ylabel('% Optimal action')
plt.xscale('symlog')
plt.legend()
plt.show()
