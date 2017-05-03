import gym
import numpy as np
import sys

env = gym.make('FrozenLake-v0')

Q = np.zeros([env.observation_space.n, env.action_space.n])
Qreward = np.ones([env.observation_space.n, env.action_space.n])

alpha = 0.85 # Future rewards and immediate rewards
gamma = 0.99 # Learning speed
num_episodes = 2000

rList = []

for i in range(num_episodes):
	s = env.reset()
	rAll = 0
	d = False
	j = 0
	while j < 999:
		j += 1
		a = np.argmax(Q[s,:] + np.random.rand(1, env.action_space.n) * (1.0/(i+1)))
		#a = np.argmax(Q[s,:] + np.random.rand(1, env.action_space.n) / 1000.0)
		s1, r, d, _ = env.step(a)
		Qreward[s, a] = alpha * (r + gamma*np.max(Q[s1,:]) - Q[s,a])
		#print Qreward[s,a], r, Q[s1,:], Q[s,a]
		Q[s, a] = Q[s, a] + Qreward[s, a]
		rAll += r
		s = s1
		if d == True:
			break
	rList.append(rAll)
	print 'Learning Score: ' + str(sum(rList)/num_episodes)

print Qreward
print Q

def run():
	num_step = 0
	s = env.reset()
	while True:
		#env.render()
		max_r = 0
		max_a = -1
		index = 0
		for q in Q[s,:]:
			if q > max_r:
				max_r = q
				max_a = index
			index += 1
		s, _, d, _ = env.step(max_a)
		num_step += 1
		if d == True:
			print num_step
			break

if sys.argv[1] == 'run':
	run()