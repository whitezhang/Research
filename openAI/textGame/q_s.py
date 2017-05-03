import numpy as np
import random
import sys
from envs import frozen_lake

debug = False

# Policy
# threshold -> 0  # greedy
# threshold -> 10 # soft
def pick_action(Q, threshold=2):
	epsilon = random.randint(0,9)
	max_q = max(Q)
	best_index = [i for i, j in enumerate(Q) if j == max_q]
	a = random.choice(best_index)

	if epsilon < threshold:
		a = random.randint(0,3)
	return a

#alpha = 0.85 # Future rewards and immediate rewards
#gamma = 0.99 # Learning speed
def sarsa(env=None, alpha=0.85, gamma=0.99, num_episodes=2000, iters_one_episode=99):
	if env == None:
		return

	Q = np.zeros([env.nS, env.nA])
	Q2 = np.zeros([env.nS, env.nA])
	Qreward = np.ones([env.nS, env.nA])

	rList = []
	for i in range(1, num_episodes):
		s = env.reset()
		rAll = 0
		d = False
		j = 0
		a = pick_action(Q[s,:], 2)
		while j < iters_one_episode:
			s1, r, d, _ = env.step(a)
			if s == s1:
				a = pick_action(Q[s,:], 2)
				continue
			j += 1

			a1 = pick_action(Q[s1,:], 2)
			Qreward[s, a] = alpha * (r + gamma*Q[s1,a1] - Q[s,a])
			Q2[s1, a1] += 1

			Q[s, a] = Q[s, a] + Qreward[s, a]
			rAll += r
			s = s1
			a = a1
			if d == True:
				break
		rList.append(rAll)
		if debug == True:
			print 'Learning Score: ' + str(sum(rList)*1./i) + '\tSteps: ' + str(run(env, Q, False)) + '\tIterations:' + str(j)
			print Q2
			print Q
	return Q

def qlearning(env=None, alpha=0.85, gamma=0.99, num_episodes=2000, iters_one_episode=99):
	if env == None:
		return

	Q = np.zeros([env.nS, env.nA])
	Qreward = np.ones([env.nS, env.nA])

	rList = []
	for i in range(1, num_episodes):
		s = env.reset()
		rAll = 0
		d = False
		j = 0
		while j < iters_one_episode:
			a = pick_action(Q[s,:], 2)

			s1, r, d, _ = env.step(a)
			if s == s1:
				a = pick_action(Q[s,:], 2)
				continue
			j += 1

			Qreward[s, a] = alpha * (r + gamma*np.max(Q[s1,:]) - Q[s,a])
			Q[s, a] = Q[s, a] + Qreward[s, a]
			rAll += r
			s = s1
			if d == True:
				break
		rList.append(rAll)
		if debug == True:
			print 'Learning Score: ' + str(sum(rList)*1./i) + '\tSteps: ' + str(run(env, Q, False)) + '\tIterations:' + str(j)
	return Q

def run(env=None, Q=None, display=True):
	if env == None or Q == None:
		return

	num_step = 0
	s = env.reset()
	while True:
		if display == True:
			env.render()
		max_r = -1
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
		if num_step > 100:
			break
	return num_step

if __name__ == '__main__':
	if len(sys.argv) != 5:
		print 'Usage:'
		print '\tpython q_s.py [1|2|3] [q|s] [r|n] [d|n]'
		print '\t\t1:4x4\t2:4x8\t3:8x8'
		exit(0)	

	if sys.argv[1] == '1':
		map_name = '4x4'
	elif sys.argv[1] == '2':
		map_name = '4x8'
	elif sys.argv[1] == '3':
		map_name = '8x8'

	env = frozen_lake.FrozenLakeEnv(map_name=map_name)
	Q = np.zeros([1, 1])

	if sys.argv[4] == 'd':
		debug = True

	if sys.argv[2] == 'q':
		Q = qlearning(env)
	elif sys.argv[2] == 's':
		Q = sarsa(env)

	if sys.argv[3] == 'r':
		run(env, Q)
