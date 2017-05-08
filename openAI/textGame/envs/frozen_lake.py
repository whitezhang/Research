import numpy as np
import sys
import utils

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

MAPS = {
	"4x4": [
		"SHHG",
		"FFFF",
		"FFFF",
		"FFFF"
	],
	"8x8": [
		"SFFFFFFF",
		"FFFFFFFF",
		"FFFHFFFF",
		"FFFFFHFF",
		"FFFHFFFF",
		"FHHFFFHF",
		"FHFFHFHF",
		"FFFHFFFG"
	],
	"4x8": [
		"SHHHHHHG",
		"FFFFFFFF",
		"FFFFFFFF",
		"FFFFFFFF"
	],
}

class FrozenLakeEnv():
	"""
	S : starting point, safe
	F : frozen surface, safe
	H : hole, fall to your doom
	G : goal, where the frisbee is located
	"""

	def __init__(self, map_name="4x4"):
		self.map_name = map_name

		self.direction = {
			0: [-1, 0],
			1: [0, 1],
			2: [1, 0],
			3: [0, -1]
		}
		desc = MAPS[map_name]
		self.desc = desc = np.asarray(desc, dtype='c')
		self.nrow, self.ncol = nrow, ncol = desc.shape
		self.srow, self. scol = srow, scol = self.get_sp()

		self.nS = nrow * ncol
		self.nA = 4


	def reset(self):
		map_name = self.map_name

		desc = MAPS[map_name]
		self.desc = desc = np.asarray(desc, dtype='c')
		self.nrow, self.ncol = nrow, ncol = desc.shape
		self.srow, self. scol = srow, scol = self.get_sp()

		self.nS = nrow * ncol
		self.nA = 4

		state = srow * ncol + scol
		return state


	def get_sp(self):
		desc = self.desc
		nrow, ncol = self.nrow, self.ncol
		for row in range(nrow):
			for col in range(ncol):
				if desc[row][col] == 'S':
					return row, col

	def step(self, action):
		nrow, ncol = self.nrow, self.ncol
		srow = self.srow + self.direction[action][0]
		scol = self.scol + self.direction[action][1]

		srow = 0 if srow < 0 else srow
		srow = nrow-1 if srow >= nrow else srow
		scol = 0 if scol < 0 else scol
		scol = scol-1 if scol >= ncol else scol
		self.srow = srow
		self.scol = scol

		nrow, ncol = self.nrow, self.ncol
		desc = self.desc

		state = srow * ncol + scol
		reward = 1 if desc[srow][scol] == 'G' else 0
		done = True if desc[srow][scol] in 'GH' else False
		info = None
		return state, reward, done, info

	def render(self, close=False):
		if close:
			return

		outfile = sys.stdout

		srow, scol = self.srow, self.scol
		nrow, ncol = self.nrow, self.ncol
		desc = self.desc.tolist()
		desc = [[c.decode('utf-8') for c in line] for line in desc]
		desc[srow][scol] = utils.colorize(desc[srow][scol], "red", highlight=True)
		outfile.write("\n".join(''.join(line) for line in desc)+"\n")
		outfile.write("\n")

		return outfile
	



