from ENV_TEST import Retail_Environment
from keras.models import load_model, Sequential
from UNSHAPED_DQN import DQN_Agent
import h5py
import numpy as np
from statistics import mean, stdev
from math import sqrt
from pandas import DataFrame

x = 0

#Define the parameters
MEAN_DEMAND = 4
CV = 0.5
LIFETIME = 3
LEADTIME = 1
COST_LOST = 5
COST_HOLDING = 1
COST_OUTDATE = 7
COST_ORDER = 3
FIFO = False
LIFO = True

TIME = 100_000
WARMUP = 10_000

MAX_ORDER = 10

#hyperparameters of the DQN (irrelevant when testing)
gamma = 0
epsilon_decay = 0
epsilon_min = 0
learning_rate = 0
epochs = 0
batch_size = 0
update = 0




def test_model(trained_DQN):
	env_test = Retail_Environment(LIFETIME, LEADTIME, MEAN_DEMAND, CV, MAX_ORDER, COST_ORDER, COST_OUTDATE, COST_LOST, COST_HOLDING, FIFO, LIFO, TIME, WARMUP)
	state_size = len(env_test.state)
	action_size = len(env_test.action_space)
	agent = DQN_Agent(state_size, action_size, gamma, epsilon_decay, epsilon_min, learning_rate, epochs, env_test, batch_size, update, iteration,0)
	agent.epsilon = 0
	agent.load(trained_DQN)

	rewards_per_period = []

	for i in range(10):
		done = False
		state, _ = env_test.reset()
		rewards = [] 

		while not done:
			state = np.reshape(state, [1, state_size])
			action = agent.act(state)
			next_state, reward, done, _ = env_test.step(action)
			rewards.append(reward)
			next_state = np.reshape(next_state, [1, state_size])
			state = next_state
		for j in range(WARMUP):
			rewards.remove(0)
		avg_per_period = sum(rewards) / (TIME-WARMUP)
		
		aid = 0
		for x in rewards:
			aid += (x - avg_per_period)**2
		std_per_period = sqrt((1/(TIME - WARMUP - 1)) * aid)

		rewards_per_period.append(avg_per_period)
		if i % 1 == 0:
			print(' ')
			print('********************')
			print('Game: ' + str(i+1))
			print('Avg per period: ' + str(avg_per_period))
			print('Std between periods: ' + str(std_per_period))
			print(' ')

	avg = mean(rewards_per_period)
	std = stdev(rewards_per_period)
	rewards_per_period.append(avg)
	rewards_per_period.append(std)
	print(' ')
	print(' ')
	print('Overall average per period: ' + str(avg))
	print('Std between games: ' + str(std))
	
	return rewards_per_period, avg


scores = [[0 for i in range(12)] for j in range(50)]
averages = []
for i in range(50):
	path = PATH
	scores[i], avg = test_model(str(path) + '/EVAL' + str(x) + '/Lifetime ' + str(LIFETIME) + ' - iteration ' + str(i) + '.h5')
	averages.append(avg)
overall_avg = mean(averages)
overall_std = stdev(averages)
df = DataFrame({'0': scores[0], '1': scores[1], '2': scores[2], '3': scores[3], '4': scores[4], '5': scores[5], '6': scores[6], '7': scores[7], '8': scores[8], '9': scores[9], '10': scores[10], '11': scores[11], '12': scores[12], '13': scores[13], '14': scores[14], '15': scores[15], '16': scores[16], '17': scores[17], '18': scores[18], '19': scores[19], '20': scores[20], '21': scores[21], '22': scores[22], '23': scores[23], '24': scores[24], '25': scores[25], '26': scores[26], '27': scores[27], '28': scores[28], '29': scores[29], '30': scores[30], '31': scores[31], '32': scores[32], '33': scores[33], '34': scores[34], '35': scores[35], '36': scores[36], '37': scores[37], '38': scores[38], '39': scores[39], '40': scores[40], '41': scores[41], '42': scores[42], '43': scores[43], '44': scores[44], '45': scores[45], '46': scores[46], '47': scores[47], '48': scores[48], '49': scores[49], 'AVG': overall_avg, 'STD': overall_std})
df.to_excel(str(path) + '/EVAL' + str(x) + '/evaluation.xlsx')


