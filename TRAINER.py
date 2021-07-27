from ENV_TRAIN import Retail_Environment
from UNSHAPED_DQN import DQN_Agent
from SHAPED_BLE import DQN_Agent
from SHAPED_B import DQN_Agent
from pandas import DataFrame
import os

x = 0

#################################
MEAN_DEMAND = 4
CV = 0.5
LIFETIME = 2
LEADTIME = 1
C_LOST = 5
C_HOLD = 1
C_PERISH = 7
C_ORDER = 3
FIFO = False	
LIFO = True
#################################
MAX_ORDER = 10
TRAIN_TIME = 1000
#################################
GAMMA = 0.99
EPSILON_DECAY = 0.997
PSI_DECAY = 0.99
EPSILON_MIN = 0.01
LEARNING_RATE = 0.001
EPOCHS = 2000
BATCH_SIZE = 32
UPDATE = 20
#################################
FACTOR = 50
BASESTOCK = 0
S1 = 0
S2 = 0
b = 0

env_train = Retail_Environment(LIFETIME, LEADTIME, MEAN_DEMAND, CV, MAX_ORDER, C_ORDER, C_PERISH, C_LOST, C_HOLD, FIFO, LIFO, TRAIN_TIME)
state_size = len(env_train.state)
action_size = len(env_train.action_space)

rows = 50
columns = EPOCHS
scores = [[0 for i in range(columns)] for j in range(rows)]

for i in range(0,50):
    print("New agent created...")
    
    #unshaped DQN
    agent = DQN_Agent(state_size, action_size, GAMMA, EPSILON_DECAY, EPSILON_MIN, LEARNING_RATE, EPOCHS, env_train, BATCH_SIZE, UPDATE, i, x)
    
    #shaped_b
    #agent = DQN_Agent(state_size, action_size, GAMMA, EPSILON_DECAY, EPSILON_MIN, LEARNING_RATE, EPOCHS, env_train, BATCH_SIZE, UPDATE, i, BASESTOCK, FACTOR, x)
    
    #shaped_ble
    #agent = DQN_Agent(state_size, action_size, GAMMA, EPSILON_DECAY, EPSILON_MIN, LEARNING_RATE, EPOCHS, env_train, BATCH_SIZE, UPDATE, i, S1, S2, b, FACTOR, x)
    
    scores[i] = agent.train()
df = DataFrame({'0': scores[0], '1': scores[1], '2': scores[2], '3': scores[3], '4': scores[4], '5': scores[5], '6': scores[6], '7': scores[7], '8': scores[8], '9': scores[9], '10': scores[10], '11': scores[11], '12': scores[12], '13': scores[13], '14': scores[14], '15': scores[15], '16': scores[16], '17': scores[17], '18': scores[18], '19': scores[19], '20': scores[20], '21': scores[21], '22': scores[22], '23': scores[23], '24': scores[24], '25': scores[25], '26': scores[26], '27': scores[27], '28': scores[28], '29': scores[29], '30': scores[30], '31': scores[31], '32': scores[32], '33': scores[33], '34': scores[34], '35': scores[35], '36': scores[36], '37': scores[37], '38': scores[38], '39': scores[39], '40': scores[40], '41': scores[41], '42': scores[42], '43': scores[43], '44': scores[44], '45': scores[45], '46': scores[46], '47': scores[47], '48': scores[48], '49': scores[49]})
path = PATH
df.to_excel(str(path) + '/EVAL' + str(x) + '/overview.xlsx')




