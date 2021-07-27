import numpy as np 
import copy
import random
import math

class Retail_Environment (object):

	def __init__(self, lifetime, leadtime, mean_demand, coef_of_var, max_order, cost_order, cost_outdate, cost_lost, cost_holding, FIFO, LIFO, time):

		self.lifetime = lifetime
		self.leadtime = leadtime
		self.mean_demand = mean_demand
		self.coef_of_var = coef_of_var
		self.max_order = max_order
		self.cost_order = cost_order
		self.cost_outdate = cost_outdate
		self.cost_lost = cost_lost
		self.cost_holding = cost_holding
		self.FIFO = FIFO
		self.LIFO = LIFO
		self.time = time

		self.demand = 0
		self.action = 0
		self.current_time = 0
		self.reward = 0

		self.shape = 1 / (self.coef_of_var ** 2)
		self.scale = self.mean_demand / self.shape

		self.state = []
		for i in range(self.lifetime + self.leadtime-1):
			self.state.append(0)

		self.render_state = self.state.copy()

		self.action_space = []
		for i in range(self.max_order + 1):
			self.action_space.append(i)

		print('Environment created...')

	def step(self, action):

		self.action = action
		self.demand = round(np.random.gamma(self.shape, self.scale, size = None))
		demand = self.demand
		self.render_state = self.state.copy()
		
		#update inventory in pipeline with order
		next_state = [None] * (self.leadtime + self.lifetime)
		for i in range(self.leadtime + self.lifetime - 1):
			next_state[i+1] = self.state[i]
		next_state[0] = action

		#inventory depletion
		calc_state = next_state.copy()
		if self.FIFO:
			for i in range(self.lifetime):
				if demand > 0:
					next_state[-i-1] = max(calc_state[-i-1] - demand, 0)
					demand = max(demand - calc_state[-i-1], 0)
		if self.LIFO:
			for i in range(self.leadtime, self.leadtime + self.lifetime):
				if demand > 0:
					next_state[i] = max(calc_state[i] - demand, 0)
					demand = max(demand - calc_state[i], 0)

		#age inventory
		calc_state = next_state.copy()
		for i in range(self.leadtime + self.lifetime):
			if i == 0:
				next_state[i] = 0
			else:
				next_state[i] = calc_state[i-1]

		order_cost = action * self.cost_order
		outdate_cost = calc_state[-1] * self.cost_outdate
		lost_sales_cost = demand * self.cost_lost
		holding_cost = 0
		for i in range(self.leadtime + 1, self.leadtime + self.lifetime):
			holding_cost += next_state[i] * self.cost_holding

		self.reward = -order_cost - outdate_cost - lost_sales_cost - holding_cost
		self.current_time += 1
		for i in range(self.lifetime + self.leadtime - 1):
			self.state[i] = next_state[i+1]

		return self.state, self.reward, self.isFinished(self.current_time), None

	def isFinished(self, current_time):
		return current_time == self.time

	def reset(self):

		self.current_time = 0

		self.state = []
		for i in range(self.leadtime + self.lifetime - 1):
			self.state.append(0)

		#print('Reset environment...')

		return self.state, self.current_time


	def render(self):

		print('---------------------------------------------------')
		print('*****   Period ' + str(self.current_time) + '   *****')
		inventory_on_hand = []
		for i in range(self.leadtime - 1 , self.leadtime + self.lifetime - 1):
			inventory_on_hand.append(self.render_state[i])
		inventory_in_pipeline = []
		inventory_in_pipeline.append(self.action)
		for i in range(0, self.leadtime - 1):
			inventory_in_pipeline.append(self.render_state[i])
		print('Inventory on hand: ' + str(inventory_on_hand))
		print('Order placed: ' + str(self.action))
		print('Orders in pipeline: ' + str(inventory_in_pipeline))
		print('Demand encountered: ' + str(self.demand))
		print('Costs: ' + str(self.reward))

	def random_action(self):
		return random.sample(self.action_space, 1)[0]

	def random_state(self):
		random_state = []
		for i in range(0, self.lifetime + self.leadtime - 1):
			random_state.append(np.random.randint(self.max_order))
		return random_state