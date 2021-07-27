# Reward-shaping-to-improve-the-performance-of-DRL-in-perishable-inventory-management
Link to paper: https://www.ssrn.com/abstract=3804655 

Abstract: 
Deep reinforcement learning (DRL) has proven to be an effective, general-purpose technology to develop good replenishment policies in inventory management. Yet, despite formidable computational effort, DRL cannot always beat the state-of-the-art heuristics for stylized problems. We show how transfer learning from existing, well-performing heuristics may stabilize the training process and improve the performance of DRL in inventory control. Specifically, we apply potential-based reward shaping to improve the performance of a deep Q-network (DQN) algorithm to manage inventory of perishable goods which, cursed by dimensionality, has proven to be notoriously complex. Applying our approach using existing replenishment policies may not only reduce firms' replenishment costs, the increased stability may also help to gain trust in the policies obtained by black box DRL algorithms.

Code:
TRAINER.py -- trains object from class unshaped_dqn, shaped_b or shaped_ble\\
UNSHAPED_DQN.py -- deep Q-network without reward shaping
SHAPED_B.py -- deep Q-network with reward shaping with base-stock policy as teacher
SHAPED_BLE.py -- deep Q-network with reward shaping with BSP-low-EW as teacher
ENV_TRAIN.py -- perishable inventory problem to train the DRL models
ENV_TEST.py -- perishable inventory problem to test the DRL models with seeded demand
EVALUATE_DRL_POLICY.py -- evaluates a trained DRL model in the test environment
