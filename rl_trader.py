import numpy as np
import pandas as pd
import random

class QLearningAgent:
    def __init__(self, state_size, action_size=3):
        self.state_size = state_size
        self.action_size = action_size
        # The Q-Table: Stores the "Quality" of each action in each state
        self.q_table = {}
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 1.0  # Exploration rate (start by guessing)
        self.epsilon_decay = 0.995

    def get_state(self, data, t, n):
        # Creates a state representing the price trend over 'n' days
        d = t - n + 1
        block = data[d:t + 1] if d >= 0 else -d * [data[0]] + list(data[0:t + 1])
        res = []
        for i in range(n - 1):
            res.append(block[i + 1] - block[i])
        return tuple(np.sign(res)) # Simplify state to UP/DOWN signs

    def act(self, state):
        # Choose between exploring (random) or exploiting (best known Q-value)
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        return np.argmax(self.q_table.get(state, np.zeros(self.action_size)))

    def train(self, state, action, reward, next_state):
        # Update the Q-table using the Bellman Equation
        old_value = self.q_table.get(state, np.zeros(self.action_size))[action]
        next_max = np.max(self.q_table.get(next_state, np.zeros(self.action_size)))
        
        # New Q-value = (1-α)*old + α*(reward + γ*next_max)
        new_value = (1 - self.learning_rate) * old_value + self.learning_rate * (reward + self.discount_factor * next_max)
        
        if state not in self.q_table: self.q_table[state] = np.zeros(self.action_size)
        self.q_table[state][action] = new_value

# --- Simulation ---
df = pd.read_csv('btc_history.csv')
prices = df['close'].values
agent = QLearningAgent(state_size=5)

print("--- Training RL Agent ---")
for e in range(10): # Train for 10 "episodes" or laps
    state = agent.get_state(prices, 0, 5)
    total_profit = 0
    
    for t in range(len(prices)-1):
        action = agent.act(state)
        next_state = agent.get_state(prices, t + 1, 5)
        
        # Simple Reward: Profit/Loss from the price change
        reward = (prices[t+1] - prices[t]) if action == 2 else (prices[t] - prices[t+1]) if action == 0 else 0
        agent.train(state, action, reward, next_state)
        
        state = next_state
        total_profit += reward
    
    agent.epsilon *= agent.epsilon_decay
    print(f"Episode {e+1}: Total Profit/Loss: ${total_profit:,.2f}")