# 🤖 Q-Learning Trader (Reinforcement Learning)

A Python-based **reinforcement learning experiment** that applies **Q-Learning** to crypto price data to learn trading behavior through **trial and error**.

Instead of predicting prices directly, this project lets an agent **interact with the market**, receive rewards, and gradually learn which actions perform better in different market states.

---

## 🚀 What This Project Does

- Implements a **Q-Learning agent from scratch**
- Converts price movements into **discrete market states**
- Trains the agent over multiple episodes
- Learns a **Q-table** mapping states → action values
- Evaluates performance using cumulative profit/loss

---

## 🧠 Reinforcement Learning Setup

### 🔹 State Representation
- Uses the **direction (UP / DOWN)** of the last `n` price changes
- State is simplified to:
(+1, -1, +1, ...)
- This reduces noise and state explosion.

---

### 🔹 Action Space

| Action | Meaning |
|------|--------|
| `0` | SELL |
| `1` | HOLD |
| `2` | BUY |

---

### 🔹 Reward Function

- BUY → profit if price goes up  
- SELL → profit if price goes down  
- HOLD → zero reward  

```text
Reward = price[t+1] - price[t]  (BUY)
Reward = price[t] - price[t+1]  (SELL)
Reward = 0                      (HOLD)

## Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/PranavVetkar/Q-Learning-Trader.git
cd Q-Learning-Trader