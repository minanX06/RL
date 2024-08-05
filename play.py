from collections import deque
import tensorflow as tf
import numpy as np
import random

class TicTacToe:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.reset()
    
    def reset(self):
        self.board = np.zeros((self.rows, self.cols))
        self.current_player = 1
        return self.board
    
    def putOn(self, value):
        row = value // self.cols
        col = value % self.cols
        if self.board[row, col] != 0:
            return False
        else:
            self.board[row, col] = self.current_player
            self.current_player = 3 - self.current_player
            return True
        
    def is_full(self):
        if self.board.all():
            return True
        else:
            return False
        
class DQNAgent:
    def __init__(self, rows, cols):
        self.memory = deque(maxlen = 2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.rows = rows
        self.cols = cols
        self.model = self.build_model()
        self.target_model = self.build_model()

    def build_model(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Flatten(input_shape=(self.rows, self.cols)))
        model.add(tf.keras.layers.Dense(8, activation='relu'))
        model.add(tf.keras.layers.Dense(8, activation='relu'))
        model.add(tf.keras.layers.Dense(2, activation='linear'))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                      loss='mse')
        return model
    
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        state = np.reshape(state, [1, 1, 2])
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.rows * self.cols)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])
    
    def predict(self, state):
        predict_values = self.model.predict(state)
        return np.argmax(predict_values[0])
    
    def predict_raw(self, state):
        predict_values = self.model.predict(state)
        return predict_values

    def replay(self, batch_size):
        experience = random.sample(self.memory, batch_size)
        state = np.array([exp[0] for exp in experience])
        action = [exp[1] for exp in experience]
        reward = [exp[2] for exp in experience]
        next_state = np.array([exp[3] for exp in experience])
        done = [exp[4] for exp in experience]

        target = self.model.predict(state)
        for i in range(batch_size):
            target[i][action[i]] = reward[i] + self.gamma * np.amax(self.target_model.predict(next_state)[0])
        self.model.fit(state, target, epochs=1, verbose=1)
        print('fit finished')
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

rows = 1
cols = 2

env = TicTacToe(rows, cols)
agent = DQNAgent(rows, cols)
episodes = 1000
batch_size = 32

for e in range(episodes):
    state = np.array(env.reset())
    state = np.reshape(state, [1, 2])
    for time in range(2):
        action = agent.act(state)
        if not env.putOn(action):
            reward = -10
            next_state = np.array(state)
            done = True
        else:
            next_state = np.array(env.board)
            reward = 1
            done = env.is_full()
        next_state = np.reshape(next_state, [1, 2])
        agent.remember(state, action, reward, next_state, done)
        state = np.array(next_state)
        if done:
            agent.update_target_model()
            break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)


env.reset()
n = 0
env.putOn(n)
state = np.reshape(env.board, [1, 1, 2])
print(agent.predict(state))
print(agent.predict_raw(state))

print('---------------')

env.reset()
n = 1
env.putOn(n)
state = np.reshape(env.board, [1, 1, 2])
print(agent.predict(state))
print(agent.predict_raw(state))
