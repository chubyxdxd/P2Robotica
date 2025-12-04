# dqn_agent.py
import numpy as np
from sklearn.neural_network import MLPRegressor
from collections import deque
import random
import pickle

class DQNAgent:
    """Deep Q-Network agent using sklearn's MLPRegressor"""
    
    def __init__(self,
                 state_size: int,
                 action_size: int,
                 learning_rate: float = 0.005,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.2,
                 epsilon_decay: float = 0.999,
                 memory_size: int = 2000,
                 batch_size: int = 64,
                 target_update_freq: int = 100):
        """
        Initialize DQN Agent
        
        Args:
            state_size: Dimension of state space
            action_size: Number of discrete actions
        """
        self.state_size = state_size 
        self.action_size = action_size 
        self.gamma = gamma 
        self.epsilon = epsilon 
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.step_count = 0
        
        # Experience replay memory
        self.memory = deque(maxlen=memory_size) 
        
        # Q-network (main network) Aprende
        self.q_network = MLPRegressor(
            hidden_layer_sizes=(128, 128),
            activation='relu',
            solver='adam',
            learning_rate_init=learning_rate,
            max_iter=1,  # We'll call partial_fit
            warm_start=True,
            random_state=42
        )
        
        # Target network Estabilidad
        self.target_network = MLPRegressor(
            hidden_layer_sizes=(128, 128),
            activation='relu',
            solver='adam',
            learning_rate_init=learning_rate,
            max_iter=1,
            warm_start=True,
            random_state=42
        )
        
        # Initialize networks with dummy data
        dummy_X = np.random.randn(1, state_size)
        dummy_y = np.random.randn(1, action_size)
        self.q_network.fit(dummy_X, dummy_y)
        self.target_network.fit(dummy_X, dummy_y)
        
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory, aprende de experiencias pasadas"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using ε-greedy policy
        """
        # Exploration
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # Exploitation
        state_reshaped = state.reshape(1, -1)
        q_values = self.q_network.predict(state_reshaped)[0]
        return int(np.argmax(q_values))
    
    def replay(self) -> float:
        """
        Train Q-network on batch from experience replay
        Returns average loss
        """
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample random batch
        minibatch = random.sample(self.memory, self.batch_size)
        
        states = np.array([transition[0] for transition in minibatch])
        actions = np.array([transition[1] for transition in minibatch])
        rewards = np.array([transition[2] for transition in minibatch])
        next_states = np.array([transition[3] for transition in minibatch])
        dones = np.array([transition[4] for transition in minibatch])
        
        # Current Q-values
        current_q_values = self.q_network.predict(states)
        
        # Next Q-values from target network
        next_q_values = self.target_network.predict(next_states)
        
        # Compute target Q-values
        target_q_values = current_q_values.copy()
        for i in range(self.batch_size):
            if dones[i]:
                target_q_values[i][actions[i]] = rewards[i]
            else:
                target_q_values[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
        
        # Train network
        self.q_network.partial_fit(states, target_q_values)
        
        # Update target network
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.update_target_network()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Compute loss (MSE)
        loss = np.mean((target_q_values - current_q_values) ** 2)
        return float(loss)
    
    def update_target_network(self):
        """Copy weights from Q-network to target network"""
        
        # Logging
        self.episode_rewards = []
        self.episode_steps = []
        self.success_count = 0
        self.target_network = pickle.loads(pickle.dumps(self.q_network))
        print("Target network updated")
    
    def save(self, filepath: str):
        """Save model to file"""
        model_data = {
            'q_network': self.q_network,
            'target_network': self.target_network,
            'epsilon': self.epsilon,
            'step_count': self.step_count
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model from file"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        self.q_network = model_data['q_network']
        self.target_network = model_data['target_network']
        self.epsilon = model_data.get('epsilon', 0.0)
        self.step_count = model_data.get('step_count', 0)
        print(f"Model loaded from {filepath}")
