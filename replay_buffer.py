import numpy as np

class ReplayBuffer:
    def __init__(self, capacity, state_shape=(4, 84, 84), dtype=np.float32):
        self.capacity = capacity
        self.position = 0
        self.size = 0
        
        # Pre-allocate memory for all arrays
        self.states = np.zeros((capacity, *state_shape), dtype=dtype)
        self.next_states = np.zeros((capacity, *state_shape), dtype=dtype)
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=dtype)
        self.dones = np.zeros(capacity, dtype=np.bool_)

    def push(self, state, action, reward, next_state, done):
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )

    def __len__(self):
        return self.size
