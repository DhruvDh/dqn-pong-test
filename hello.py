import random
import numpy as np
import gymnasium as gym
import ale_py

# Import tinygrad's key components.
from tinygrad.tensor import Tensor
from tinygrad import nn, TinyJit
from tinygrad.nn import optim, state
from tinygrad.dtype import dtypes

from replay_buffer import ReplayBuffer  # Remove config import

# Register ALE-Py environments with gymnasium.
gym.register_envs(ale_py)


# Training parameters
NUM_EPISODES = 500
BATCH_SIZE = 32
GAMMA = 0.99
LEARNING_RATE = 0.00025

# Epsilon-greedy parameters
INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.1
EPSILON_DECAY_STEPS = 30000  # In env steps (not gradient updates)

# Network update and buffer parameters
REPLAY_BUFFER_SIZE = 1000000  # 1M frames
# Original DQN paper updates target net every 10k gradient updates
# Since we do 1 gradient update every 4 env steps (TRAIN_FREQ),
# we need 40k env steps to get 10k gradient updates
TARGET_UPDATE_FREQ = 40000  # In env steps (= 10k gradient updates)
TRAIN_FREQ = 4  # Do a gradient update every 4 env steps
MIN_REPLAY_SIZE = BATCH_SIZE  # Need at least enough samples to fill a batch

# Environment parameters
FRAME_HEIGHT = 84
FRAME_WIDTH = 84
FRAME_STACK = 4
INPUT_SHAPE = (FRAME_STACK, FRAME_HEIGHT, FRAME_WIDTH)
NUM_ACTIONS = 6  # Number of actions in ALE-Py Pong


##########################
# DQN Network Definition #
##########################

class DQN:
    def __init__(self, input_shape, NUM_ACTIONS):
        # Here we assume input_shape is (C, H, W), for example (4,84,84)
        # This network roughly follows the architecture from the DQN paper.
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # After three convolutions on a (84,84) input, the feature map will be 7×7 (if you do the math)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, NUM_ACTIONS)

    def __call__(self, x: Tensor) -> Tensor:
        # x is expected to have shape (batch, C, H, W)
        x = self.conv1(x).relu()
        x = self.conv2(x).relu()
        x = self.conv3(x).relu()
        # Flatten the convolutional features into (batch, 64*7*7)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x).relu()
        x = self.fc2(x)
        return x

@TinyJit
def jit_forward(model, x):
    # Use .realize() to force JIT evaluation for repeated inference.
    return model(x).realize()

@TinyJit
def jit_forward_target(target_model, x):
    return target_model(x).realize()

##########################
# Utility Functions      #
##########################

def preprocess_state(state):
    state = np.array(state, dtype=np.float32) / 255.0
    # Remove transpose as state is already in (C, H, W) format.
    return state

def update_target(model, target_model):
    """Copy model parameters to target_model."""
    sd = state.get_state_dict(model)
    state.load_state_dict(target_model, sd)

def epsilon_by_step(step_count):
    if step_count >= EPSILON_DECAY_STEPS:
        return FINAL_EPSILON
    slope = (FINAL_EPSILON - INITIAL_EPSILON) / EPSILON_DECAY_STEPS
    return INITIAL_EPSILON + slope * step_count

##########################
# Checkpoint Functions   #
##########################

def save_checkpoint(model, target_model, optimizer, total_env_steps, num_gradient_updates, 
                   filename="dqn_checkpoint.safetensors"):
    # Get model parameters
    model_sd = state.get_state_dict(model)
    target_model_sd = state.get_state_dict(target_model)
    opt_sd = state.get_state_dict(optimizer)
    
    # Convert training progress to tensors (renamed for clarity)
    env_steps_tensor = Tensor([total_env_steps], requires_grad=False, dtype=dtypes.int64)
    grad_updates_tensor = Tensor([num_gradient_updates], requires_grad=False, dtype=dtypes.int64)
    
    # Merge everything into a single dict with unique prefixes
    ckpt_dict = {
        "total_env_steps": env_steps_tensor,
        "num_gradient_updates": grad_updates_tensor
    }
    for k, v in model_sd.items():
        ckpt_dict[f"model.{k}"] = v
    for k, v in target_model_sd.items():
        ckpt_dict[f"target_model.{k}"] = v
    for k, v in opt_sd.items():
        ckpt_dict[f"optim.{k}"] = v

    state.safe_save(ckpt_dict, filename)
    print(f"Checkpoint saved to {filename}")

def load_checkpoint(filename, model, target_model, optimizer):
    ckpt_dict = state.safe_load(filename)
    
    # Recover the progress counters (renamed for clarity)
    total_env_steps = int(ckpt_dict["total_env_steps"].numpy()[0])
    num_gradient_updates = int(ckpt_dict["num_gradient_updates"].numpy()[0])

    # Separate model, target model, and optimizer parameters
    model_sd = {}
    target_model_sd = {}
    optim_sd = {}
    for k, v in ckpt_dict.items():
        if k.startswith("model."):
            model_sd[k[len("model."):]] = v
        elif k.startswith("target_model."):
            target_model_sd[k[len("target_model."):]] = v
        elif k.startswith("optim."):
            optim_sd[k[len("optim."):]] = v
    
    # Load parameters back into the models and optimizer
    state.load_state_dict(model, model_sd)
    state.load_state_dict(target_model, target_model_sd)
    state.load_state_dict(optimizer, optim_sd)
    
    print(f"Loaded checkpoint from {filename}:")
    print(f"  Total environment steps: {total_env_steps}")
    print(f"  Gradient updates completed: {num_gradient_updates}")
    return total_env_steps, num_gradient_updates

##########################
# Main Training Loop     #
##########################

def main():
    # Environment setup with consistent frame skip
    gym.register_envs(ale_py)
    env = gym.make("ALE/Pong-v5", render_mode="rgb_array", frameskip=1)
    env = gym.wrappers.AtariPreprocessing(
        env, 
        grayscale_obs=True, 
        frame_skip=4,  # This is the only frame skip we'll use
        scale_obs=True
    )
    env = gym.wrappers.FrameStackObservation(env, stack_size=FRAME_STACK)
    
    # Initialize networks and optimizer
    model = DQN(INPUT_SHAPE, NUM_ACTIONS)
    target_model = DQN(INPUT_SHAPE, NUM_ACTIONS)
    optimizer = optim.Adam(state.get_parameters(model), lr=LEARNING_RATE)
    
    # Initialize step counters
    total_env_steps = 0      # Total steps taken in environment
    num_gradient_updates = 0  # Number of gradient updates performed
    
    # Try to load checkpoint
    try:
        total_env_steps, num_gradient_updates = load_checkpoint(
            "dqn_checkpoint.safetensors", 
            model, target_model, optimizer
        )
    except Exception as e:
        print(f"Starting fresh (checkpoint load failed: {e})")

    # Initialize fixed-size replay buffer
    replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE, state_shape=INPUT_SHAPE)
    
    with Tensor.train():
        for episode in range(NUM_EPISODES):
            obs, info = env.reset()
            state_np = preprocess_state(obs)
            total_reward = 0
            done = False
            
            # Collect experience for this episode
            while not done:
                # Each env step represents 4 frames due to frame skip
                total_env_steps += 4  
                
                # Epsilon follows env steps for consistent exploration
                epsilon = epsilon_by_step(total_env_steps)

                # Epsilon-greedy action selection.
                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    # Add a batch dimension (1, C, H, W) before feeding into the network.
                    state_tensor = Tensor(np.expand_dims(state_np, axis=0))
                    # Use jit_forward for fused inference
                    q_values = jit_forward(model, state_tensor)
                    # (Assuming q_values.numpy() returns an array of shape (1, NUM_ACTIONS).)
                    action = int(np.argmax(q_values.numpy()[0]))

                # Take a step in the environment.
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                next_state_np = preprocess_state(next_obs)

                # Store the transition.
                replay_buffer.push(state_np, action, reward, next_state_np, done)
                state_np = next_state_np
                total_reward += reward

                # Train when we have enough samples for a batch
                if len(replay_buffer) >= MIN_REPLAY_SIZE and total_env_steps % TRAIN_FREQ == 0:
                    # Perform gradient update
                    num_gradient_updates += 1
                    
                    # Sample a batch from the replay buffer
                    batch_state, batch_action, batch_reward, batch_next_state, batch_done = replay_buffer.sample(BATCH_SIZE)
                    
                    # Convert to tensors
                    batch_state = Tensor(batch_state)
                    batch_next_state = Tensor(batch_next_state)
                    batch_reward = Tensor(batch_reward).reshape(BATCH_SIZE, 1)
                    batch_done = Tensor(batch_done.astype(np.float32)).reshape(BATCH_SIZE, 1)
                    
                    # Current Q-values
                    q_values = model(batch_state)
                    one_hot = np.zeros((BATCH_SIZE, NUM_ACTIONS), dtype=np.float32)
                    for i, a in enumerate(batch_action):
                        one_hot[i, a] = 1.0
                    one_hot = Tensor(one_hot)
                    q_value = (q_values * one_hot).sum(axis=1).reshape(BATCH_SIZE, 1)
                    
                    # Target Q-values
                    next_q_values = jit_forward_target(target_model, batch_next_state)
                    max_next_q = Tensor(next_q_values.numpy().max(axis=1, keepdims=True))
                    target = batch_reward + GAMMA * max_next_q * (1 - batch_done)
                    
                    # Update
                    loss = ((q_value - target) ** 2).mean()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    if num_gradient_updates % 100 == 0:
                        loss_str = f"{loss.item():.6f}"
                        print(f"Gradient update {num_gradient_updates}")
                        print(f"  Environment steps: {total_env_steps}")
                        print(f"  Loss: {loss_str}")

                # Update target network every TARGET_UPDATE_FREQ env steps
                # (equivalent to 10k gradient updates since TRAIN_FREQ=4)
                if total_env_steps % TARGET_UPDATE_FREQ == 0:
                    update_target(model, target_model)
                    print(f"Updated target network at env step {total_env_steps}")
                    print(f"  (Gradient update #{num_gradient_updates})")
                
                # Save checkpoint every 10k env steps
                if total_env_steps % 10000 == 0:
                    save_checkpoint(
                        model, target_model, optimizer,
                        total_env_steps, num_gradient_updates
                    )
            
            print(f"Episode {episode + 1}: Reward = {total_reward}, Steps = {total_env_steps}")

    env.close()

if __name__ == "__main__":
    main()
