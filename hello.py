import numpy as np
import gymnasium as gym
import ale_py

import torch
import torch.nn as nn
import torch.nn.functional as F
from schedulefree import AdamWScheduleFree

from replay_buffer import ReplayBuffer

# Register ALE-Py environments with gymnasium.
gym.register_envs(ale_py)

# --------------------
# Training Parameters
# --------------------
NUM_EPISODES = 1500
BATCH_SIZE = 32
GAMMA = 0.99
LEARNING_RATE = 0.00025

# Epsilon-greedy parameters
INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.1
EPSILON_DECAY_STEPS = 30000  # In env steps (not gradient updates)

# Network update and buffer parameters
REPLAY_BUFFER_SIZE = 1000000  # 1M frames
TARGET_UPDATE_FREQ = (
    40000  # In env steps (= ~10k gradient updates if 1 update every 4 steps)
)
TRAIN_FREQ = 4  # Do a gradient update every 4 env steps
MIN_REPLAY_SIZE = BATCH_SIZE

# Environment parameters
FRAME_HEIGHT = 84
FRAME_WIDTH = 84
FRAME_STACK = 4
INPUT_SHAPE = (FRAME_STACK, FRAME_HEIGHT, FRAME_WIDTH)
NUM_ACTIONS = 6  # Number of actions in ALE-Py Pong


# --------------------
# Define DQN Network
# --------------------
class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        # input_shape is (C, H, W), e.g. (4, 84, 84)
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # After these conv layers (for 84x84), the spatial size is 7x7
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# --------------------
# Utility Functions
# --------------------
def preprocess_state(state):
    # Just ensure float32 type - scaling to [0,1] is already done by AtariPreprocessing
    return np.array(state, dtype=np.float32)


def update_target(model, target_model):
    """Copy model parameters to target_model."""
    target_model.load_state_dict(model.state_dict())


def epsilon_by_step(step_count):
    if step_count >= EPSILON_DECAY_STEPS:
        return FINAL_EPSILON
    slope = (FINAL_EPSILON - INITIAL_EPSILON) / EPSILON_DECAY_STEPS
    return INITIAL_EPSILON + slope * step_count


# --------------------
# Checkpoint Functions
# --------------------
def save_checkpoint(
    model,
    target_model,
    optimizer,
    total_env_steps,
    num_gradient_updates,
    filename="dqn_checkpoint.pth",
):
    # 2) Switch optimizer to eval mode before saving (per our optimizer’s requirement)
    optimizer.eval()

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "target_model_state_dict": target_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "total_env_steps": total_env_steps,
        "num_gradient_updates": num_gradient_updates,
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")

    # return to train mode if you’re still training:
    optimizer.train()


def load_checkpoint(filename, model, target_model, optimizer):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint["model_state_dict"])
    target_model.load_state_dict(checkpoint["target_model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    total_env_steps = checkpoint["total_env_steps"]
    num_gradient_updates = checkpoint["num_gradient_updates"]
    print(f"Loaded checkpoint from {filename}:")
    print(f"  Total environment steps: {total_env_steps}")
    print(f"  Gradient updates completed: {num_gradient_updates}")
    return total_env_steps, num_gradient_updates


# --------------------
# Main Training Loop
# --------------------
def main():
    # Detect device (GPU/MPS if available, otherwise CPU)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # Environment setup with consistent frame skip
    env = gym.make("ALE/Pong-v5", render_mode="rgb_array", frameskip=1)
    env = gym.wrappers.AtariPreprocessing(
        env,
        grayscale_obs=True,
        frame_skip=4,  # The only frame skip we'll use
        scale_obs=True,
    )
    env = gym.wrappers.FrameStackObservation(env, stack_size=FRAME_STACK)

    # Initialize networks
    model = DQN(INPUT_SHAPE, NUM_ACTIONS).to(device)
    target_model = DQN(INPUT_SHAPE, NUM_ACTIONS).to(device)
    update_target(model, target_model)

    optimizer = AdamWScheduleFree(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,  # typical AdamW usage
        warmup_steps=5000,  # linear LR warmup steps
        r=0.0,
        weight_lr_power=2.0,
        foreach=True,
    )

    # 4) Tell the optimizer we’re in training mode (important for RAdamScheduleFree)
    optimizer.train()

    # Initialize counters
    total_env_steps = 0
    num_gradient_updates = 0

    # checkpoint loading
    try:
        total_env_steps, num_gradient_updates = load_checkpoint(
            "dqn_checkpoint.pth", model, target_model, optimizer
        )
        # Re-enter train mode after checkpoint load overwrote it
        optimizer.train()
    except Exception as e:
        print(f"Starting fresh (checkpoint load failed: {e})")

    # Initialize replay buffer (now NumPy-based, no device needed)
    replay_buffer = ReplayBuffer(
        REPLAY_BUFFER_SIZE, state_shape=INPUT_SHAPE, dtype=np.float32
    )

    model.train()
    for episode in range(NUM_EPISODES):
        obs, info = env.reset()
        state_cur = preprocess_state(obs)
        total_reward = 0.0
        done = False

        while not done:
            total_env_steps += 4  # Each environment step represents 4 frames

            epsilon = epsilon_by_step(total_env_steps)
            # Use torch's RNG for consistent random number generation across all randomness
            if torch.rand(1).item() < epsilon:
                action = env.action_space.sample()
            else:
                state_t = torch.tensor(
                    state_cur, dtype=torch.float32, device=device
                ).unsqueeze(0)
                q_values = model(state_t)
                action = int(torch.argmax(q_values, dim=1).item())

            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_state_cur = preprocess_state(next_obs)

            replay_buffer.push(state_cur, action, reward, next_state_cur, done)
            state_cur = next_state_cur
            total_reward += reward

            if len(replay_buffer) >= MIN_REPLAY_SIZE and (
                total_env_steps % TRAIN_FREQ == 0
            ):
                num_gradient_updates += 1

                # Sample and prepare tensors
                (
                    batch_state_np,
                    batch_action_np,
                    batch_reward_np,
                    batch_next_state_np,
                    batch_done_np,
                ) = replay_buffer.sample(BATCH_SIZE)

                # Convert to torch tensors with correct shapes
                batch_state = torch.from_numpy(batch_state_np).to(
                    device
                )  # [32, 4, 84, 84]
                batch_action = (
                    torch.from_numpy(batch_action_np).long().to(device)
                )  # [32]
                batch_reward = torch.from_numpy(batch_reward_np).to(device)  # [32]
                batch_next_state = torch.from_numpy(batch_next_state_np).to(
                    device
                )  # [32, 4, 84, 84]
                batch_done = torch.from_numpy(batch_done_np.astype(np.float32)).to(
                    device
                )  # [32]

                # Add batch dimension where needed
                batch_action = batch_action.unsqueeze(1)  # [32, 1]
                batch_reward = batch_reward.unsqueeze(1)  # [32, 1]
                batch_done = batch_done.unsqueeze(1)  # [32, 1]

                # Debug shapes (optional)
                # print(f"Shapes: state={batch_state.shape}, action={batch_action.shape}, "
                #       f"reward={batch_reward.shape}, done={batch_done.shape}")

                # Compute current Q-values (shape: [32, 1])
                q_value = model(batch_state).gather(1, batch_action)

                # Compute target Q-values (shape: [32, 1])
                with torch.no_grad():
                    max_next_q = target_model(batch_next_state).max(
                        dim=1, keepdim=True
                    )[0]
                    target = batch_reward + GAMMA * max_next_q * (1 - batch_done)

                loss = F.mse_loss(q_value, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if num_gradient_updates % 100 == 0:
                    print(
                        f"Step: {total_env_steps}, Updates: {num_gradient_updates}, "
                        f"Loss: {loss.item():.4f}, Current Reward: {total_reward:.1f}, Epsilon: {epsilon:.3f}"
                    )

            # Update target network periodically
            if total_env_steps % TARGET_UPDATE_FREQ == 0:
                update_target(model, target_model)
                print(
                    f"Target network updated - Step: {total_env_steps}, "
                    f"Updates: {num_gradient_updates}, Current Reward: {total_reward:.1f}"
                )

            # Save checkpoint every 10k env steps
            if total_env_steps % 10000 == 0:
                save_checkpoint(
                    model,
                    target_model,
                    optimizer,
                    total_env_steps,
                    num_gradient_updates,
                    filename="dqn_checkpoint.pth",
                )

        print(
            f"Episode {episode + 1}: Steps: {total_env_steps}, "
            f"Final Reward: {total_reward:.1f}, Updates: {num_gradient_updates}, Epsilon: {epsilon:.3f}"
        )

    env.close()
    # Final checkpoint save
    save_checkpoint(
        model, target_model, optimizer, total_env_steps, num_gradient_updates
    )


if __name__ == "__main__":
    main()
