from typing import List, Tuple

import gym
import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer


class ReplayBufferWithDemonstrations(ReplayBuffer):
    def __init__(self, env: gym.Env, buffer_size: int, expert_demonstrations):
        super(ReplayBufferWithDemonstrations, self).__init__(buffer_size,
                                                             observation_space=env.observation_space,
                                                             action_space=env.action_space)
        self.handle_timeout_termination = False
        # Add expert demonstrations to the replay buffer
        self.add_expert_demonstrations(expert_demonstrations)

    def add_expert_demonstrations(self, expert_demonstrations):
        # Add expert demonstrations to the replay buffer
        for state, state_next, action, reward, done in expert_demonstrations:
            self.add(state, state_next, action, reward, done, [])


def parse_old_expert_data(expert_data_path: str) \
        -> List[Tuple[np.ndarray[np.float64], np.ndarray[np.float64], np.int32, np.float64, bool]]:
    # numpy_dict = {
    #     'actions': actions,
    #     'obs': observations,
    #     'rewards': rewards,
    #     'episode_returns': episode_returns,
    #     'episode_starts': episode_starts
    # }
    expert_data = np.load(expert_data_path)
    actions = expert_data['actions']
    obs = expert_data['obs']
    rewards = expert_data['rewards']
    episode_returns = expert_data['episode_returns']
    episode_starts = expert_data['episode_starts']

    # expert_demonstrations is a list of transitions (s, s\', a, r, done)
    expert_demonstrations: List[Tuple[np.ndarray[np.float64], np.ndarray[np.float64], np.int32, np.float64, bool]] = []
    terminal_count = 0
    for i in range(len(actions) - 1):
        state = obs[i]
        state_next = obs[i + 1]
        action = actions[i]
        reward = rewards[i]
        done = episode_starts[i]
        # TODO validate that we should be adding in the episode return
        if done:
            reward += episode_returns[terminal_count]
            terminal_count += 1
        expert_demonstrations.append((state, state_next, action, reward, done))

    return expert_demonstrations


def configure_imitation(model, env, expert_path, model_path, learning_rate, n_epochs=1000):
    expert_demonstrations = parse_old_expert_data(expert_path)
    # Fill the replay buffer with expert demonstrations. Kinda a hacky way to learn from expert demonstrations that
    # doesn't have a ton of theoretical basis, but it's how the code originally did this and seems to work in practice
    replay_buffer = ReplayBufferWithDemonstrations(env, len(expert_demonstrations) * 2, expert_demonstrations)

    # Set the replay buffer for the model
    model.replay_buffer = replay_buffer

    # TODO maybe replace pretraining with some other form of behavioral cloning?
    #  https://github.com/DLR-RM/stable-baselines3/issues/27
    # model.pretrain(n_epochs=n_epochs, learning_rate=learning_rate)
