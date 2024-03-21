from typing import List, Tuple

import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer


class ReplayBufferWithDemonstrations(ReplayBuffer):
    def __init__(self, expert_demonstrations, *args, **kwargs):
        super(ReplayBufferWithDemonstrations, self).__init__(*args, **kwargs)
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


def imitate(model, expert_path, model_path, learning_rate, n_epochs=1000):

    # TODO I think this and the recorder need to be entirely rewritten
    # See https://github.com/HumanCompatibleAI/imitation/blob/a8b079c469bb145d1954814f22488adff944aa0d/docs/tutorials/3_train_gail.ipynb#L7

    expert_demonstrations = parse_old_expert_data(expert_path)
    # Fill the replay buffer with expert demonstrations. Kinda a hacky way to learn from expert demonstrations that
    # doesn't have a ton of theoretical basis, but it's how the code originally did this and seems to work in practice
    replay_buffer = ReplayBufferWithDemonstrations(expert_demonstrations, buffer_size=len(expert_demonstrations))

    # Set the replay buffer for the model
    model.load_replay_buffer(replay_buffer)

    # TODO validate that we train on the replay buffer
    model.pretrain(replay_buffer, n_epochs=n_epochs, learning_rate=learning_rate)

    model.save(model_path)
