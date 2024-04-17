import os
import stable_baselines3.common.type_aliases
import time
from enum import Enum
from typing import Tuple, Type

import typer
import gymnasium as gym
from stable_baselines3 import DQN, SAC
from sb3_contrib import SACD
from sb3_contrib.sacd.policies import BasePolicy, SACDPolicy
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.type_aliases import TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.monitor import Monitor
import torch
import numpy as np

from game.env import ACTIONS
from game.env import QWOPEnv
from pretrain import imitation_learning
from pretrain import recorder

# Training parameters
DEFAULT_MODEL_NAME = 'my_model'
DEFAULT_MODEL_PATH = os.path.join('models', DEFAULT_MODEL_NAME)
EXPLORATION_FRACTION = 0.3
LEARNING_STARTS = 3000
EXPLORATION_INITIAL_EPS = 0.01
EXPLORATION_FINAL_EPS = 0.001
BUFFER_SIZE = 300000
BATCH_SIZE = 64
TRAIN_FREQ = TrainFreq(frequency=4, unit=TrainFrequencyUnit.STEP)
LEARNING_RATE = 0.0001
TRAIN_TIME_STEPS = 600000
TENSORBOARD_PATH = './tensorboard/'

# Imitation learning parameters
EXPERT_PATH = os.path.join('pretrain', 'kuro_1_to_5.npz')
N_EPISODES = 10
N_EPOCHS = 500
PRETRAIN_LEARNING_RATE = 0.00001  # 0.0001

app = typer.Typer(no_args_is_help=True)


class CustomDqnPolicy(stable_baselines3.dqn.MlpPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomDqnPolicy, self).__init__(
            *args,
            **kwargs,
            activation_fn=torch.nn.modules.activation.ReLU,  # Should be the same as default
            net_arch=[256, 128],
            normalize_images=True,
        )


class CustomSacPolicy(SACDPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomSacPolicy, self).__init__(
            *args,
            **kwargs,
            activation_fn=torch.nn.modules.activation.ReLU,  # Should be the same as default
            net_arch=[256, 128],
            normalize_images=True,
        )


# From https://stable-baselines3.readthedocs.io/en/master/guide/tensorboard.html
class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        value = np.random.random()
        self.logger.record("random_value", value)
        return True


class EnvType(str, Enum):
    QWOP = "qwop",
    WALKER_2D = "walker_2d"


class RenderMode(str, Enum):
    NONE = "none",
    HUMAN = "human"


class ModelType(str, Enum):
    DQN = "dqn",
    SAC = "sac"


def get_model_type_data(env, model_type: ModelType) -> Tuple[Type[OffPolicyAlgorithm], Type[BasePolicy]]:
    match model_type:
        case ModelType.DQN:
            return DQN, CustomDqnPolicy
        case ModelType.SAC:
            if isinstance(env.action_space, gym.spaces.Discrete):
                return SACD, CustomSacPolicy
            elif isinstance(env.action_space, gym.spaces.Box):
                return SAC, CustomSacPolicy


def get_env(env_type: EnvType, render_mode: RenderMode, fine_tune: bool = False, intermediate_rewards: bool = True):
    if render_mode == "none":
        # This is the way that OpenAI's Gym prefers it, and heterogenous enums don't seem to work well with Typer
        render_mode = None

    match env_type:
        case EnvType.QWOP:
            env = QWOPEnv(render_mode=render_mode, fine_tune=fine_tune, intermediate_rewards=intermediate_rewards)
        case EnvType.WALKER_2D:
            env = gym.make('Walker2d-v4', render_mode=render_mode)
    wrapped_env = Monitor(env, TENSORBOARD_PATH, allow_early_resets=True)
    return wrapped_env


def get_new_model(env, model_type: ModelType):
    model_class, policy = get_model_type_data(env, model_type)
    # prioritized_replay is not implemented in StableBaselines3 so this is fine
    return model_class(policy, env, verbose=1, tensorboard_log=TENSORBOARD_PATH)


def get_existing_model(env, model_type, model_path):
    model_class, policy = get_model_type_data(env, model_type)
    model = model_class.load(model_path, tensorboard_log=TENSORBOARD_PATH)

    # Set environment
    model.set_env(env)

    return model


def get_model(env, model_type: ModelType, model_path: str | None):
    print(model_path)
    if model_path is not None and os.path.isfile(model_path + '.zip'):
        print('--- Training from existing model', model_path, '---')
        model = get_existing_model(env, model_type, model_path)
    else:
        print('--- Training from new model ---')
        model = get_new_model(env, model_type)

    return model


@app.command()
def train(env_type: EnvType = EnvType.QWOP, render_mode: RenderMode = RenderMode.HUMAN,
          model_type: ModelType = ModelType.DQN, load_model_path: str = DEFAULT_MODEL_PATH,
          save_model_path: str = DEFAULT_MODEL_PATH, fine_tune=False,
          intermediate_rewards=True):
    """
    Run training; will train from existing model if path exists.
    """
    env = get_env(env_type, render_mode, fine_tune, intermediate_rewards)

    callbacks = [
        CheckpointCallback(
            save_freq=1000, save_path='./logs/', name_prefix=save_model_path
        ),
        EvalCallback(env, eval_freq=100)
    ]

    model = get_model(env, model_type, load_model_path)
    model.learning_rate = LEARNING_RATE
    model.learning_starts = LEARNING_STARTS
    model.exploration_initial_eps = EXPLORATION_INITIAL_EPS
    model.exploration_final_eps = EXPLORATION_FINAL_EPS
    model.buffer_size = BUFFER_SIZE
    model.batch_size = BATCH_SIZE
    model.train_freq = TRAIN_FREQ

    # Train and save
    t = time.time()

    model.learn(
        total_timesteps=TRAIN_TIME_STEPS,
        callback=callbacks,
        reset_num_timesteps=False,
    )
    model.save(save_model_path)

    print(f"Trained {TRAIN_TIME_STEPS} steps in {time.time() - t} seconds.")


def print_probs(model, obs):
    # Print action probabilities
    probs = model.action_probability(obs)
    topa = sorted(
        [(prob, kv[1]) for kv, prob in zip(ACTIONS.items(), probs)],
        reverse=True,
    )[:3]
    print(
        'Top 3 actions - {}: {:3.0f}%, {}: {:3.0f}%, {}: {:3.0f}%'.format(
            topa[0][1],
            topa[0][0] * 100,
            topa[1][1],
            topa[1][0] * 100,
            topa[2][1],
            topa[2][0] * 100,
        )
    )


@app.command()
def test(env_type: EnvType = EnvType.QWOP,
         render_mode: RenderMode = RenderMode.HUMAN,
         model_type: ModelType = ModelType.DQN,
         load_model_path=DEFAULT_MODEL_PATH):
    """
    Test the model.
    """

    env = get_env(env_type, render_mode)

    model = get_existing_model(env, model_type, load_model_path)

    obs = env.reset()[0]
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, _, done, _, _ = env.step(int(action))


@app.command()
def record(env_type: EnvType = EnvType.QWOP, render_mode: RenderMode = RenderMode.HUMAN, record_path=EXPERT_PATH):
    """
    Record observations for pretraining
    """
    env = get_env(env_type, render_mode)
    recorder.generate_obs(env, record_path, N_EPISODES)


@app.command()
def imitate(env_type: EnvType = EnvType.QWOP, render_mode: RenderMode = RenderMode.HUMAN,
            model_type: ModelType = ModelType.DQN, load_model_path=DEFAULT_MODEL_PATH,
            save_model_path=DEFAULT_MODEL_PATH, expert_path=EXPERT_PATH, fine_tune=False, intermediate_rewards=True):
    """
    Train agent from recordings; will use existing model if path exists
    """
    env = get_env(env_type, render_mode)
    model = get_model(env, model_type, load_model_path)
    imitation_learning.configure_imitation(
        model, env, expert_path, save_model_path, PRETRAIN_LEARNING_RATE, N_EPOCHS
    )

    callbacks = [
        CheckpointCallback(
            save_freq=1000, save_path='./logs/', name_prefix=save_model_path
        ),
        EvalCallback(env, eval_freq=100)
    ]

    # Train and save
    t = time.time()

    model.learn(
        total_timesteps=TRAIN_TIME_STEPS,
        callback=callbacks,
        reset_num_timesteps=False,
    )
    model.save(save_model_path)

    print(f"Imitated {TRAIN_TIME_STEPS} steps in {time.time() - t} seconds.")


if __name__ == '__main__':
    app()
