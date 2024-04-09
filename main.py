import os
import stable_baselines3.common.type_aliases
import time

import click
import gymnasium as gym
from stable_baselines3 import DQN, SAC
from sb3_contrib import SACD
from sb3_contrib.sacd.policies import SACDPolicy
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
MODEL_NAME = 'dqn_test_v2'
EXPLORATION_FRACTION = 0.3
LEARNING_STARTS = 3000
EXPLORATION_INITIAL_EPS = 0.01
EXPLORATION_FINAL_EPS = 0.001
BUFFER_SIZE = 300000
BATCH_SIZE = 64
TRAIN_FREQ = TrainFreq(frequency=4, unit=TrainFrequencyUnit.STEP)
LEARNING_RATE = 0.0001
TRAIN_TIME_STEPS = 600000
MODEL_PATH = os.path.join('models', MODEL_NAME)
TENSORBOARD_PATH = './tensorboard/'

# Imitation learning parameters
RECORD_PATH = os.path.join('pretrain', 'kuro_1_to_5')
N_EPISODES = 10
N_EPOCHS = 500
PRETRAIN_LEARNING_RATE = 0.00001  # 0.0001


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
            activation_fn=torch.nn.modules.activation.ReLU, # Should be the same as default
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


def get_env():
    env = QWOPEnv()  # SubprocVecEnv([lambda: QWOPEnv()])
    # env = gym.make('Walker2d-v4')
    return env


def get_new_model(env, fine_tune=False):
    if fine_tune:
        # Initialize env and model
        model = DQN(
            CustomDqnPolicy,
            env,
            # TODO not implemented in stables-baselines3
            # prioritized_replay=True,
            verbose=1,
            tensorboard_log=TENSORBOARD_PATH,
        )

        return model
    else:
        # Initialize env and model
        model = SACD(
            CustomSacPolicy,
            env,
            verbose=1,
            tensorboard_log=TENSORBOARD_PATH,
        )

        return model


def get_existing_model(env, model_path, fine_tune=False):
    if fine_tune:
        model = DQN.load(model_path, tensorboard_log=TENSORBOARD_PATH)
    else:
        model = SACD.load(model_path, tensorboard_log=TENSORBOARD_PATH)

    # Set environment
    model.set_env(env)

    return model


def get_model(env, model_path, fine_tune=False):
    if os.path.isfile(model_path + '.zip'):
        print('--- Training from existing model', model_path, '---')
        model = get_existing_model(env, model_path)
    else:
        print('--- Training from new model ---')
        model = get_new_model(env, fine_tune)

    return model


def run_train(env, model_path=MODEL_PATH, fine_tune=False):
    callbacks = [
        CheckpointCallback(
            save_freq=1000, save_path='./logs/', name_prefix=MODEL_NAME
        ),
        EvalCallback(env, eval_freq=100)
    ]

    model = get_model(env, model_path, fine_tune)
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
    model.save(MODEL_PATH)

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


def run_test(env, fine_tune=False):
    # Initialize model
    if fine_tune:
        model = DQN.load(MODEL_PATH)
    else:
        model = SACD.load(MODEL_PATH)

    obs = env.reset()[0]
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, _, done, _, _ = env.step(int(action))


@click.command()
@click.option(
    '--train',
    default=False,
    is_flag=True,
    help='Run training; will train from existing model if path exists',
)
@click.option('--test', default=False, is_flag=True, help='Run test')
@click.option(
    '--record',
    default=False,
    is_flag=True,
    help='Record observations for pretraining',
)
@click.option(
    '--imitate',
    default=False,
    is_flag=True,
    help='Train agent from recordings; will use existing model if path exists',
)
def main(train, test, record, imitate):
    """Train and test an agent for QWOP."""

    env = QWOPEnv()
    wrapped_env = Monitor(env, TENSORBOARD_PATH, allow_early_resets=True)

    if train:
        run_train(wrapped_env, fine_tune=True)
    if test:
        run_test(env, fine_tune=True)

    if record:
        recorder.generate_obs(env, RECORD_PATH, N_EPISODES)

    if imitate:
        model = get_model(MODEL_PATH)
        imitation_learning.imitate(
            model, RECORD_PATH, MODEL_PATH, PRETRAIN_LEARNING_RATE, N_EPOCHS
        )

    if not (test or train or record or imitate):
        with click.Context(main) as ctx:
            click.echo(main.get_help(ctx))


if __name__ == '__main__':
    main()
