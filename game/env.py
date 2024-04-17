import time
from typing import Dict, Tuple, Any

import gymnasium
import numpy as np
from gymnasium import spaces
from selenium import webdriver
from selenium.webdriver import Keys, ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from stable_baselines3.common.env_checker import check_env

PORT = 8000
PRESS_DURATION = 0.1
MAX_EPISODE_DURATION_SECS = 120
STATE_SPACE_N = 71
ACTIONS = {
    0: 'qw',
    1: 'qo',
    2: 'qp',
    3: 'q',
    4: 'wo',
    5: 'wp',
    6: 'w',
    7: 'op',
    8: 'o',
    9: 'p',
    10: '',
}


class QWOPEnv(gymnasium.Env):

    meta_data = {'render.modes': ['human']}
    pressed_keys = set()

    def __init__(self, render_mode='human', intermediate_rewards=True, fine_tune=False):
        self.intermediate_rewards = intermediate_rewards
        self.fine_tune = fine_tune

        # Open AI gym specifications
        super(QWOPEnv, self).__init__()
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=[STATE_SPACE_N], dtype=np.float32
        )
        self.num_envs = 1

        # QWOP specific stuff
        self.gameover = False
        self.previous_score = 0
        self.previous_time = 0
        self.previous_torso_x = 0
        self.previous_torso_y = 0
        self.evoke_actions = True

        # Open browser and go to QWOP page
        options = Options()
        if render_mode is None:
            self.headless = True
            options.add_argument('--headless=new')
        else:
            self.headless = False
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        self.driver = webdriver.Chrome(options=options)
        self.driver.get(f'http://localhost:{PORT}/Athletics.html')

        # Wait a bit and then start game
        self.driver.implicitly_wait(10)
        window = self.driver.find_element(By.ID, "window1")

        self.keyboard = ActionChains(self.driver)
        self.keyboard.move_to_element(window)

        # Click twice to get past the start screen
        start = time.time()
        while time.time() < start + 1:
            self.keyboard.click().perform()
        self.last_press_time = time.time()

    def _get_variable_(self, var_name):
        return self.driver.execute_script(f'return {var_name};')

    def _get_state_(self):
        # Since the framerate may or may not be consistent

        game_state = self._get_variable_('globalgamestate')
        body_state = self._get_variable_('globalbodystate')

        # Get done
        if game_state['gameEnded'] > 0 or game_state['gameOver'] > 0:
            self.gameover = True
            truncated = False
        elif game_state['scoreTime'] > MAX_EPISODE_DURATION_SECS:
            self.gameover = True
            truncated = True
        else:
            self.gameover = False
            truncated = False

        # Get reward

        # Position is measured in decimeters, amusingly
        torso_position_x = body_state['torso']['position_x']
        torso_position_y = body_state['torso']['position_y']
        torso_velocity_x = body_state['torso']['linear_velocity_x']

        # Time is measured in seconds
        time = game_state['scoreTime']
        # Prevents divide by zero. In practice, this only happens if the game gets paused for some reason (usually
        # going out of focus when rendering with human rendering), but it's worth having to keep from crashing anyways
        dt = max(time - self.previous_time, 1 / 60)

        # Reward for moving forward
        x_movement = torso_position_x - self.previous_torso_x

        if self.fine_tune:
            reward_forward = max(torso_position_x - self.previous_torso_x, 0) * 2
            reward_velocity = torso_velocity_x / 10

            # # Boost the terminal state based on total time to complete the run and the jump distance
            # if self.gameover and not truncated:
            #     reward_overall_velocity = 5000 / time
            #     reward_jump = (torso_x - 1000) * 100
            #     reward_terminal = reward_overall_velocity + reward_jump
            # else:
            #     reward_terminal = 0
            reward_terminal = 0

            reward = reward_forward + reward_velocity + reward_terminal
        else:
            reward_forward = max(x_movement, 0) * 2
            reward_velocity = torso_velocity_x / 10

            # Penalize for low torso
            if torso_position_y > 0:
                penalty_low = -torso_position_y / 5
            else:
                penalty_low = 0

            if self.intermediate_rewards:
                # Penalize for torso vertical velocity
                penalty_falling = -abs(torso_position_y - self.previous_torso_y) / 4

                # Penalize for bending knees too much
                if (
                    body_state['joints']['leftKnee'] < -0.9
                    or body_state['joints']['rightKnee'] < -0.9
                ):
                    penalty_bending = (
                        min(body_state['joints']['leftKnee'], body_state['joints']['rightKnee'])
                        / 6
                    )
                else:
                    penalty_bending = 0
            else:
                penalty_falling = penalty_bending = 0

            # Combine rewards
            reward = reward_forward + reward_velocity + penalty_low + penalty_falling + penalty_bending

        # print(
        #     'Rewards: {:3.1f}, {:3.1f}, {:3.1f}, {:3.1f}, {:3.1f}'.format(
        #         reward1, reward2, reward3, reward4, reward
        #     )
        # )

        # Update previous scores
        self.previous_torso_x = torso_position_x
        self.previous_torso_y = torso_position_y
        self.previous_score = game_state['score']
        self.previous_time = time

        # Normalize torso_x
        for part, values in body_state.items():
            if 'position_x' in values:
                values['position_x'] -= torso_position_x

        # print('Positions: {:3.1f}, {:3.1f}, {:3.1f}'.format(
        #     body_state['torso']['position_x'],
        #     body_state['leftThigh']['position_x'],
        #     body_state['rightCalf']['position_x']
        # ))

        # print('Knee angles: {:3.2f}, {:3.2f}'.format(
        #     body_state['joints']['leftKnee'],
        #     body_state['joints']['rightKnee']
        # ))

        # Convert body state
        state = []
        for part in body_state.values():
            state = state + list(part.values())
        state = np.array(state)

        return state, reward, self.gameover, truncated, {}

    def _release_all_keys_(self):

        for char in self.pressed_keys:
            self.keyboard.key_up(char).perform()

        self.pressed_keys.clear()

    def send_keys(self, keys):

        # Release all keys
        self._release_all_keys_()

        # Hold down current key
        for char in keys:
            self.keyboard.key_down(char).perform()
            self.pressed_keys.add(char)

        # print('pressed for', time.time() - self.last_press_time)
        # self.last_press_time = time.time()
        time.sleep(PRESS_DURATION)

    def reset(self, **kwargs):

        # Send 'R' key press to restart game
        self.send_keys(['r', Keys.SPACE])
        self.gameover = False
        self.previous_score = 0
        self.previous_time = 0
        self.previous_torso_x = 0
        self.previous_torso_y = 0
        self._release_all_keys_()

        observation, _, _, _, info = self._get_state_()
        return observation, info

    def step(self, action_id):

        # send action
        keys = ACTIONS[action_id]

        if self.evoke_actions:
            self.send_keys(keys)
        else:
            time.sleep(PRESS_DURATION)

        return self._get_state_()

    def render(self, mode='human'):
        # Unfortunately Selenium cannot switch between headless and rendering, so all we can do is validate that we're
        # in the correct rendering mode
        if mode == 'human' and self.headless:
            print("WARNING: Selenium is currently in headless mode, but the rendering mode is set to human. "
                  "You will not see any output.")
        elif mode is None and not self.headless:
            print("WARNING: The rendering mode is currently set to None, but Selenium is not set to headless.")

    def close(self):
        pass


if __name__ == '__main__':
    env = QWOPEnv()
    check_env(env)
    while True:
        if env.gameover:
            env.reset()
        else:
            env.step(env.action_space.sample())
