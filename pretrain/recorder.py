import time

# from imitation.testing.expert_trajectories import generate_expert_trajectories

env = None
model = None

MAPPING = {
    'qw': 0,
    'qo': 1,
    'qp': 2,
    'q': 3,
    'wo': 4,
    'wp': 5,
    'w': 6,
    'op': 7,
    'o': 8,
    '': 10,
}


def dummy_expert(_obs):

    global env

    time.sleep(0.5)
    print(_obs)
    print(_obs.shape)

    return env.action_space.sample()


def human_expert(_obs):

    env.evoke_actions = False
    game_state = env._get_variable_('globalgamestate')

    string = ''
    for char in ['q', 'w', 'o', 'p']:
        if game_state[char]:
            string = string + char
    if 'q' in string and 'p' in string:
        string = 'qp'
    if 'w' in string and 'o' in string:
        string = 'wo'
    string = string[:2]
    for key, value in MAPPING.items():
        if set(string) == set(key):
            # print(f'returning {value} for {key}')
            return value

    raise ValueError(f'Key presses not found {string}')


def get_existing_model(model_path):

    print('--- Training from existing model', model_path, '---')

    # Load model
    model = SAC.load(model_path)

    return model


def sac_expert(_obs):
    global model
    action, _states = model.predict(_obs)
    return action


def generate_obs(environment, record_path, n_episodes):
    global env, model
    env = environment

    print('Starting record...')
    # TODO rewrite me
    # model = get_existing_model(os.path.join('models', 'Self6hr_human50_self114hr'))
    # generate_expert_traj(acer_expert, record_path, env, n_episodes=n_episodes)
    # trajectories = generate_expert_trajectories(env, n_episodes, human_expert)
    print(
        f'Recording of {n_episodes} episodes complete. Saved file to {record_path}.npz'
    )