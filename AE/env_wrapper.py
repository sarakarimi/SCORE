import cv2
import gym
import d4rl
import mujoco_py
import numpy
import numpy as np
import h5py
from tqdm import tqdm


def antmaze_costume_env():
    # import gymnasium as gym

    example_map = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 'r', 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
        [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
        [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
        [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
        [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
        [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
        [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
        [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 'g', 1, 1, 1],
        [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
    # R = 'r'  # Reset position.
    # G = 'g'
    # example_map= [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #                [1, R, 0, 0, 0, 1, G, 0, 0, 0, 0, 0, 1],
    #                [1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1],
    #                [1, 0, 0, 0, 0, G, 0, 1, 0, 0, 0, G, 1],
    #                [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
    #                [1, 0, G, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1],
    #                [1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1],
    #                [1, 0, 0, 1, G, 0, G, 1, 0, 0, G, 0, 1],
    #                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
    env = gym.make('antmaze-large-diverse-v0', maze_map=example_map, reward_type='dense', render_mode='human')
    return env

def get_keys(h5file):
    keys = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)

    h5file.visititems(visitor)
    return keys


def get_dataset(h5path):
    data_dict = {}
    with h5py.File(h5path, 'r') as dataset_file:
        for k in tqdm(get_keys(dataset_file), desc="load datafile"):
            try:  # first try loading as an array
                data_dict[k] = dataset_file[k][:]
            except ValueError as e:  # try loading as a scalar
                data_dict[k] = dataset_file[k][()]
    return data_dict


def get_trajectory(env_name, traj_len, dataset=None, random_start=False):
    list_of_states, list_of_actions = [], []
    if dataset is None:
        env = gym.make(env_name)
        dataset = env.get_dataset()
    if random_start is False:
        dataset_len = len(dataset['observations'])
        print(dataset_len)
        list_of_states = [dataset['observations'][i:i + traj_len] for i in range(0, dataset_len, traj_len)]
        list_of_actions = [dataset['actions'][i:i + traj_len] for i in range(0, dataset_len, traj_len)]
    else:
        num_samples = len(dataset['observations'])
        print(num_samples)
        number_of_traj = int(num_samples / traj_len)
        start_indexes = np.random.randint(0, num_samples - traj_len - 1, size=number_of_traj)
        list_of_states = [dataset['observations'][i:i + traj_len] for i in start_indexes]
        list_of_actions = [dataset['actions'][i:i + traj_len] for i in start_indexes]
        # terms = dataset["rewards"]
        # print(terms[:100])
        # print(len(np.where(terms)[0]))
        # print(np.where(dataset["terminals"]))
        # print(np.where(dataset["timeouts"]))
    return list_of_states, list_of_actions


if __name__ == '__main__':
    # get_trajectory('antmaze-large-diverse-v0', 10)
    # dataset = get_dataset("/home/sara/repositories/RL-skill-extraction/data/Antmaze_largest_multistart_False_multigoal_False.hdf5")
    # d = get_trajectory('coinrun', 10, dataset=dataset, random_start=True)
    # print(np.asarray(d[0]).shape)
    import matplotlib.pyplot as plt
    # from IPython import display
    env = antmaze_costume_env()
    env.reset()
    while True:
        env.render()
    # plt.figure(3)
    # plt.clf()
    # plt.imsave("img.png", env.render(mode='human'))
    # plt.title("%s | Step: %d %s" % (env._spec.id,0, ""))
    # plt.axis('off')


