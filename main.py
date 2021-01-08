import argparse
import os
import time
import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PPO import PPO
from unity_wrapper_zzy import UnityWrapper

parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='test', action='store_true', default=False)
args = parser.parse_args()

#####################  hyper parameters  ####################

ENV_ID = 'CarVerification-2'  # environment id
RANDOM_SEED = 1  # random seed
RENDER = True  # render while training

ALG_NAME = 'PPO'
TRAIN_EPISODES = 3000  # total number of episodes for training
TEST_EPISODES = 10  # total number of episodes for testing
MAX_STEPS = 200  # total number of steps for each episode
# GAMMA = 0.9  # reward discount
# LR_A = 0.00001  # learning rate for actor0.00001
# LR_C = 0.00002  # learning rate for critic0.0005
BATCH_SIZE = 32  # update batch size
# ACTOR_UPDATE_STEPS = 10  # actor update steps
# CRITIC_UPDATE_STEPS = 10  # critic update steps
#
# # ppo-penalty parameters
# KL_TARGET = 0.01
# LAM = 0.5
#
# # ppo-clip parameters
# EPSILON = 0.2

LogDir = os.path.join("logs/"+ENV_ID)

if __name__ == '__main__':
    if RENDER:  # 流畅渲染，看训练效果
        env = UnityWrapper(train_mode=False, base_port=5004)
    else:
        env = UnityWrapper(train_mode=True, base_port=5004)
    obs_shape_list, d_action_dim, c_action_dim = env.init()
    print("obs_shape_list", obs_shape_list)
    print("d_action_dim:", d_action_dim)
    print("c_action_dim:", c_action_dim)

    agent = PPO(obs_shape_list[0], d_action_dim, c_action_dim, 1)

    t0 = time.time()
    if args.train:
        summary_writer = tf.summary.create_file_writer(LogDir)
        all_episode_reward = []
        for episode in range(TRAIN_EPISODES):
            obs_list = env.reset()
            state = obs_list[0][0]
            # print("state:", state)
            # print("state_dim:", state.shape)
            # print("type_state:", type(state))
            n_agents = obs_list[0].shape[0]
            # print("n_agent:", n_agents)
            episode_reward = 0
            for step in range(MAX_STEPS):  # in one episode
                if d_action_dim != 0:  # 离散动作
                    d_action = agent.get_action(state, d_action_dim)
                    # print("d_action:", d_action)  # d_action: [3]
                    d_action_to_unity = np.eye(d_action_dim, dtype=np.int32)[d_action]
                    # print("d_action_toUnity:", d_action_to_unity)
                    obs_list, reward, done, max_step = env.step(d_action_to_unity, None)
                    agent.store_transition(state, d_action, reward)
                else:  # 连续动作
                    c_action = agent.get_action(state, d_action_dim)
                    # print("c_action:", c_action)  # c_action: [1. 1.]
                    # print("type_c_action:", type(c_action))
                    c_action_to_unity = c_action[np.newaxis, :]
                    # print("c_action_to_unity", c_action_to_unity)
                    obs_list, reward, done, max_step = env.step(None, c_action_to_unity)
                    # print("obs_list", obs_list)
                    # print("reward:", reward)  # [0.]
                    agent.store_transition(state, c_action, reward)

                # print("state_buffer:", agent.state_buffer)
                # print("length:", len(agent.state_buffer))
                # print("action_buffer:", agent.action_buffer)
                # print("len:", len(agent.action_buffer))
                # print("reward_buffer:", agent.reward_buffer)
                # print("len:", len(agent.reward_buffer))

                state = obs_list[0][0]  # state=state_
                episode_reward += reward[0]

                # update ppo
                if len(agent.state_buffer) >= BATCH_SIZE:
                    agent.finish_path(obs_list[0][0], done)
                    agent.update(d_action_dim)
                if done:
                    break
            agent.finish_path(obs_list[0][0], done)
            print(
                'Training  | Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                    episode + 1, TRAIN_EPISODES, episode_reward, time.time() - t0)
            )
            if episode == 0:
                all_episode_reward.append(episode_reward)
            else:
                all_episode_reward.append(all_episode_reward[-1] * 0.9 + episode_reward * 0.1)

            with summary_writer.as_default():  # 希望使用的记录器
                tf.summary.scalar('reward', episode_reward, step=episode)
        agent.save(ALG_NAME, ENV_ID)

        plt.plot(all_episode_reward)
        if not os.path.exists('image'):
            os.makedirs('image')
        plt.savefig(os.path.join('image', '_'.join([ALG_NAME, ENV_ID])))

    if args.test:
        # test
        agent.load(ALG_NAME, ENV_ID)
        average_success_rate = 0
        for episode in range(TEST_EPISODES):
            obs_list = env.reset()
            state = obs_list[0][0]
            # print("state:", state)
            # print("state_dim:", state.shape)
            n_agents = obs_list[0].shape[0]
            episode_reward = 0
            success_tag = 0
            for step in range(MAX_STEPS):
                if d_action_dim != 0:  # 离散动作
                    d_action = agent.get_action(state, d_action_dim)
                    # print("d_action:", d_action)  # d_action: [3]
                    d_action_to_unity = np.eye(d_action_dim, dtype=np.int32)[d_action]
                    # print("d_action_toUnity:", d_action_to_unity)
                    obs_list, reward, done, max_step = env.step(d_action_to_unity, None)
                    if done:
                        success_tag += 1
                    state = obs_list[0][0]
                    episode_reward += reward[0]
                else:  # 连续动作
                    c_action = agent.get_action(state, d_action_dim)
                    # print("c_action:", c_action)  # c_action: [1. 1.]
                    # print("type_c_action:", type(c_action))
                    c_action_to_unity = c_action[np.newaxis, :]
                    # print("c_action_to_unity", c_action_to_unity)
                    obs_list, reward, done, max_step = env.step(None, c_action_to_unity)
                    if done:
                        success_tag += 1
                state = obs_list[0][0]  # state=state_
                episode_reward += reward[0]
                if done:
                    break
            print("success:", success_tag)
            average_success_rate += success_tag
            print(
                'Testing  | Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                    episode + 1, TEST_EPISODES, episode_reward,
                    time.time() - t0))
        print('Testing  | Average Success Rate: {}'.format(
                    average_success_rate/TEST_EPISODES))