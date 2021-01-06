import tensorflow as tf
import logging
from unity_wrapper_zzy import UnityWrapper
import numpy as np
import os
from PPO import PPO
#####################  hyper parameters  ####################
ENV_ID = 'CarVerification-1'  # environment id
ALG_NAME = 'PPO'
TEST_EPISODES = 1000  # total number of episodes for testing



def LoadModel(self):
    model_path = os.path.join('../../model', '_'.join([ALG_NAME, ENV_ID]))
    model = tf.keras.models.load_model(model_path)
    model.summary()  # #输出模型各层的参数状况
    return model

def GetAction(self):
    model.predict()


if __name__ == '__main__':


    logging.basicConfig(level=logging.DEBUG)
    env = UnityWrapper(train_mode=True, base_port=5004)
    obs_shape_list, d_action_dim, c_action_dim = env.init()
    print("obs_shape_list", obs_shape_list)
    print("type:", type(obs_shape_list[0]))
    print("dim:", len(obs_shape_list[0]))
    print("d_action_dim:", d_action_dim)
    print("c_action_dim:", c_action_dim)
    agent = PPO(obs_shape_list[0], d_action_dim, c_action_dim, 1)
    agent.load()

    for episode in range(10):
        obs_list = env.reset()
        print("obs_list:", obs_list)
        n_agents = obs_list[0].shape[0]
        print("n_agents:", n_agents)
        for step in range(100):
            d_action, c_action = None, None
            if d_action_dim:
                d_action = np.random.randint(0, d_action_dim, size=n_agents)
                print("d_action:", d_action)  # d_action: [4]
                d_action = np.eye(d_action_dim, dtype=np.int32)[d_action]
                print("d_action_onehot:", d_action)  # d_action_onehot: [[0 0 0 0 1]]
            if c_action_dim:
                c_action = np.random.randn(n_agents, c_action_dim)
                print("c_action:", c_action)  # [[1.50210385 0.39819464]]
            print("episode:", episode, ",  step:", step)
            obs_list, reward, done, max_step = env.step(d_action, c_action)
            print("obs_list:", obs_list)
            print("reward:", reward)  # reward: [-0.0025]

    env.close()