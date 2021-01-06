from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
import numpy as np
class Unity_Env(object):
    def __init__(self, file_name="", no_graphics=False, time_scale=20):
        self.engine_configuration_channel = EngineConfigurationChannel()
        if file_name == "":
            self.env = UnityEnvironment(side_channels=[self.engine_configuration_channel])
        else:
            self.env = UnityEnvironment(file_name=file_name, no_graphics=
            no_graphics, side_channels=[self.engine_configuration_channel])

        self.engine_configuration_channel.set_configuration_parameters(
            width = 800,
            height = 512,
            # quality_level = 5, #1-5
            time_scale = time_scale  # 1-100, 10执行一轮的时间约为10秒，20执行一轮的时间约为5秒。
            # target_frame_rate = 60, #1-60
            # capture_frame_rate = 60 #default 60
        )

        self.reset()

        self.team_number = len(self.env.get_behavior_names())
        self.team_name = self.env.get_behavior_names()
        self.agent_number, self.agent_number_for_each_team = self.agent_num()

        self.state_shapes_for_each_team = [self.env.get_behavior_spec(behavior_name).observation_shapes[0][0] for behavior_name in
                             self.env.get_behavior_names()]
        self.action_dims_for_each_team = [self.env.get_behavior_spec(behavior_name).action_shape for behavior_name in
                             self.env.get_behavior_names()]
        self.action_type_for_each_team = [self.env.get_behavior_spec(behavior_name).is_action_continuous() for behavior_name in
                             self.env.get_behavior_names()]


    def agent_num(self):
        agent_num = 0
        agent_number_for_each_team = []
        behavior_names = self.env.get_behavior_names()
        for behavior_name in behavior_names:
            DecisionSteps, TerminalSteps = self.env.get_steps(behavior_name)
            agent_num = agent_num + len(DecisionSteps.agent_id)
            agent_number_for_each_team.append(len(DecisionSteps.agent_id))
        return agent_num, agent_number_for_each_team

    def reset(self):
        self.env.reset()
        cur_state = []
        for behavior_name in self.env.get_behavior_names():
            DecisionSteps, TerminalSteps = self.env.get_steps(behavior_name)
            cur_state.append(DecisionSteps.obs[0])
        return cur_state #team_number x (agent_number_for_each_team, obs_length)

    def step(self, actions):
        next_state = []
        reward = []
        done = []
        for behavior_name_index, behavior_name in enumerate(self.env.get_behavior_names()):
            self.env.set_actions(behavior_name=behavior_name, action=np.asarray(actions[behavior_name_index]))
        self.env.step()

        for i, behavior_name in enumerate(self.env.get_behavior_names()):
            DecisionSteps, TerminalSteps = self.env.get_steps(behavior_name)
            if len(TerminalSteps.reward) == 0:
                next_state.append(DecisionSteps.obs[0])
                reward.append([[reward] for reward in DecisionSteps.reward])
                done.append([[False] for j in range(self.agent_number_for_each_team[i])])
            else:
                next_state.append(TerminalSteps.obs[0])
                reward.append([[reward] for reward in TerminalSteps.reward])
                done.append([[True] for j in range(self.agent_number_for_each_team[i])])

        return next_state, reward, done

    def close(self):
        self.env.close()

def try_env():
    env = UnityEnvironment()
    env.reset()
    behavior_names = env.get_behavior_names()
    print(len(behavior_names))
    for behavior_name in behavior_names:
        BehaviorSpec = env.get_behavior_spec(behavior_name)
        print(behavior_name+" "+str(BehaviorSpec))
        print(BehaviorSpec.is_action_continuous())
        DecisionSteps, TerminalSteps = env.get_steps(behavior_name)

        print(DecisionSteps.agent_id)
        print(DecisionSteps.reward)
        print(DecisionSteps.obs)
        #print(TerminalSteps.agent_id)
        #print(TerminalSteps.reward)
        #print(TerminalSteps.obs)
    env.close()


import time
if __name__ == "__main__":
    env = Unity_Env()
    print(env.agent_number_for_each_team)
    cur_state = env.reset()
    print(cur_state)
    step = 0
    cur_time = time.time()
    while True:
        actions = []
        for i in range(env.team_number):
            actions.append(np.random.uniform(-1, 1, (env.agent_number_for_each_team[i], 2)))
        next_state, reward, done = env.step(actions)
        step = step + 1
        if all(done):
            print(str(step), str(done), str(time.time()-cur_time))
            break
    #try_env()


