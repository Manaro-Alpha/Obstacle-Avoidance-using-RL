import time
import gym
import torch
import numpy as np
from PPO import PPO,ActorCritic

def make_env(env_id,idx,run_name,gamma):
    def thunk():
        env = gym.make(env_id,render_mode = "human")
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if idx == 0:
            env = gym.wrappers.RecordVideo(env,f"videos/{run_name}")
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env
    return thunk

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_test_episodes = 10
    max_ep_length = 500
    gamma = 0.999
    env_id = "custom_env/ExpWorld-v1"
    run_name = f"{env_id}__{int(time.time())}"
    envs = gym.vector.SyncVectorEnv([make_env(env_id,i,run_name,gamma) for i in range(1)])
    # env = gym.make("BipedalWalker-v3",render_mode= 'human')
    # obs_dim = env.observation_space.shape[0]
    # action_dim = env.action_space.shape[0]
    test_agent = PPO(envs)
    test_agent.load("PPO__model_1695294304")

    print("##############################")
    test_running_award = 0
    for step in range(1,total_test_episodes+1):
        ep_reward = 0
        obs,_ = envs.reset() 
        for t in range(1,max_ep_length):
            action,_ = test_agent.agent.select_action(torch.tensor(obs,dtype=torch.float,device=device))
            obs,reward,done,_,_ = envs.step(action)
            ep_reward += reward
            if done:
                break
        
        test_running_award += ep_reward
        print(f"episode: {step} \n Reward: {round(ep_reward,2)}")
        ep_reward = 0
    envs.close()

    avg_test_reward = test_running_award/total_test_episodes
    print(f"avg_test_rew: {avg_test_reward}")

if __name__ == "__main__":
    test()