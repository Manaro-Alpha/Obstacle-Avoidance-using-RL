import time
import gym
import torch
import numpy as np
from PPO import PPO

def make_env(env_id,idx,run_name,gamma):
    def thunk():
        env = gym.make(env_id,render_mode = "rgb_array")
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        # if idx == 0:
        #     env = gym.wrappers.RecordVideo(env,f"videos/{run_name}")
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env
    return thunk

def train(total_timesteps):
    gamma = 0.999
    env_id = "custom_env/ExpWorld-v1"
    run_name = f"{env_id}__{int(time.time())}"
    envs = gym.vector.SyncVectorEnv([make_env(env_id,i,run_name,gamma) for i in range(1)])
    model = PPO(envs)
    model.env_id = env_id
    model.learn(total_timesteps)
    path = 'PPO_' + "_model_" + str(int(time.time()))
    model.save(path)

if __name__ == '__main__':
    train(1000000)