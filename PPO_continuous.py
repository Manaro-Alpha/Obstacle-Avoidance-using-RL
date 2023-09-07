import os
import random
import time
from distutils.util import strtobool
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from custom_env.envs import ObsAv_Env

class ActorCritic(nn.Module):
    def __init__(self,obs_dim,action_dim,action_std):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.actor = nn.Sequential(
            nn.Linear(self.obs_dim,64),
            nn.Tanh(),
            nn.Linear(64,64),
            nn.Tanh(),
            nn.Linear(64,self,action_dim)
        )

        self.critic = nn.Sequential(
            nn.Linear(self.obs_dim,64),
            nn.Tanh(),
            nn.Linear(64,64),
            nn.Tanh(),
            nn.Linear(64,1)
        )

        self.actor_std = torch.full((self.action_dim),action_std) ## dimension might be incorrect

    def get_value(self,obs):
        return self.critic(obs)
    
    def get_actionAndvalue(self,obs,action = None):
        action_mean = self.actor(obs)
        action_std = self.actor_std
        dist = Normal(action_mean,action_std)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action).sum(1), dist.entropy().sum(1), self.critic(obs)
    
def rollout(envs,obs_buffer,values_buffer,action_buffer,rewards_buffer,dones_buffer,logprobs_buffer,obs,done,step,global_step,agent:ActorCritic,writer,device):
    obs_buffer[step] = obs
    dones_buffer[step] = done
    with torch.no_grad():
        action,logprob,_,value = agent.get_actionAndvalue(obs)
    action_buffer[step] = action
    values_buffer[step] = value
    logprobs_buffer[step] = logprob

    next_obs,reward,next_done,infos = envs.step(action)
    rewards_buffer[step] = torch.Tensor(reward).to(device)
    obs = torch.Tensor(next_obs).to(device)
    done = torch.Tensor(next_done).to(device)

    for info in infos["final_info"]:
        if info is None:
            continue
        print(f"global_step = {global_step}, episodic_return = {info['episode']['r']}")
        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

def train(envs,agent:ActorCritic,optim,obs_buffer,values_buffer,action_buffer,rewards_buffer,dones_buffer,logprobs_buffer,advantage_buffer,reward_togo,next_obs,next_done,gamma,num_steps,batch_size,num_epochs,mbatch_size,clip_coef):
    with torch.no_grad():
        for step in reversed(range(num_steps)):
            if step == num_steps-1:
                next_nonterminal = next_done ## get next_done and next_nonterminal values by running envs.step one last time
                next_value = agent.get_value(next_obs)
            else:
                next_nonterminal = 1 - dones_buffer[step+1]
                next_value = values_buffer[step+1]
            Q_value = rewards_buffer[step] + gamma*next_nonterminal*next_value
            advantage_buffer[step] = Q_value - values_buffer[step]
            reward_togo[step] = Q_value
        reward_togo = (reward_togo - reward_togo.mean())/(reward_togo.std() + 1e-8)
    
    b_obs = obs_buffer.reshape((-1,) + envs.single_observation_space.shape)
    b_logprobs = logprobs_buffer.reshape(-1)
    b_actions = action_buffer.reshape((-1,) + envs.single_action_space.shape)
    b_advantages = advantage_buffer.reshape(-1)
    b_reward_togo = reward_togo.reshape(-1)
    b_values = values_buffer.reshape(-1)

    b_inds = np.arange(batch_size)

    for epoch in range(num_epochs):
        np.random.shuffle(b_inds)
        for start in range(0,batch_size,mbatch_size):
            end = start + mbatch_size
            mb_inds = b_inds[start:end]

            _,newlogprob,entropy,newvalues = agent.get_actionAndvalue(b_obs[mb_inds],b_actions[mb_inds])
            ratio = (newlogprob - b_logprobs[mb_inds]).exp()
            mb_advantages = b_advantages[mb_inds]

            pg_loss1 = mb_advantages*ratio
            pg_loss2 = mb_advantages*torch.clamp(ratio,1-clip_coef,1+clip_coef)
            pg_loss = torch.min(pg_loss1,pg_loss2).mean()

            newvalues = newvalues.reshape((-1,))
            v_loss = 0.5*((newvalues - b_reward_togo[mb_inds])**2).mean()

            entropy_loss = entropy.mean()
            loss = -pg_loss - 0.01*entropy_loss + 0.5*v_loss

            optim.zero_grad()
            loss.backward()
            optim.step()

def save(agent:ActorCritic,path):
    torch.save(agent.state_dict(),path)

def make_env(env_id,idx,run_name,gamma):
    def thunk():
        env = gym.make(env_id,render_mode = "rgb_array")
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

def PPO(env_id,num_envs,num_steps,learning_rate,gamma,total_timesteps,batch_size,num_epochs,mbatch_size,clip_coef,save_freq):
    run_name = f"{env_id}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    envs = gym.vector.SyncVectorEnv([make_env(env_id,i,run_name,gamma) for i in range(num_envs)])

    agent = ActorCritic(envs.single_observation_space.shape,envs.single_action_space.shape,1).to(device)
    optimizer = optim.Adam(agent.parameters(),lr = learning_rate, eps=1e-5)

    obs_buffer = torch.zeros((num_steps,num_envs) + envs.single_observation_space.shape).to(device)
    action_buffer = torch.zeros((num_steps,num_envs) + envs.single_action_space.shape).to(device)
    logprobs_buffer = torch.zeros((num_steps,num_envs)).to(device)
    rewards_buffer = torch.zeros((num_steps,num_envs)).to(device)
    dones_buffer = torch.zeros((num_steps,num_envs)).to(device)
    values_buffer = torch.zeros((num_steps,num_envs)).to(device)

    global_step = 0
    obs,_ = envs.reset()
    obs = torch.Tensor(obs).to(device)
    done = torch.zeros(num_envs).to(device)
    num_updates = int(total_timesteps/batch_size)

    for update in range(num_updates):
        for step in range(num_steps):
            global_step += num_envs
            rollout(envs,obs_buffer,values_buffer,action_buffer,rewards_buffer,dones_buffer,logprobs_buffer,obs,done,step,global_step,agent,writer)
        
        next_obs = obs_buffer[num_steps-1]
        _,_,next_done,_ = envs.step(action_buffer[num_steps-1])
        advantages = torch.zeros_like(rewards_buffer).to(device).detach()
        reward_togo = torch.zeros_like(rewards_buffer).to(device).detach()
        train(envs,agent,optimizer,obs_buffer,values_buffer,action_buffer,rewards_buffer,dones_buffer,logprobs_buffer,advantages,reward_togo,
              next_obs,next_done,gamma,num_steps,batch_size,num_epochs,mbatch_size,clip_coef)
        
        if update % save_freq == 0:
            save(agent,path="runs/policy")

if __name__ == "__main__":
    env_id = "custom_env/ExpWorld-v1"
    PPO(env_id = env_id,
        num_envs = 5,
        num_steps = 2000,
        learning_rate=3e-4,
        gamma=0.99,
        total_timesteps=int(1e6),
        batch_size = 5*2000,
        num_epochs=10,
        mbatch_size=250,
        clip_coef=0.2,
        save_freq=50)


        
    


    





