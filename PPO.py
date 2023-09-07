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

class Agent(nn.Module):
    def __init__(self,envs):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(np.prod(envs.single_observation_space.shape),64),
            nn.Tanh(),
            nn.Linear(64,64),
            nn.Tanh,
            nn.Linear(64,1)
        )

        self.actor_mean = nn.Sequential(
            nn.Linear(np.prod(envs.single_observation_space.shape),64),
            nn.Tanh(),
            nn.Linear(64,64),
            nn.Tanh(),
            nn.Linear(64,envs.single_action_space.shape())
        )

        self.actor_logstd = nn.Parameter(torch.zeros(1,np.prod(envs.single_action_space.shape)))

    def get_value(self,obs):
        return self.critic(obs)
    
    def get_action_and_value(self,obs,action=None):
        action_mean = self.actor_mean(obs)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean,action_std)
        if action is None:
            action = probs.sample()
        return action,probs.log_prob(action).sum(1),probs.entropy().sum(1),self.critic(obs)
    
def make_env(env_id, seed):
    def thunk():
        env = gym.make(env_id, render_mode="rgb_array")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

def rollout(envs,obs_buffer,values_buffer,actions_buffer,rewards_buffer,logprobs_buffer,dones_buffer,agent,next_obs,done,step,num_steps,num_envs):
    obs_buffer[step] = next_obs
    dones_buffer[step] = done
    with torch.no_grad():
        action,logprob,_,value = agent.get_action_and_value()
        values_buffer[step] = value.flatten()
    actions_buffer[step] = action
    logprobs_buffer[step] = logprob

    next_obs,reward,done,info = envs.step(action)
    rewards_buffer[step] = reward

def bootstrap(agent,rewards_buffer,dones_buffer,values_buffer,advantages_buffer,next_obs,next_done,next_value,num_steps,gamma,gae_lambda,lastgaelam,step):
    for t in reversed(range(num_steps)):
        if step == num_steps-1:
            nextnonterminal = 1 - next_done
            nextvalues = next_value
        else:
            nextnonterminal = 1 - dones_buffer[step+1]
            nextvalues = values_buffer[step+1]
        delta = rewards_buffer[step] + gamma*nextvalues*nextnonterminal - values_buffer[step]
        advantages_buffer[step] = lastgaelam = delta + gamma*gae_lambda*nextnonterminal*lastgaelam
    returns_buffer = advantages_buffer + values_buffer

def train(envs,agent,obs_buffer,actions_buffer,logprobs_buffer,advantages_buffer,returns_buffer,values_buffer,buffer_size,batch_size,clip_coef,ent_coef,v_coef,optimizer):
    batch_obs = obs_buffer.reshape((-1)+envs.single_observation_space.shape)
    batch_actions = actions_buffer.reshape((-1)+envs.single_observation_space.shape)
    batch_logprobs = logprobs_buffer.reshape(-1)
    batch_advantages =advantages_buffer.reshape(-1)
    batch_returns = returns_buffer.reshape(-1)
    batch_values = values_buffer.reshape(-1)

    batch_i = random.sample(range(buffer_size),batch_size)
    _,newlogprob,entropy,newvalue = agent.get_action_and_value(batch_obs[batch_i],batch_actions.long()[batch_i])
    logratio = newlogprob - batch_logprobs[batch_i]
    ratio = logratio.exp()

    with torch.no_grad():
        old_kl_approx = (-logratio).mean()
        kl_approx = (ratio-1-logratio).mean()

    mbatch_advantages = batch_advantages[batch_i]
    mbatch_advantages = (mbatch_advantages - mbatch_advantages.mean()) / (mbatch_advantages.std() + 1e-8)

    pg_loss1 = -ratio*mbatch_advantages
    pg_loss2 = -mbatch_advantages * torch.clamp(ratio, 1-clip_coef, 1+clip_coef)
    pg_loss = torch.max(pg_loss1,pg_loss2)

    newvalue = newvalue.view(-1)
    v_loss = 0.5*((newvalue - batch_returns[batch_i])**2).mean()
    entropy_loss = entropy.mean()
    loss = pg_loss - ent_coef*entropy_loss + v_coef*v_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def PPO(num_envs,num_steps,total_timesteps,batch_size,update_epochs,learning_rate,gamma,gae_lambda,lastgaelam):
    seed = 1
    env_id = 'custom_env/ExpWorld-v1'
    run_name = f"{env_id}__{seed}__{int(time.time())}"
    writer = writer = SummaryWriter(f"runs/{run_name}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id, seed + i) for i in range(num_envs)]
    )
    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)

    obs_buffer = torch.zeros((num_steps, num_envs) + envs.single_observation_space.shape).to(device)
    actions_buffer = torch.zeros((num_steps, num_envs) + envs.single_action_space.shape).to(device)
    logprobs_buffer = torch.zeros((num_steps, num_envs)).to(device)
    rewards_buffer = torch.zeros((num_steps, num_envs)).to(device)
    dones_buffer = torch.zeros((num_steps, num_envs)).to(device)
    values_buffer = torch.zeros((num_steps, num_envs)).to(device)
    advantages_buffer = torch.zeros_like(rewards_buffer).to(device)

    global_step = 0
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(num_envs).to(device)
    num_updates = total_timesteps // batch_size
    
    for update in range(1,num_updates+1):
        for step in range(num_steps):
            global_step += 1*num_envs 
            rollout(envs,obs_buffer,values_buffer,actions_buffer,rewards_buffer,logprobs_buffer,dones_buffer,agent,next_obs,next_done,step,num_steps,num_envs)
            for item in info:
                if "episode" in item.keys():
                    print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
                    break
            
            next_value = agent.get_value(next_obs).reshape(1.-1)
            bootstrap(agent,rewards_buffer,dones_buffer,values_buffer,advantages_buffer,next_obs,next_done,next_value,num_steps,gamma,gae_lambda,lastgaelam)
            train(envs,agent,obs_buffer,actions_buffer,logprobs_buffer,advantages_buffer,)

    


    