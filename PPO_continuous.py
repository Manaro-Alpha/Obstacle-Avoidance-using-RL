import os
import gym
import random
import time
from distutils.util import strtobool
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from custom_env.envs import ObsAv_Env
start_time = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActorCritic(nn.Module):
    def __init__(self,obs_dim,action_dim,action_std):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.actor = nn.Sequential(
            nn.Linear(np.prod(self.obs_dim),64),
            nn.Tanh(),
            nn.Linear(64,64),
            nn.Tanh(),
            nn.Linear(64,np.prod(self.action_dim))
        )

        self.critic = nn.Sequential(
            nn.Linear(np.prod(self.obs_dim),64),
            nn.Tanh(),
            nn.Linear(64,64),
            nn.Tanh(),
            nn.Linear(64,1)
        )

        self.actor_std = torch.full((self.action_dim),action_std) ## dimension might be incorrect

    def get_value(self,obs):
        return self.critic(obs)
    
    def get_actionAndvalue(self,obs,action = None):
        action_mean = self.actor(obs).to(device)
        action_std = self.actor_std.to(device)
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
    values_buffer[step] = value.flatten()
    logprobs_buffer[step] = logprob

    next_obs,reward,next_done,truncated,infos = envs.step(action.cpu().numpy())
    rewards_buffer[step] = torch.Tensor(reward).to(device)
    obs = torch.Tensor(next_obs).to(device)
    done = torch.Tensor(next_done).to(device)

    for i in range(len(infos['final_info'])):
        if infos["final_info"][i] == 0:
            continue
        print(f"global_step = {global_step}, episodic_return = {infos['episodic_return'][i]}")
        writer.add_scalar("charts/episodic_return", infos['episodic_return'][i], global_step)
        writer.add_scalar("charts/episodic_length", infos["ep_l"][i], global_step)

def train(envs,agent:ActorCritic,optim,obs_buffer,values_buffer,action_buffer,rewards_buffer,dones_buffer,logprobs_buffer,advantage_buffer,reward_togo,next_obs,next_done,gamma,num_steps,global_step,batch_size,num_epochs,mbatch_size,clip_coef,writer):
    global start_time
    with torch.no_grad():
        for step in reversed(range(num_steps)):
            if step == num_steps-1:
                next_nonterminal = 1 - torch.Tensor(next_done).to(device) ## get next_done and next_nonterminal values by running envs.step one last time
                next_value = agent.get_value(next_obs).reshape(1,-1)
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
    b_reward_togo = reward_togo.reshape(-1)
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
        
    y_pred, y_true = b_values.cpu().numpy(), b_reward_togo.cpu().numpy()
    var_y = np.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

    writer.add_scalar("charts/learning_rate", optim.param_groups[0]["lr"], global_step)
    writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
    writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
    writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
    writer.add_scalar("losses/explained_variance", explained_var, global_step)
    print("SPS:", int(global_step / (time.time() - start_time)))
    # if num_steps%500 == 0:
    print("####################","\n")
    print(f"loss: {loss}","\n",f"pg_loss: {pg_loss}",'\n',f"value_loss: {v_loss}",'\n',"####################")
    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

def save(agent:ActorCritic,path):
    torch.save(agent.state_dict(),path)

def make_env(env_id,idx,run_name,gamma):
    def thunk():
        env = gym.make(env_id,render_mode = "rgb_array")
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        # if idx == 1:
        #     env = gym.wrappers.RecordVideo(env,f"videos/{run_name}")
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env
    return thunk

def PPO(env_id,num_envs,num_steps,learning_rate,gamma,total_timesteps,batch_size,num_epochs,mbatch_size,clip_coef,save_freq):
    global start_time
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

    start_time = time.time()
    global_step = 0
    obs,_ = envs.reset()
    obs = torch.Tensor(obs).to(device)
    done = torch.zeros(num_envs).to(device)
    num_updates = int(total_timesteps/batch_size)

    for update in range(num_updates):
        for step in range(num_steps):
            global_step += num_envs
            rollout(envs,obs_buffer,values_buffer,action_buffer,rewards_buffer,dones_buffer,logprobs_buffer,obs,done,step,global_step,agent,writer,device)
        
        next_obs = obs_buffer[num_steps-1]
        _,_,next_done,_,_ = envs.step(action_buffer[num_steps-1].cpu().numpy())
        advantages = torch.zeros_like(rewards_buffer).to(device).detach()
        reward_togo = torch.zeros_like(rewards_buffer).to(device).detach()
        train(envs,agent,optimizer,obs_buffer,values_buffer,action_buffer,rewards_buffer,dones_buffer,logprobs_buffer,advantages,reward_togo,
              next_obs,next_done,gamma,num_steps,global_step,batch_size,num_epochs,mbatch_size,clip_coef,writer)
        
        if update % save_freq == 0:
            save(agent,path="runs/policy")

if __name__ == "__main__":
    env_id = "custom_env/ExpWorld-v1"
    PPO(env_id = env_id,
        num_envs = 1,
        num_steps = 2000,
        learning_rate=3e-4,
        gamma=0.99,
        total_timesteps=1000000,
        batch_size = 2000,
        num_epochs=10,
        mbatch_size=250,
        clip_coef=0.2,
        save_freq=50)


        
    


    





