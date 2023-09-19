import time
import gym
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from custom_env.envs import ObsAv_Env

class ActorCritic(nn.Module):
    def __init__(self,obs_dim,action_dim,device):
        super(ActorCritic,self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device

        self.actor = nn.Sequential(
            nn.Linear(obs_dim,64),
            nn.Tanh(),
            nn.Linear(64,64),
            nn.Tanh(),
            nn.Linear(64,action_dim),
            nn.Tanh()
        )

        self.critic = nn.Sequential(
            nn.Linear(obs_dim,64),
            nn.Tanh(),
            nn.Linear(64,64),
            nn.Tanh(),
            nn.Linear(64,1)
        )

        self.action_var = torch.full((self.action_dim,),0.5).to(self.device)

    def set_action_var(self,action_var):
        self.action_var = torch.full((self.action_dim,),action_var**2).to(self.device)

    def select_action(self,obs):
        action_mean = self.actor(obs)
        cov_matrix = torch.diag(self.action_var).unsqueeze(0)
        dist = MultivariateNormal(action_mean,cov_matrix)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_value = self.critic(obs)

        return action.detach().cpu().numpy().flatten(),action_logprob.detach()
    
    def evaluate(self,batch_obs,batch_acts):
        mean = self.actor(batch_obs)
        var = self.action_var.expand_as(mean)
        cov_mat = torch.diag_embed(var).to(self.device)
        dist = MultivariateNormal(mean,cov_mat)
        logprobs = dist.log_prob(batch_acts)
        V = self.critic(batch_obs).squeeze()
        return V,logprobs,dist.entropy()
    
class Rollout_Buffer():
    def __init__(self):
        self.obs = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.lengths = []
    
    def clear(self):
        del self.obs[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.values[:]
        del self.dones[:]
        del self.lengths[:]

class PPO():
    def __init__(self,envs):
        self.device = torch.device('cpu')
        if(torch.cuda.is_available()):
            self.device = torch.device("cuda:0")
            torch.cuda.empty_cache()
        print("Device set to " + str(torch.cuda.get_device_name(self.device)))
        self.env = envs
        self.run_name = f"ppo_log_{int(time.time())}"
        
        self.writer = SummaryWriter(f"runs/{self.run_name}")
        # assert(type(env.observation_space) == gym.spaces.box.Box)
        # assert(type(env.action_space) == gym.spaces.box.Box)
        self.obs_dim = np.prod(envs.observation_space.shape)
        self.action_dim = np.prod(envs.action_space.shape)
        self.agent = ActorCritic(self.obs_dim,self.action_dim,self.device).to(self.device)
        
        self._init_hyperparams()
        self.rollout_buffer = Rollout_Buffer()
        self.actor_optim = torch.optim.Adam(self.agent.actor.parameters(),lr=self.lr)
        self.critic_optim = torch.optim.Adam(self.agent.critic.parameters(),lr=self.lr)
        self.logger = {
            "timesteps": 0,
            "iterations": 0,
            "batch_length": [],
            "batch_rewards": [],
            "actor_loss": [],
            "value_loss": [],
            "lr": 0
        }

    def rollout(self):
        t = 0
        ep_rews = []
        ep_vals = []
        ep_dones = []
        while t < self.timesteps_per_batch:
            ep_rews = []
            ep_vals = []
            ep_dones = []
            obs,_ = self.env.reset()
            done = False
            for step in range(self.max_timesteps_per_episode):
                ep_dones.append(done)
                t+=1
                self.rollout_buffer.obs.append(obs)
                action,logprob = self.agent.select_action(torch.tensor(obs,dtype=torch.float,device=self.device))
                value = self.agent.critic(torch.tensor(obs,dtype=torch.float,device=self.device))
                obs,reward,done,trunc,info = self.env.step(action)
                ep_rews.append(reward)
                ep_vals.append(value.flatten())
                
                self.rollout_buffer.actions.append(action)
                self.rollout_buffer.logprobs.append(logprob)
                if done:
                    break
            
            self.rollout_buffer.lengths.append(step+1)
            self.rollout_buffer.rewards.append(ep_rews)
            self.rollout_buffer.values.append(ep_vals)
            self.rollout_buffer.dones.append(ep_dones)

        batch_obs = torch.tensor(self.rollout_buffer.obs,dtype=torch.float).to(self.device)
        batch_actions = torch.tensor(self.rollout_buffer.actions,dtype=torch.float).to(self.device)
        batch_logprobs = torch.tensor(self.rollout_buffer.logprobs,dtype=torch.float).to(self.device)
        batch_dones = self.rollout_buffer.dones
        batch_rewards = self.rollout_buffer.rewards
        batch_values = self.rollout_buffer.values
        batch_lengths = self.rollout_buffer.lengths
        batch_advantages = []
        for epi_rews,epi_values,epi_dones in zip(batch_rewards,batch_values,batch_dones):
            adv = []
            last_adv = 0
            for t in reversed(range(len(epi_rews))):
                if t+1 < len(epi_rews):
                    delta = torch.tensor(epi_rews[t],dtype=torch.float,device=self.device) + self.gamma*epi_values[t+1]*(1-torch.tensor(epi_dones[t+1],dtype=torch.int,device=self.device)) - epi_values[t]
                else:
                    delta = torch.tensor(epi_rews[t],dtype=torch.float,device=self.device) - epi_values[t]
                advantage = delta + self.gamma*self.lam*(1-torch.tensor(epi_dones[t],dtype=torch.int,device=self.device))*last_adv
                last_adv = advantage
                adv.insert(0,advantage)
            batch_advantages.extend(adv)
        
        batch_advantages = torch.tensor(batch_advantages,dtype=torch.float).to(self.device)
        self.logger["batch_rewards"] = batch_rewards
        self.logger["batch_length"] = batch_lengths
        return batch_obs,batch_actions,batch_logprobs,batch_rewards,batch_lengths,batch_values,batch_dones,batch_advantages


    def learn(self,total_timesteps):
        t = 0
        i = 0
        while t < total_timesteps:
            self.logger["timesteps"] = t
            self.logger["iterations"] = i
            self.rollout_buffer.clear()
            b_obs,b_actions,b_logprobs,b_rewards,b_lengths,b_values,b_dones,b_advantages = self.rollout()
            t += np.sum(b_lengths)
            i += 1
            if i%self.video_freq == 0:
                self.env  = gym.wrappers.RecordVideo(self.env,f"videos/{self.run_name}")

            V = self.agent.critic(b_obs).squeeze()
            A = b_advantages
            A = (A - A.mean())/(A.std()+1e-10)
            b_reward_togo = A + V.detach()
            
            size = b_obs.size(0)
            inds = np.arange(size)
            minibatch_size = size//self.num_minibatch
            a_loss = []
            v_loss = []

            for _ in range(self.num_epochs):
                frac = (t - 1.0)/total_timesteps
                new_lr = self.lr*(1.0-frac)
                new_lr = max(new_lr,0.0)
                self.actor_optim.param_groups[0]["lr"] = new_lr
                self.critic_optim.param_groups[0]["lr"] = new_lr
                
                self.logger["lr"] = new_lr
                
                np.random.shuffle(inds)
                for start in range(0,size,minibatch_size):
                    end = start + minibatch_size
                    idx = inds[start:end]

                    mini_obs = b_obs[idx]
                    mini_actions = b_actions[idx]
                    mini_logprobs = b_logprobs[idx]
                    mini_A = A[idx]
                    mini_reward_togo = b_reward_togo[idx]
                    V,new_logprob,entropy = self.agent.evaluate(mini_obs,mini_actions)
                    logratio = new_logprob - mini_logprobs
                    ratio = torch.exp(logratio)
                    entropy_loss = entropy.mean()
                    approx_kl = ((ratio-1) - logratio).mean()
                    surr1 = ratio*mini_A
                    surr2 = mini_A*torch.clamp(ratio,1-self.clip,1+self.clip)
                    actor_loss = (-torch.min(surr1,surr2)).mean()
                    actor_loss = actor_loss - self.ent_coef*entropy_loss
                    
                    self.actor_optim.zero_grad()
                    actor_loss.backward(retain_graph=True)
                    nn.utils.clip_grad_norm_(self.agent.actor.parameters(),self.max_grad_norm)
                    self.actor_optim.step()
                    
                    critic_loss = nn.MSELoss()(V,mini_reward_togo)
                    self.critic_optim.zero_grad()
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.critic.parameters(),self.max_grad_norm)
                    self.critic_optim.step()

                    a_loss.append(actor_loss.detach())
                    v_loss.append(critic_loss.detach())
                    if approx_kl>self.target_kl:
                        break
            avg_a_loss = sum(a_loss)/len(a_loss)
            avg_v_loss = sum(v_loss)/len(v_loss)
            
            self.logger["actor_loss"].append(avg_a_loss.cpu())
            self.logger["value_loss"].append(avg_v_loss.cpu())
            self.log_summary()
            if i % 10000 == 0:
                self.save('./ppo_LunarLanderContinuous-v2')
                
    def _init_hyperparams(self):
        self.timesteps_per_batch = 1024
        self.max_timesteps_per_episode = 1000
        self.num_epochs = 4
        self.clip = 0.2
        self.lr = 3e-4
        self.gamma = 0.999
        self.lam = 0.98
        self.num_minibatch = 6
        self.ent_coef = 0.01
        self.target_kl = 0.02
        self.max_grad_norm = 0.5
        self.env_id = "Pendulum-v1"
        self.video_freq = 10000
    
    def save(self,path):
        torch.save(self.agent.state_dict(),path)
    
    def load(self,path):
        self.agent.load_state_dict(torch.load(path))

    def log_summary(self):
        timesteps = self.logger['timesteps']
        self.writer.add_scalar("timesteps_so_far",timesteps)
        iterations = self.logger["iterations"]
        self.writer.add_scalar("iterations_so_far",iterations)
        lr = self.logger["lr"]
        self.writer.add_scalar("learning_rate",lr,iterations)
        ep_len = np.mean(self.logger["batch_length"])
        ep_rew = np.mean([np.sum(ep_rews) for ep_rews in self.logger["batch_rewards"]])
        self.writer.add_scalar("episode_reward",ep_rew,iterations)
        self.writer.add_scalar("episode_length",ep_len,iterations)
        avg_actor_loss = np.mean([losses.float().mean() for losses in self.logger['actor_loss']])
        avg_value_loss = np.mean([losses.float().mean() for losses in self.logger['value_loss']])
        self.writer.add_scalar("actor_loss",avg_actor_loss,iterations)
        self.writer.add_scalar("critic_loss",avg_value_loss,iterations)

        ep_len = str(round(ep_len, 2))
        ep_rew = str(round(ep_rew, 2))
        avg_actor_loss = str(round(avg_actor_loss, 5))
        avg_value_loss = str(round(avg_value_loss, 5))
        print(flush=True)
        print(f"-------------------- Iteration #{iterations} --------------------", flush=True)
        print(f"Average Episodic Length: {ep_len}", flush=True)
        print(f"Average Episodic Return: {ep_rew}", flush=True)
        print(f"Average Loss: {avg_actor_loss}", flush=True)
        print(f"Average value Loss: {avg_value_loss}", flush=True)
        print(f"Timesteps So Far: {timesteps}", flush=True)
        print(f"Learning rate: {lr}", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)
        self.logger['batch_lens'] = []
        self.logger['batch_rews'] = []
        self.logger['actor_losses'] = []


# def make_env(env_id,idx,run_name,gamma):
#     def thunk():
#         env = gym.make(env_id,render_mode = "rgb_array")
#         env = gym.wrappers.FlattenObservation(env)
#         env = gym.wrappers.RecordEpisodeStatistics(env)
#         # if idx == 1:
#         #     env = gym.wrappers.RecordVideo(env,f"videos/{run_name}")
#         env = gym.wrappers.ClipAction(env)
#         env = gym.wrappers.NormalizeObservation(env)
#         env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
#         env = gym.wrappers.NormalizeReward(env, gamma=gamma)
#         env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
#         return env
#     return thunk

# gamma = 0.999
# env_id = "custom_env/ExpWorld-v1"
# run_name = f"{env_id}__{int(time.time())}"
# envs = gym.vector.SyncVectorEnv([make_env(env_id,i,run_name,gamma) for i in range(1)])
# model = PPO(envs)
# model.learn(1000000)