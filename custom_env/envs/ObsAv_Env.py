import gym
from gym import spaces
import pygame
import numpy as np
import random

reward = 0
class ExpWorld1(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    def __init__(self,render_mode = None, size = 50, fps = 4):
        self.size = size
        self.window_size = 512
        #self.start = random()
        self.obstacle_num = 2
        self.dt = 0.1
        self.path = []
        self.observation_space = spaces.Dict(
            {
                "agent_pos": spaces.Box(0,size-1,shape=(2,)),
                "agent_vel": spaces.Box(-np.inf,np.inf,shape=(2,)),
                "target_pos": spaces.Box(0, size - 1, shape=(2,)),
                "target_vel": spaces.Box(-np.inf, np.inf, shape=(2,)),
                "obstacle": spaces.Box(0, size - 1, shape=(self.obstacle_num,2)),
                "obstacle_radius": spaces.Box(0,1,shape=(self.obstacle_num,))
            }
        )
        self.action_space = spaces.Discrete(4)
    
        self._action_set = {
            0:np.array([1,-1]),
            1:np.array([-1,1]),
            2:np.array([1,1]),
            3:np.array([-1,-1]),
        }
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        
        self.window = None
        self.clock = None

    def _get_obs(self):
        return {"agent_pos": self._agent_location,
                "agent_vel": self._agent_velocity, 
                "target_pos": self._target_location, 
                "target_vel": self._target_velocity, 
                "obstacle": self._obstacle_location,
                "obstacle_radius": self._obstacle_radius}
    def _get_info(self):
        return {
            "distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)
        }
    
    def reset(self, seed=None,options=None):
        super().reset(seed=seed)
        # self._agent_location = self.np_random.integers(0,self.size/2,size=2,dtype=float)
        # self._agent_velocity = self.np_random.integers(-1,1,size=2,dtype=float)
        # self._target_location = self._agent_location
        # while np.array_equal(self._target_location, self._agent_location):
        #     self._target_location = self.np_random.integers(0, self.size, size=2, dtype=float)
        # self._target_velocity = self.np_random.integers(-1,1,size=2,dtype=float)
        # self._obstacle_location = self._agent_location
        # for obs in range(self.obstacle_num):
        #     while np.array_equal(self._obstacle_location[obs], self._agent_location) and np.array_equal(self._obstacle_location[obs], self._target_location):
        #         self._obstacle_location[obs] = self.np_random.floats(0, self.size, size=2, dtype=float)
        # self._obstacle_radius = self.np_random.integers(0,1,size=1)
        # observation = self._get_obs()
        # info = self._get_info()
        # if self.render_mode == "human":
        #     self._render_frame()
        # return observation, info
        self.path = []
        self._agent_location = self.np_random.uniform(-self.size,self.size,2)
        self.path.append(self._agent_location+50)
        self._agent_velocity = self.np_random.uniform(-1,1,2)
        self._target_location = self._agent_location
        self._obstacle_location = []
        self._obstacle_radius = []
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.uniform(-self.size,self.size,size=2)
        self._target_velocity = self.np_random.uniform(-1,1,2)
        for n in range(self.obstacle_num):
            obs = self._agent_location
            while np.array_equal(self._agent_location,obs) or np.array_equal(self._target_location,obs):
                obs = self.np_random.uniform(-self.size,self.size,size=2)
            self._obstacle_location.append(obs)
        for n in range(self.obstacle_num):
            radius = self.np_random.uniform(0,10,1)
            self._obstacle_radius.append(radius)
        observation = self._get_obs()
        info = self._get_info()
        return observation,info

    def step(self,action):
        # terminated = False
        # acc = self._action_set[action]
        # dv = acc*self.dt
        # self._agent_velocity = self._agent_velocity + dv
        # dx = self._agent_velocity*self.dt
        # self._agent_location = self._agent_location + dx
        
        # if np.linalg.norm(self._agent_location - self._target_location,ord=1) < 0.1:
        #     done = True
        # reward = 0
        # ## conditions for termination:
        # # going out of bounds
        # # colliding with an obstacle
        # for obs in range(len(self.obstacle_num)):
        #     if np.linalg.norm(self._agent_location - self._obstacle_location[obs],ord=2) <= self._obstacle_radius[obs]:
        #         terminated = True
        # if self._agent_location.any() < 0 or self._agent_location.any() > self.size-1:
        #     terminated = True 
        # if terminated:
        #     reward -= 1
        # if done:
        #     reward += 10
        # reward -= 0.1

        # observation = self._get_obs()
        # info = self._get_info()
        # if self.render_mode == "human":
        #     self._render_frame()
        
        # return observation, reward, done, terminated, info
        acc = self._action_set[action]
        dv = acc*self.dt
        self._agent_velocity = self._agent_velocity + dv
        dx = self._agent_velocity*self.dt
        self._agent_location = self._agent_location + dx
        self.path.append(self._agent_location+50)
        global reward
        done = False
        #terminated = False
        if np.linalg.norm(self._agent_location - self._target_location,ord=1) < 0.1 and np.linalg.norm(self._agent_velocity - self._target_velocity,ord=1) < 0.1:
            done = True
            reward += 10
        for obs in range(self.obstacle_num):
            if np.linalg.norm(self._agent_location - self._obstacle_location[obs],ord=2) <= self._obstacle_radius[obs]:
                done = True
                reward -= 1
        if any(pos < -self.size for pos in self._agent_location) or any(pos > self.size for pos in self._agent_location):
            done = True
            reward -= 1
        
        reward = reward - 0.01*np.linalg.norm(self._agent_location - self._target_location,ord=1) - 0.1
        
        #print('acceleration =',acc,'\n',' velocity = ',self._agent_velocity,'\n',' position = ',self._agent_location,'\n','terminated = ',terminated,'\n','done = ',done,'\n')
        observation = self._get_obs()
        info = self._get_info()
        if self.render_mode == "human":
            self._render_frame()
        return observation, reward, done,False, info
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        
    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            self.clock = pygame.time.Clock()
        
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((0, 0, 0))
        pygame.draw.circle(
            canvas,[255,0,0],[self._agent_location[0]+50,self._agent_location[1]+50],2
        )# agent

        pygame.draw.circle(
            canvas,[0,255,0],[self._target_location[0]+50,self._target_location[1]+50],5
        )
        for n in range(self.obstacle_num):
            pygame.draw.circle(
                canvas,[255,255,255],[self._obstacle_location[n][0]+50,self._obstacle_location[n][1]+50],self._obstacle_radius[n][0]
            )
        pygame.draw.lines(
            canvas,[4,217,255],False,self.path,1
        )
        if self.render_mode == "human":
            self.window.blit(canvas, (0, 0))
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(4)
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
    
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit() 
