import gym
from gym import spaces
import pygame
import numpy as np
import random
import utils
import math
import time

scale = 0
MAX_TIME = 100
reward = 0
Eptime = 0
class Env(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}
    def __init__(self,render_mode = None, size = 50):
        self.size = size
        self.window_size = 768
        self.dt = 0.1
        self.path = []
        polar_grid = utils.polar_occupancy_grid(0,0,0.1,1)
        self.observation_space = spaces.Dict(
            {
                "agent_pos": spaces.Box(-np.inf,np.inf,shape=(2,)),
                "agent_angle": spaces.Box(-3.14,3.14,(1,)),
                "agent_vel": spaces.Box(-np.inf,np.inf,shape=(1,)),
                "target_pos": spaces.Box(-np.inf,np.inf, shape=(2,)),
                "target_angle": spaces.Box(-3.14159,3.14159,(1,)),
                "target_vel": spaces.Box(-np.inf, np.inf, shape=(2,)),
                "occupancy_grid": spaces.Box(0,1,shape=polar_grid.shape(),dtype=bool)
            }
        )
        self.action_space = spaces.Box(np.array([-1,-0.52356]).astype(np.float32),np.array([1,0.52356]).astype(np.float32))
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        
        self.window = None
        self.clock = None

    def _get_obs(self):
        return {"agent_pos": self._agent_location,
                "agent_angle": self._agent_angle,
                "agent_vel": self._agent_velocity, 
                "target_pos": self._target_location,
                "target_angle": self._target_angle, 
                "target_vel": self._target_velocity, 
                "occupancy_grid": self._occupancy_grid
                }
    def _get_info(self):
        return {
            "distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)
        }
    
    def reset(self, seed=None,options=None):
        super().reset(seed=seed)

        global Eptime
        Eptime = time.time()
        global reward
        reward = 0
        self.path = []
        self._agent_location = self.np_random.uniform(-self.size,self.size,2)
        self.path.append(self._agent_location+384)
        self._agent_velocity = self.np_random.uniform(-3,3,1)
        self._agent_angle = self.np_random.uniform(-3.14,3.14,1)
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.uniform(-self.size,self.size,size=2)
        self._target_velocity = self.np_random.uniform(-3,3,1)
        self._target_angle = self.np_random.uniform(-3.14,3.14,1)
        self.polar_occupancy_grid = utils.polar_occupancy_grid(self._agent_location[0],self._agent_location[1],0.1,1)
        self._occupancy_grid = np.zeros(np.prod(self.polar_occupancy_grid.shape()),dtype=bool)
        self.obstacle_num = 0
        self.obstacle_list = []
        for i in range(self.obstacle_num):
            n = random.randint(0,1)
            if n == 0:
                centre = self._agent_location
                radius = self.np_random.uniform(0,5,1)
                while np.linalg.norm(centre-self._agent_location,2) < radius or np.linalg.norm(centre-self._target_location,2) < radius:
                    centre = self.np_random.uniform(-self.size,self.size,2)
                obs = utils.circular_obstacle(centre[0],centre[1],radius)
                self.obstacle_list.append(obs)
            else:
                centre = self._agent_location
                #p1 = np.random.uniform(centre-2.5,centre+2.5,2)
                p1 = self.np_random.uniform(centre-2.5,centre+2.5,2)
                p2 = self.np_random.uniform(centre-2.5,centre+2.5,2)
                p3 = self.np_random.uniform(centre-2.5,centre+2.5,2)
                p4 = self.np_random.uniform(centre-2.5,centre+2.5,2)
                obs = utils.rectangular_obstacle(p1,p2,p3,p4)

                while obs.collision(self._agent_location) or obs.collision(self._target_location):
                    centre = np.random.uniform(-self.size,self.size,2)
                    p1 = self.np_random.uniform(centre-2.5,centre+2.5,2)
                    p2 = self.np_random.uniform(centre-2.5,centre+2.5,2)
                    p3 = self.np_random.uniform(centre-2.5,centre+2.5,2)
                    p4 = self.np_random.uniform(centre-2.5,centre+2.5,2)

                    obs = utils.rectangular_obstacle(p1,p2,p3,p4)
                self.obstacle_list.append(obs)

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
        # acc = self._action_set[action]
        # dv = acc*self.dt
        # self._agent_velocity = self._agent_velocity + dv
        # dx = self._agent_velocity*self.dt
        # self._agent_location = self._agent_location + dx
        # self.path.append(self._agent_location+50)
        # global reward
        # done = False
        # #terminated = False
        # if np.linalg.norm(self._agent_location - self._target_location,ord=1) < 0.1 and np.linalg.norm(self._agent_velocity - self._target_velocity,ord=1) < 0.1:
        #     done = True
        #     reward += 10
        # for obs in range(self.obstacle_num):
        #     if np.linalg.norm(self._agent_location - self._obstacle_location[obs],ord=2) <= self._obstacle_radius[obs]:
        #         done = True
        #         reward -= 1
        # if any(pos < -self.size for pos in self._agent_location) or any(pos > self.size for pos in self._agent_location):
        #     done = True
        #     reward -= 1
        
        # reward = reward - 0.01*np.linalg.norm(self._agent_location - self._target_location,ord=1) - 0.1
        
        # #print('acceleration =',acc,'\n',' velocity = ',self._agent_velocity,'\n',' position = ',self._agent_location,'\n','terminated = ',terminated,'\n','done = ',done,'\n')
        global MAX_TIME
        global scale
        global Eptime
        global reward
        done = False
        trajectory = utils.trajectory(self._agent_location,self._agent_angle,self._agent_velocity,action[0],action[1],self.dt)
        self._agent_location,self._agent_angle,self._agent_velocity = trajectory.traj_generate()
        self.path.append(self._agent_location+384)
        self._target_location = self._target_location
        self._target_angle = self._target_angle

        for i in range(self.polar_occupancy_grid.points.shape[0]):
            for k in range(self.polar_occupancy_grid.points.shape[1]):
                point_coord = self.polar_occupancy_grid.global_pos(self.polar_occupancy_grid.points[i][k],self._agent_angle)
                for obs in self.obstacle_list:
                    if obs.shape == 'circle':
                        if np.linalg.norm(point_coord-[obs.x,obs.y],2) < obs.radius:
                            self.polar_occupancy_grid.points[i][k].occupied = True
                    if obs.shape == 'rectangle':
                        if obs.collision(point_coord):
                            self.polar_occupancy_grid.points[i][k].occupied = True

        for i in range(self.polar_occupancy_grid.points.shape[0]):
            for k in range(self.polar_occupancy_grid.points.shape[1]):
                self._occupancy_grid[i+k] = self.polar_occupancy_grid.points[i][k].occupied


        ## episode ends

        if np.linalg.norm(self._agent_location - self._target_location,2) < 0.2:
            reward += 20 - scale*1e-4*np.linalg.norm(np.array([self._agent_velocity*math.cos(self._agent_angle),self._agent_velocity*math.sin(self._agent_angle)]) - np.array([self._target_velocity*math.cos(self._agent_angle),self._target_velocity*math.sin(self._agent_angle)]))
            done = True
            scale += 1

        for obs in self.obstacle_list:
            if obs.shape == 'circle':
                if np.linalg.norm(self._agent_location-[obs.x,obs.y],ord=2) < obs.radius:
                    reward -= 0.1
                    done = True
                    print('collision')
            if obs.shape == 'rectangle':
                if obs.collision(self._agent_location):
                    reward -= 0.1
                    done = True
                    print('collision')

        if time.time() - Eptime > MAX_TIME:
            reward -= 5
            done = True
        
        reward = reward - 0.01 
        self._target_location = self._target_location
        self._target_velocity = self._target_velocity
        observation = self._get_obs()
        info = self._get_info()
        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, done,False, info
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        
    # def _render_frame(self):
    #     if self.window is None and self.render_mode == "human":
    #         pygame.init()
    #         pygame.display.init()
    #         self.window = pygame.display.set_mode((self.window_size, self.window_size))
    #         self.clock = pygame.time.Clock()
        
    #     canvas = pygame.Surface((self.window_size, self.window_size))
    #     canvas.fill((0, 0, 0))
    #     pygame.draw.circle(
    #         canvas,[255,0,0],[self._agent_location[0]+50,self._agent_location[1]+50],2
    #     )# agent

    #     pygame.draw.circle(
    #         canvas,[0,255,0],[self._target_location[0]+50,self._target_location[1]+50],5
    #     )
    #     for n in range(self.obstacle_num):
    #         pygame.draw.circle(
    #             canvas,[255,255,255],[self._obstacle_location[n][0]+50,self._obstacle_location[n][1]+50],self._obstacle_radius[n][0]
    #         )
    #     pygame.draw.lines(
    #         canvas,[4,217,255],False,self.path,1
    #     )
    #     if self.render_mode == "human":
    #         self.window.blit(canvas, (0, 0))
    #         pygame.event.pump()
    #         pygame.display.update()
    #         self.clock.tick(4)
    #     else:  # rgb_array
    #         return np.transpose(
    #             np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
    #         )

    def _render_frame(self):
            if self.window is None and self.render_mode == "human":
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode((self.window_size, self.window_size))
                self.clock = pygame.time.Clock()
            
            canvas = pygame.Surface((self.window_size, self.window_size))
            canvas.fill((0, 0, 0))

            pygame.draw.circle(
                canvas,[255,255,0],[float(self._agent_location[0])+384,float(self._agent_location[1])+384],2
            )## agent
            #Theta = math.atan2(self._agent_velocity[1],self._agent_velocity[0])
            for i in range(self.polar_occupancy_grid.points.shape[0]):
                for k in range(self.polar_occupancy_grid.points.shape[1]):
                    point_coord = self.polar_occupancy_grid.global_pos(self.polar_occupancy_grid.points[i][k],self._agent_angle)
                    if self.polar_occupancy_grid.points[i][k].occupied:
                        pygame.draw.circle(
                            canvas,[255,20,0],point_coord+384,0.5
                        )
                    else:
                        pygame.draw.circle(
                            canvas,[20,255,0],point_coord+384,0.5
                        )


            pygame.draw.circle(
                canvas,[0,255,0],[self._target_location[0]+384,self._target_location[1]+384],3
            )## target

            for n in range(len(self.obstacle_list)):
                obs = self.obstacle_list[n]
                if obs.shape == 'circle':
                    pygame.draw.circle(
                        canvas,[255,255,255],[obs.x+384,obs.y+384],float(obs.radius)
                    )
                if obs.shape == 'rectangle':
                    pygame.draw.polygon(
                        canvas,[255,255,255],points=[obs.p1+384,obs.p2+384,obs.p3+384,obs.p4+384]
                    )
            
            pygame.draw.lines(
                canvas,[4,217,255],False,self.path
            )

            if self.render_mode == "human":
                self.window.blit(canvas, (0, 0))
                pygame.event.pump()
                pygame.display.update()
                self.clock.tick(4)
            else:
                return np.transpose(
                    np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
                )    
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit() 


# if __name__ == '__main__':
#     env = Env(render_mode="human")
#     obs,_ = env.reset()
#     #print(env.action_dim()) 
#     done = False
#     while not done:
#         action = np.array([1,0])
#         observation,reward,done,_,_ = env.step(action)
#         env.render()
#         print(done,'\n','pos =',env._agent_location,'\n','vel = ',env._agent_velocity,math.degrees(env._agent_angle),'\n','Target = ',env._target_location,'\n','Tar_Vel = ',env._target_velocity)
#     env.close()