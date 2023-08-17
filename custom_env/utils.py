import pygame
import numpy as np
import random
import math

class circular_obstacle():
    def __init__(self,x,y,radius,shape="circle"):
        self.x = x
        self.y = y
        self.radius = radius 
        self.shape = shape

class rectangular_obstacle():
    def __init__(self,p1:np.ndarray,p2:np.ndarray,p3:np.ndarray,p4:np.ndarray,shape="rectangle"):
        self.shape = shape
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4
    
    def collision(self,pt:np.ndarray):
        points = np.array([self.p1,self.p2,self.p3,self.p4])
        p1 = points[0]
        inside = False
        for i in range(1,len(points)+1):
            p2 = points[i%len(points)]
            if pt[1] > min(p1[1],p2[1]):
                if pt[1] <= max(p1[1],p2[1]):
                    if pt[0] <= max(p1[0],p2[0]):
                        x_intersection = (pt[1] - p1[1])*(p2[0]-p1[0])/(p2[1]-p1[1]) + p1[0]
                        if p1[0] == p2[0] or pt[0] <= x_intersection:
                            inside = not inside
            p1 = p2
        return inside

class point():
    def __init__(self,r,theta):
        self.r = r
        self.theta = theta # in degrees
        self.occupied = False

class polar_occupancy_grid():
    def __init__(self,x,y,R,Theta):
        self.x = x #x position of agent
        self.y = y #y position of agent
        self.R = R #max R upto which agent can see
        self.Theta = Theta #in degrees # FOV of agent
        self.dr = 0.01
        self.dtheta = 0.1
        self.points = np.empty((int(self.R/self.dr),int(self.Theta/self.dtheta)),dtype=point)
        for r in range(int(self.R/self.dr)):
            for theta in range(int(self.Theta/self.dtheta)):
                self.points[r][theta] = point((r+1)*self.dr,((theta+1)*self.dtheta)-(self.Theta/2))

    def shape(self):
        return np.shape(self.points)
    def occupancy_grid(self):
        return self.points
    def global_pos(self,pt:point,theta):
        rot_matrix = np.array([[math.cos(theta),-math.sin(theta)], # theta in radians
                              [math.sin(theta),math.cos(theta)]])
        current_pos = np.array([[self.x,self.y]])
        global_pos_T = current_pos.T + np.dot(rot_matrix,np.transpose([[pt.r*math.cos(math.radians(pt.theta)),pt.r*math.sin(math.radians(pt.theta))]]))
        global_pos_T = np.transpose(global_pos_T).reshape(-1,)
        return global_pos_T
    
class trajectory():
    def __init__(self,agent_location,agent_velocity,target_location,target_velocity):
        self.agent_pos = agent_location
        self.agent_vel = agent_velocity
        self.target_pos = target_location
        self.target_vel = target_velocity

    def TfandU(self,x0,v0,xf,vf):
        u = 1
        sqr = 0.5*((v0)**2) + 0.5*((vf)**2) - u*(x0 - xf)
        if sqr>0:
            Tf = (-v0-vf)/u + 2*math.sqrt(sqr)
        elif sqr<0:
            u = -1
            sqr = 0.5*((v0)**2) + 0.5*((vf)**2) - u*(x0 - xf)
            Tf = (-v0-vf)/u + 2*math.sqrt(sqr)
        else:
            Tf = (-v0-vf)/u + 2*math.sqrt(sqr)
        return u,Tf
    
    def single_switch(self,x0,xf,v0,vf,t):
        u,Tf = self.TfandU(x0,xf,v0,vf)
        Tm = 0.5*(((vf - v0)/u) + Tf)
        if t >= 0 and t < Tm:
            V = v0 + u*(t)
            X = x0 + v0*t + 0.5*u*t**2
        elif t >= Tm and t <= Tf:
            V = v0 + u*(2*Tm - t)
            X = x0 + v0*t - u*Tm**2 - 0.5*u*t**2 + 2*u*Tm*t
        return X,V
    
    def double_switch(self,x0,xf,v0,vf,u,Tfd,t):
        alpha = (vf - v0 - u*Tfd)/(2*u)
        t2 = (x0 - xf + v0*Tfd + 2*alpha*u*Tfd + 0.5*u*(Tfd**2) - u*(alpha**2))/(2*alpha*u)
        t1 = t2 + alpha
        if t >= 0 and t < t1:
            V = v0 + u*t
            X = x0 + v0*t + 0.5*u*t**2
        elif t >= t1 and t < t2:
            V = v0 + u*(2*t1-t)
            X = x0 + v0*t + 2*u*t*t1 - u*(t1**2) - 0.5*u*(t**2)
        elif t >= t2 and t <=Tfd:
            V = v0 + u*(2*t1 - 2*t2 + t)
            X = x0 + v0*t + 2*u*t1*t - u*(t1**2) + u*(t2**2) - 2*u*t2*t + 0.5*u*(t**2)
        return X,V
    
    def unconstrained_traj_max(self,t):
        _,Tfx = self.TfandU(self.agent_pos[0],self.agent_vel[0],self.target_pos[0],self.target_vel[0])
        _,Tfy = self.TfandU(self.agent_pos[1],self.agent_vel[1],self.target_pos[1],self.target_vel[1])
        Tf = max(Tfx,Tfy)
        if Tfx > Tfy:
            x,Vx = self.single_switch(self.agent_pos[0],self.target_pos[0],self.agent_vel[0],self.target_vel[0],t)
            ymax,Vymax = self.double_switch(self.agent_pos[1],self.target_pos[1],self.agent_vel[1],self.target_vel[1],1,Tf,t)
            pos = np.array([x,ymax])
            vel = np.array([Vx,Vymax])
        elif Tfy>Tfx:
            y,Vy = self.single_switch(self.agent_pos[1],self.target_pos[1],self.agent_vel[1],self.target_vel[1],t)
            xmax,Vxmax = self.double_switch(self.agent_pos[0],self.target_pos[0],self.agent_vel[0],self.target_vel[0],1,Tf,t)
            pos = np.array([xmax,y])
            vel = np.array([Vxmax,Vy])
        else:
            x,Vx = self.single_switch(self.agent_pos[0],self.target_pos[0],self.agent_vel[0],self.target_vel[0],t)
            y,Vy = self.single_switch(self.agent_pos[1],self.target_pos[1],self.agent_vel[1],self.target_vel[1],t)
            pos = np.array([x,y])
            vel = np.array([Vx,Vy])
        return pos,vel
    
    def unconstrained_traj_min(self,t):
        _,Tfx = self.TfandU(self.agent_pos[0],self.agent_vel[0],self.target_pos[0],self.target_vel[0])
        _,Tfy = self.TfandU(self.agent_pos[1],self.agent_vel[1],self.target_pos[1],self.target_vel[1])
        Tf = max(Tfx,Tfy)
        if Tfx > Tfy:
            x,Vx = self.single_switch(self.agent_pos[0],self.target_pos[0],self.agent_vel[0],self.target_vel[0],t)
            ymax,Vymax = self.double_switch(self.agent_pos[1],self.target_pos[1],self.agent_vel[1],self.target_vel[1],-1,Tf,t)
            pos = np.array([x,ymax])
            vel = np.array([Vx,Vymax])
        elif Tfy>Tfx:
            y,Vy = self.single_switch(self.agent_pos[1],self.target_pos[1],self.agent_vel[1],self.target_vel[1],t)
            xmax,Vxmax = self.double_switch(self.agent_pos[0],self.target_pos[0],self.agent_vel[0],self.target_vel[0],-1,Tf,t)
            pos = np.array([xmax,y])
            vel = np.array([Vxmax,Vy])
        else:
            x,Vx = self.single_switch(self.agent_pos[0],self.target_pos[0],self.agent_vel[0],self.target_vel[0],t)
            y,Vy = self.single_switch(self.agent_pos[1],self.target_pos[1],self.agent_vel[1],self.target_vel[1],t)
            pos = np.array([x,y])
            vel = np.array([Vx,Vy])
        return pos,vel
    

        






