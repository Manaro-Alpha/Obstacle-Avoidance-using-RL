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
    

    

        






