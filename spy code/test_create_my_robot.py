import numpy as np
import time
import sys
import os
import pybullet as p 
softmanisim_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(softmanisim_path)
from visualizer.visualizer import ODE
from pybullet_env.BasicEnvironment import SoftRobotBasicEnvironment
from visualizer.visualizer import ODE 

if __name__ == "__main__":
    
    yomo = SoftRobotBasicEnvironment(number_of_segment=1, head_color=[1,0.75, 1])
 
    t = 0
    dt = 0.01
    counter = 0
    
    
    while True:
        t += dt
        counter += 1
        #TODO this part substitute the real data
        yomo_cable_1 = 0
        yomo_cable_2 = 0
        yomo_cable_3 = 0
        
        yomo.move_robot_ori(action=np.array([0.0, yomo_cable_1, yomo_cable_2]),
                            base_pos = np.array([0.0, 0.0, 1.0]), #修改底座
                            base_orin= np.array([0, 0, -np.pi/2])) #修改底座     
        
        head_pose, head_ori = yomo.calc_tip_pos(action=np.array([0.0, yomo_cable_1, yomo_cable_2]),
                            base_pos = np.array([0.0, 0.0, 1.0]), #修改底座
                            base_orin= np.array([0, 0, -np.pi/2])) #修改底座     
        
        if counter % 200 == 0:
            print("hello world")
            formatted_head_pose = [f"{head_pose:.2f}" for head_pose in head_pose]
            formatted_head_ori = [f"{head_ori:.2f}" for head_ori in head_ori]
            print(f"the time at {t:.2f}, the head position is {formatted_head_pose}, and the orientation is {formatted_head_ori}")
        
        
        
        # add a feature to draw the tip position 
        
        