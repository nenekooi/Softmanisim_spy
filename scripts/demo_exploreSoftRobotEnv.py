import numpy as np
import time 
import pybullet as p

import sys
import os

# 动态添加 SoftManiSim 文件夹到 sys.path
softmanisim_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if softmanisim_path not in sys.path:
    sys.path.append(softmanisim_path)
from pybullet_env.BasicEnvironment import SoftRobotBasicEnvironment

if __name__ == "__main__":
    
    # soft_robot_1 = SoftRobotBasicEnvironment() 
    env = SoftRobotBasicEnvironment(number_of_sphere=60, 
                                    number_of_segment=3, 
                                    body_color=[1, 0, 0, 1])
    # env.create_robot()
    # env.add_a_cube(pos=[0.5, 0.5, 0.1], size=[0.1, 0.1, 0.1])
    
    base_link_shape = env.bullet.createVisualShape(env.bullet.GEOM_BOX, halfExtents=[0.05, 0.05, 0.03], rgbaColor=[0.6, 0.6, 0.6, 1])
    base_link_pos, base_link_ori = env.bullet.multiplyTransforms([0,0,0.5], [0,0,0,1], [0,-0.0,0], [0,0,0,1])
    base_link_id    = env.bullet.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=base_link_shape,
                                                        baseVisualShapeIndex=base_link_shape,
                                                        basePosition= base_link_pos , baseOrientation=base_link_ori)
       
    print(f"base_link_id: {base_link_id}, base_link_pos: {base_link_pos}, base_link_ori: {base_link_ori}")   
    # for i in range(2):   
    #     color = np.ones([1,4]).squeeze()
    #     color[:3] = np.random.rand(1,3).squeeze()
    #     obj_shape = env.bullet.createVisualShape(env.bullet.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02], rgbaColor=color)
    #     obj_pos, obj_ori = env.bullet.multiplyTransforms([0.005*np.random.randint(-100,100),0.005*np.random.randint(-100,100),0.05], [0,0,0,1], [0,-0.0,0], [0,0,0,1])
    #     obj_id    = env.bullet.createMultiBody(baseMass=0.00, baseCollisionShapeIndex=base_link_shape,
    #                                                         baseVisualShapeIndex=obj_shape,
    #                                                         basePosition= obj_pos , baseOrientation=obj_ori)
    
    
    #TODO : add a head shape   
    sf1_seg3_cable_0_Slider = p.addUserDebugParameter("sf1_seg3_cable_0", -0.05, 0.05, 0)
    sf1_seg3_cable_1_Slider = p.addUserDebugParameter("sf1_seg3_cable_1", -0.05, 0.05, 0)
    sf1_seg3_cable_2_Slider = p.addUserDebugParameter("sf1_seg3_cable_2", -0.05, 0.05, 0)
    base_orin_x_slider = p.addUserDebugParameter("base_orin_x", -np.pi/2, np.pi/2 , 0) 
    # env.move_robot_ori(action=np.array([0.3, 0.3, 0.3]))
    t = 0
    dt = 0.01
    while True:
        # soft_robot_1.updateAction(np.array([0.5,0.5,0.5]))
        # soft_robot_1.odeStepFull()
        t += dt
        # sf1_seg1_cable_1   = .005*np.sin(0.5*np.pi*t)
        # sf1_seg1_cable_2   = .005*np.sin(0.5*np.pi*t)
        # sf1_seg2_cable_1   = .005*np.sin(0.5*np.pi*t+1)
        # sf1_seg2_cable_2   = .005*np.sin(0.5*np.pi*t+1)
        # sf1_seg3_cable_0   = .00*np.sin(0.5*np.pi*t)
        # sf1_seg3_cable_1   = .005*np.sin(0.5*np.pi*t+2)
        # sf1_seg3_cable_2   = .005*np.sin(0.5*np.pi*t+2)
        # sf1_gripper_pos    = np.abs(np.sin(np.pi*t))
        
        # ampitude = 0
        # sf1_seg1_cable_1   = ampitude*np.sin(0.5*np.pi*t)
        # sf1_seg1_cable_2   = ampitude*np.sin(0.5*np.pi*t)
        # sf1_seg2_cable_1   = ampitude*np.sin(0.5*np.pi*t+1)
        # sf1_seg2_cable_2   = ampitude*np.sin(0.5*np.pi*t+1)
        # sf1_seg3_cable_0   = ampitude*np.sin(0.5*np.pi*t)
        # sf1_seg3_cable_1   = ampitude*np.sin(0.5*np.pi*t+2)
        # sf1_seg3_cable_2   = ampitude*np.sin(0.5*np.pi*t+2)
        # sf1_gripper_pos    = np.abs(np.sin(np.pi*t))
        
        new_pos = np.array([ 1, 1, 1])
        base_orin = np.array([0,0,0])
        new_pos = base_link_pos
        base_orin_x = p.readUserDebugParameter(base_orin_x_slider)
        base_orin = np.array([base_orin_x,0,0])       
        # env.move_robot_ori(action=np.array([0.0, sf1_seg1_cable_1, sf1_seg1_cable_2, 
        #                                     0.0, sf1_seg2_cable_1, sf1_seg2_cable_2,
        #                                     sf1_seg3_cable_0, sf1_seg3_cable_1, sf1_seg3_cable_2]),
        #                 base_pos = new_pos, base_orin = base_orin) # 计算第一个软体机械臂的末端
        sf1_seg3_cable_0 = p.readUserDebugParameter(sf1_seg3_cable_0_Slider)
        sf1_seg3_cable_1 = p.readUserDebugParameter(sf1_seg3_cable_1_Slider)
        sf1_seg3_cable_2 = p.readUserDebugParameter(sf1_seg3_cable_2_Slider)

        env.move_robot_ori(action=np.array([0.0, 0, 0, 
                                            0.0, 0, 0,
                                            sf1_seg3_cable_0, sf1_seg3_cable_1, sf1_seg3_cable_2]),
                        base_pos = new_pos, base_orin = base_orin) # 计算第一个软体机械臂的末端        
        

        env.wait(0.01)
        
        