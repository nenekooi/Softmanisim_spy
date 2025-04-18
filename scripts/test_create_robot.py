
import sys
import os

# 动态添加 SoftManiSim 文件夹到 sys.path
softmanisim_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if softmanisim_path not in sys.path:
    sys.path.append(softmanisim_path)

from visualizer.visualizer import ODE 
import numpy as np
import pybullet as p
import pybullet_data
import time 
import math 
from pprint import pprint

def calculate_orientation(point1, point2):
    # Calculate the difference vector
    diff = np.array(point2) - np.array(point1)

    # Calculate yaw (around z-axis)
    yaw = math.atan2(diff[1], diff[0])

    # Calculate pitch (around y-axis)
    pitch = math.atan2(-diff[2], math.sqrt(diff[0] ** 2 + diff[1] ** 2))

    # Roll is arbitrary in this context, setting it to zero
    roll = 0
    
    if pitch < 0 : 
        pitch += np.pi*2
    if yaw < 0 : 
        yaw += np.pi*2
        
    return p.getQuaternionFromEuler([roll, pitch, yaw]),[roll, pitch, yaw]


def rotate_point_3d(point, rotation_angles):
    """
    Rotates a 3D point around the X, Y, and Z axes.

    :param point: A tuple or list of 3 elements representing the (x, y, z) coordinates of the point.
    :param rotation_angles: A tuple or list of 3 elements representing the rotation angles (in rad) around the X, Y, and Z axes respectively.
    :return: A tuple representing the rotated point coordinates (x, y, z).
    """
    # Convert angles to radians
    # rotation_angles = np.radians(rotation_angles)
    
    rx, ry, rz = rotation_angles

    # Rotation matrices for X, Y, Z axes
    rotation_x = np.array([[1, 0, 0],
                        [0, np.cos(rx), -np.sin(rx)],
                        [0, np.sin(rx), np.cos(rx)]])
    
    rotation_y = np.array([[np.cos(ry), 0, np.sin(ry)],
                        [0, 1, 0],
                        [-np.sin(ry), 0, np.cos(ry)]])
    
    rotation_z = np.array([[np.cos(rz), -np.sin(rz), 0],
                        [np.sin(rz), np.cos(rz), 0],
                        [0, 0, 1]])

    # Combined rotation matrix
    rotation_matrix = np.dot(rotation_z, np.dot(rotation_y, rotation_x))

    # Rotate the point
    rotated_point = np.dot(rotation_matrix, point)

    return tuple(rotated_point)




if __name__ == "__main__":

    # body_color = [0.5, .0, 0.6, 1] # purple body
    body_color = [1, 0.0, 0.0, 1] # red body
    head_color = [0.0, 0.0, 0.75, 1] # blue head
    body_sphere_radius = 0.02
    number_of_sphere = 30
    number_of_segment = 3
    my_sphere_radius = body_sphere_radius 
    my_number_of_sphere = number_of_sphere
    my_number_of_segment = number_of_segment
    my_head_color = head_color     
    my_max_grasp_width = 0.02
    my_grasp_width = 1 * my_max_grasp_width
    
    
    simulationStepTime = 0.05
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(simulationStepTime)
    planeId = p.loadURDF("plane.urdf")
    print(f"planeId: {planeId}")

    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
    p.resetDebugVisualizerCamera(cameraDistance=0.7, 
                                cameraYaw=180, 
                                cameraPitch=-35, 
                                cameraTargetPosition=[0, 0, 0.1])

    # initialize the system. 
    my_ode = ODE()
    act = np.array([0,0,0])
    my_ode.updateAction(act)
    sol = my_ode.odeStepFull()
    print(sol)
    my_base_pos_init = np.array([0, 0, 0.0])
    my_base_pos      = np.array([0, 0, 0.5])
    radius = my_sphere_radius # _xxx internal variable is named with my_ begin
    
    shape = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
    visualShapeId = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=body_color)
    #TODO : add a head shape   
    redSlider = p.addUserDebugParameter("red", 0, 1, 1)
    visualShapeId_tip = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.02, 0.002, 0.001], rgbaColor=head_color)  
    visualShapeId_tip_ = p.createVisualShape(p.GEOM_SPHERE, radius=radius+0.0025, rgbaColor=my_head_color)
    
    t = 0
    dt = 0.01

    # load the position
    idx = np.linspace(0, sol.shape[1] -1, my_number_of_sphere, dtype=int)    # 包含最后一个数 序数0-19
    positions = [(sol[0, i], sol[2, i], sol[1, i]) for i in idx]
    
    pprint(idx)
    pprint(positions)
    
    my_robot_bodies = [p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=shape,
                                            baseVisualShapeIndex=visualShapeId,
                                            basePosition=pos + my_base_pos) for pos in positions]
    pprint(my_robot_bodies)
    
    ori, _ = calculate_orientation(positions[-2], positions[-1]) 
    print(f"ori: {ori}")  
    my_robot_bodies.append(p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=shape,
                                             baseVisualShapeIndex=visualShapeId_tip_,
                                             basePosition=positions[-1] + my_base_pos,
                                             baseOrientation=ori))
    my_robot_bodies.append(p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=shape,
                                             baseVisualShapeIndex=visualShapeId_tip,
                                             basePosition=positions[-1] + my_base_pos + [-0.01, 0, 0],
                                             baseOrientation=ori))
    my_robot_bodies.append(p.createMultiBody(baseMass=0, baseCollisionShapeIndex=shape,
                                            baseVisualShapeIndex=visualShapeId_tip,
                                            basePosition=positions[-1] + my_base_pos + [0.01,0,0], baseOrientation=ori))    
    
    my_robot_line_ids = []
    
    #TODO: create the robot move code 
    #TODO: 阅读相关的paper 和数学模型相关和pybullet的API
        
    while True:
        red = p.readUserDebugParameter(redSlider)
        p.changeVisualShape(my_robot_bodies[-1], -1, rgbaColor=[red, 0.0, 0.75, 1])
        
        
        # move_robot_ori
        # action 里面的元素 
        # [ l0_seg1, uy_seg1, ux_seg1,
        #   l0_seg2, uy_seg2, ux_seg2,
        #   l0_seg3, uy_seg3, ux_seg3] 
        action = np.array([0,0,0, 
                           0,-0.02,0, 
                           0,0,0.02])
        base_pos = np.array([0, 0, 0.5])
        base_ori = np.array([0, 0, 0])
        
        
        my_ode._reset_y0() 
        sol = None
        
        for n in range(my_number_of_segment):         
            my_ode.updateAction(action[n*3:(n+1)*3])
            sol_n = my_ode.odeStepFull()
            my_ode.y0 = sol_n[:,-1] #!最后一行，y的末端信息
            
            if sol is None:
                sol = np.copy(sol_n)
            else:
                sol = np.concatenate((sol, sol_n), axis=1) #按列拼接 
                
        # print(sol.shape)    # (12, 60)
        
        base_ori = p.getQuaternionFromEuler(base_ori)
        my_base_pos, my_base_ori = base_pos, base_ori
        my_base_pos_init = np.array(p.multiplyTransforms([0, 0, 0], [0, 0 ,0 ,1],
                                                         my_base_pos_init, base_ori)[0])
        my_base_pos_offset = np.array(p.multiplyTransforms([0, 0, 0], [0, 0 ,0 ,1],
                                                           [0, -0.0, 0], base_ori)[0])
        #! sol中包含60个点，涵盖整个机械臂，
        idx = np.linspace(0, sol.shape[1] -1, my_number_of_sphere, dtype=int)    # 包含最后一个数 序数0-19
        #! 仍然是三十个点
        #! 参考函数updateAction, 先计算uy 再计算ux
        positions = [(sol[0, i], sol[2, i], sol[1, i]) for i in idx] 
        my_robot_line_ids = []
        # print(f"positions shape: {len(positions)}")
        pose_in_word_frame = []
        for i, pos in enumerate(positions):
            pos, orin = p.multiplyTransforms(my_base_pos + my_base_pos_offset,
                                             my_base_ori,
                                             pos,
                                             [0, 0, 0, 1])
            pose_in_word_frame.append(np.concatenate((np.array(pos), np.array(orin))))
            p.resetBasePositionAndOrientation(my_robot_bodies[i], pos, orin)
        
        #TODO：计算末端的方向 更新末端    
        head_pos = np.array(p.multiplyTransforms(my_base_pos + my_base_pos_offset,
                                                 my_base_ori,
                                                 positions[-1] + np.array([0, 0.0, 0]),
                                                 [0, 0, 0, 1])[0])
        
        my_tip_ori, tip_ori_euler = calculate_orientation(positions[-3], positions[-1]) 
        _, tip_ori = p.multiplyTransforms([0, 0, 0], base_ori,
                                                 [0, 0, 0],
                                                 my_tip_ori)
        
        
        gripper_pos1 = rotate_point_3d([0.02, -my_grasp_width, 0], tip_ori_euler)
        gripper_pos2 = rotate_point_3d([0.02, my_grasp_width, 0], tip_ori_euler)
        
        gripper_pos1 = np.array(p.multiplyTransforms(head_pos, my_base_ori,
                                                     gripper_pos1, [0, 0, 0, 1])[0])
        gripper_pos2 = np.array(p.multiplyTransforms(head_pos, my_base_ori,
                                                        gripper_pos2, [0, 0, 0, 1])[0]) 
        
        p.resetBasePositionAndOrientation(my_robot_bodies[-3], head_pos, my_tip_ori)
        my_head_pose = [head_pos, base_ori]
        p.resetBasePositionAndOrientation(my_robot_bodies[-2], gripper_pos1, my_tip_ori)
        p.resetBasePositionAndOrientation(my_robot_bodies[-1], gripper_pos2, my_tip_ori)
        
        
                                                                                  
                                    
        p.stepSimulation()
        time.sleep(dt)
        
        
