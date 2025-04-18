
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

# --- 辅助函数 (完整定义) ---
def calculate_orientation(point1, point2):
    """根据两点计算方向 (四元数和欧拉角)"""
    diff = np.array(point2) - np.array(point1)
    # 防止除零或无效输入
    if np.linalg.norm(diff) < 1e-6:
        # 如果两点重合，返回默认姿态 (无旋转)
        return p.getQuaternionFromEuler([0,0,0]), [0,0,0]

    # 计算 Yaw (绕 Z 轴)
    if np.linalg.norm(diff[:2]) < 1e-6: # 水平投影接近零 (垂直线)
        yaw = 0 # Yaw 可以任意，设为0
    else:
        yaw = math.atan2(diff[1], diff[0])

    # 计算 Pitch (绕 Y 轴)
    pitch = math.atan2(-diff[2], math.sqrt(diff[0] ** 2 + diff[1] ** 2))

    # 假设 Roll (绕 X 轴) 为 0
    roll = 0

    # 返回四元数和欧拉角
    return p.getQuaternionFromEuler([roll, pitch, yaw]), [roll, pitch, yaw]

def rotate_point_3d(point, rotation_angles):
    """根据欧拉角旋转一个3D点"""
    rx, ry, rz = rotation_angles

    # 检查输入角度是否有效
    if any(math.isnan(angle) or math.isinf(angle) for angle in rotation_angles):
        print(f"Warning: Invalid rotation angles {rotation_angles}. Returning original point.")
        return tuple(point)

    # 将角度限制在 0-2pi (可选，math.cos/sin 可以处理)
    rx = rx % (2 * math.pi)
    ry = ry % (2 * math.pi)
    rz = rz % (2 * math.pi)

    # 计算各轴旋转矩阵
    cos_rx, sin_rx = np.cos(rx), np.sin(rx)
    cos_ry, sin_ry = np.cos(ry), np.sin(ry)
    cos_rz, sin_rz = np.cos(rz), np.sin(rz)

    rotation_x = np.array([[1, 0, 0], [0, cos_rx, -sin_rx], [0, sin_rx, cos_rx]])
    rotation_y = np.array([[cos_ry, 0, sin_ry], [0, 1, 0], [-sin_ry, 0, cos_ry]])
    rotation_z = np.array([[cos_rz, -sin_rz, 0], [sin_rz, cos_rz, 0], [0, 0, 1]])

    # 组合旋转矩阵 (ZYX顺序)
    rotation_matrix = np.dot(rotation_z, np.dot(rotation_y, rotation_x))

    # 应用旋转
    rotated_point = np.dot(rotation_matrix, point)
    return tuple(rotated_point)

# --- 新增：绳长变化量 -> 曲率 的转换函数 ---
def calculate_curvatures_from_dl(dl_segment, d, L0_seg, num_cables=3):
    """
    根据绳长变化量计算曲率 ux, uy。
    !!! 重要: 这是一个基于简化假设的示例函数 !!!
    !!! 你需要根据你的机器人几何结构和论文模型替换这里的计算逻辑 !!!

    Args:
        dl_segment (np.array): 当前段的绳长变化量 [dl1, dl2, ..., dln].
        d (float): 绳索到中心线的距离 (半径).
        L0_seg (float): 当前段的参考长度 (假设不可伸长).
        num_cables (int): 绳索数量.

    Returns:
        tuple: (ux, uy) 计算得到的曲率.
    """
    ux = 0.0
    uy = 0.0
    # 避免除零
    if abs(d) < 1e-9 or abs(L0_seg) < 1e-9:
        return 0.0, 0.0
    # 检查输入长度
    if len(dl_segment) != num_cables:
        print(f"Warning: dl_segment length {len(dl_segment)} != num_cables {num_cables}")
        return ux, uy

    # --- 在这里替换成你论文或推导的公式 ---
    # 以下是针对 3 根对称绳索 (0, 120, 240度) 的简化示例
    if num_cables == 3:
        dl1, dl2, dl3 = dl_segment[0], dl_segment[1], dl_segment[2]
        # 简化比例因子 (需要标定或精确推导!)
        gain = 1.0 / (d * L0_seg) # 简化近似 gain

        # uy (绕y轴弯曲) 主要由绳1相对于绳2和绳3的平均长度变化引起
        uy = gain * (dl1 - (dl2 + dl3) / 2.0)
        # ux (绕x轴弯曲) 主要由绳2和绳3的长度差异引起 (注意符号和坐标系!)
        ux = gain * (math.sqrt(3.0) / 2.0) * (dl3 - dl2) # 乘以 sqrt(3)/2 是几何关系
    else:
        # 可以为其他绳索数量添加逻辑，或报错
        print(f"Error: Curvature calculation only implemented for 3 cables in this example.")
        # 或者返回0，或者抛出异常
        # raise NotImplementedError(f"Curvature calculation for {num_cables} cables not implemented.")
    # --- 替换结束 ---

    # 添加一些限制避免曲率过大导致数值问题 (可选)
    max_curvature = 10.0 # 示例限制值
    ux = np.clip(ux, -max_curvature, max_curvature)
    uy = np.clip(uy, -max_curvature, max_curvature)

    return ux, uy

# --- 新增：曲率 -> ODE抽象动作 的转换函数 ---
def convert_curvatures_to_ode_action(ux, uy, length_change, d, L0_seg):
    """将计算出的曲率转换为原始ODE.updateAction期望的抽象action格式。"""
    # 根据 ODE.updateAction 的反向逻辑:
    # uy = action[1] / (l * d)  => action[1] = uy * l * d
    # ux = action[2] / -(l * d) => action[2] = -ux * l * d
    l = L0_seg # 假设长度不变
    action_ode = np.zeros(3)
    action_ode[0] = length_change # 长度变化量
    action_ode[1] = uy * l * d
    action_ode[2] = -ux * l * d
    return action_ode

# --- 主程序 ---
if __name__ == "__main__":

    # --- 参数设置 ---
    print("--- Setting Parameters ---")
    body_color = [1, 0.0, 0.0, 1] # red body
    head_color = [0.0, 0.0, 0.75, 1] # blue head
    body_sphere_radius = 0.02
    number_of_sphere = 30 # PyBullet中显示用的球体数
    number_of_segment = 1 # !!! 机器人只有一段 !!!
    num_cables = 3        # !!! 你的机器人的绳索数量 !!!
    cable_distance = 7.5e-3 # !!! 你的机器人的绳索分布半径 d !!!
    initial_length = 0.06  # !!! 你的机器人的总初始长度 L0 !!!

    my_sphere_radius = body_sphere_radius
    my_number_of_sphere = number_of_sphere
    my_number_of_segment = number_of_segment # = 1
    my_head_color = head_color
    my_max_grasp_width = 0.02
    my_grasp_width = 1 * my_max_grasp_width

    if my_number_of_segment <= 0:
        print("Error: number_of_segment must be positive.")
        sys.exit(1)
    L0_seg = initial_length / my_number_of_segment # = initial_length
    print(f"Segments: {my_number_of_segment}, Cables: {num_cables}, Total Length: {initial_length:.4f}")

    # --- PyBullet 初始化 ---
    print("--- Initializing PyBullet ---")
    simulationStepTime = 0.01
    try:
        physicsClientId = p.connect(p.GUI)
        if physicsClientId < 0: raise ConnectionError("Failed to connect.")
        print(f"Connected to PyBullet with Client ID: {physicsClientId}")
    except Exception as e:
        print(f"Error connecting to PyBullet: {e}"); sys.exit(1)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(simulationStepTime)
    try:
        planeId = p.loadURDF("plane.urdf")
        print(f"Loaded plane.urdf with ID: {planeId}")
    except p.error as e:
        print(f"Error loading plane.urdf: {e}"); sys.exit(1)

    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
    p.resetDebugVisualizerCamera(cameraDistance=0.7, cameraYaw=90, cameraPitch=-30, cameraTargetPosition=[0, 0.2, 0.1])

    # --- 初始化 ODE 对象 ---
    print("--- Initializing ODE Object ---")
    my_ode = ODE()
    my_ode.l0 = initial_length
    my_ode.d = cable_distance
    print(f"ODE initialized with L0={my_ode.l0:.4f}, d={my_ode.d:.4f}")

    # --- 计算初始形态 ---
    print("--- Calculating Initial Shape ---")
    act0_segment = np.zeros(3)
    my_ode.y0 = np.array([0,0,0, 1,0,0, 0,1,0, 0,0,1]) # 确保初始 y0 正确
    my_ode.updateAction(act0_segment)
    sol0 = my_ode.odeStepFull()
    if sol0 is None or sol0.shape[1] == 0:
        print("Error: Initial ODE solve failed.")
        p.disconnect()
        sys.exit(1)
    print(f"Initial shape calculated. Sol0 shape: {sol0.shape}")

    my_base_pos = np.array([0, 0, 0.3]) # 世界坐标基座位置
    radius = my_sphere_radius

    # --- 创建 PyBullet 形状 ---
    print("--- Creating PyBullet Shapes ---")
    try:
        shape = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
        visualShapeId = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=body_color)
        # 创建3个滑块分别控制3根绳的 dl
        dl_sliders = []
        dl_sliders.append(p.addUserDebugParameter("dl_cable_1", -0.01, 0.01, 0.0))
        dl_sliders.append(p.addUserDebugParameter("dl_cable_2", -0.01, 0.01, 0.0))
        dl_sliders.append(p.addUserDebugParameter("dl_cable_3", -0.01, 0.01, 0.0))
        # 末端的可视形状
        visualShapeId_tip = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.02, 0.002, 0.001], rgbaColor=head_color)
        visualShapeId_tip_ = p.createVisualShape(p.GEOM_SPHERE, radius=radius+0.0025, rgbaColor=my_head_color)
    except p.error as e:
        print(f"Error creating shapes: {e}"); p.disconnect(); sys.exit(1)

    # --- 创建 PyBullet 物体 ---
    print("--- Creating PyBullet Bodies ---")
    t = 0
    dt = 0.01
    if sol0.shape[1] < 2: print("Error: Initial ODE solution has < 2 points."); p.disconnect(); sys.exit(1)
    # 从初始解采样点
    idx0 = np.linspace(0, sol0.shape[1] - 1, my_number_of_sphere, dtype=int)
    positions0 = [(sol0[0, i], sol0[2, i], sol0[1, i]) for i in idx0] # 提取局部位置
    my_robot_bodies = []
    # 创建身体球体
    for i, pos in enumerate(positions0):
        try:
            # 确保 pos 是 NumPy 数组以便向量加法
            current_pos_world = np.array(pos) + my_base_pos
            body_id = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=shape,
                                          baseVisualShapeIndex=visualShapeId,
                                          basePosition=current_pos_world)
            my_robot_bodies.append(body_id)
        except p.error as e:
            print(f"Error creating body {i}: {e}"); p.disconnect(); sys.exit(1)

    # 创建末端物体 (头部+夹爪指示器)
    if len(positions0) >= 2 and len(my_robot_bodies) >= 1: # 至少需要身体部分才能创建末端
        try:
            ori0_local, _ = calculate_orientation(positions0[-2], positions0[-1])
            ori0_world = ori0_local # 初始基座姿态是 identity
            last_pos_world = np.array(positions0[-1]) + my_base_pos

            # 头部 (使用 visualShapeId_tip_)
            tip_id1 = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=shape,
                                       baseVisualShapeIndex=visualShapeId_tip_,
                                       basePosition=last_pos_world, baseOrientation=ori0_world)
            my_robot_bodies.append(tip_id1)
            # 夹爪指示器1 (使用 visualShapeId_tip)
            offset1 = p.rotateVector(ori0_world, [-0.01, 0, 0]) # 计算世界坐标下的偏移
            tip_id2 = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=shape,
                                       baseVisualShapeIndex=visualShapeId_tip,
                                       basePosition=last_pos_world + offset1, baseOrientation=ori0_world)
            my_robot_bodies.append(tip_id2)
            # 夹爪指示器2 (使用 visualShapeId_tip)
            offset2 = p.rotateVector(ori0_world, [0.01, 0, 0]) # 计算世界坐标下的偏移
            tip_id3 = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=shape,
                                       baseVisualShapeIndex=visualShapeId_tip,
                                       basePosition=last_pos_world + offset2, baseOrientation=ori0_world)
            my_robot_bodies.append(tip_id3)
            print(f"Created {len(my_robot_bodies)} bodies in total.")
        except p.error as e:
            print(f"Error creating tip bodies: {e}"); p.disconnect(); sys.exit(1)
        except IndexError:
             print("Error: Not enough points in positions0 to calculate orientation.")
             p.disconnect(); sys.exit(1)
    else:
        print("Warning: Cannot create tip bodies due to insufficient initial points or body parts.")
        # 如果没有末端物体，后续更新末端的代码会出错，可能需要调整

    my_robot_line_ids = []
    print("--- Initialization Complete, Starting Simulation Loop ---")

    # --- 主循环 ---
    frame_count = 0
    last_print_time = time.time()
    while True:
        try:
            # --- 读取滑块值 (作为绳长变化量) ---
            target_cable_changes = np.zeros(num_cables) # 长度为 3
            target_cable_changes[0] = p.readUserDebugParameter(dl_sliders[0])
            target_cable_changes[1] = p.readUserDebugParameter(dl_sliders[1])
            target_cable_changes[2] = p.readUserDebugParameter(dl_sliders[2])

            # --- 设定基座位姿 ---
            base_pos = np.array([0, 0, 0.3])
            base_ori_euler = np.array([0, 0, 0]) # (Roll, Pitch, Yaw)

            # --- 计算新形态 (单段) ---
            my_ode._reset_y0() # 每次都从基座状态开始

            # 1. 获取单段的绳长变化量
            dl_segment = target_cable_changes

            # 2. 计算对应的曲率 ux, uy
            #    !!! 使用脚本中定义的参数 !!!
            ux, uy = calculate_curvatures_from_dl(dl_segment, cable_distance, L0_seg, num_cables)

            # 3. 将曲率转换为原始 ODE 的抽象 action
            action_ode_segment = convert_curvatures_to_ode_action(ux, uy, 0.0, cable_distance, L0_seg)

            # 4. 调用原始 ODE 求解 (单次调用)
            my_ode.updateAction(action_ode_segment)
            sol = my_ode.odeStepFull() # sol 直接就是最终解 (形状为 (12, k))

            # --- 后续更新 PyBullet 对象 ---
            if sol is not None and sol.shape[1] >= 2: # 检查解是否有效且至少有2个点
                base_ori = p.getQuaternionFromEuler(base_ori_euler)
                my_base_pos, my_base_ori = base_pos, base_ori # 更新当前的基座位姿
                my_base_pos_offset = np.array([0,0,0]) # 简化偏移

                # 从解中采样点
                idx = np.linspace(0, sol.shape[1] -1, my_number_of_sphere, dtype=int)
                positions = [(sol[0, i], sol[2, i], sol[1, i]) for i in idx] # 提取局部位置
                my_robot_line_ids = [] # 清除旧线 (如果绘制了的话)

                pose_in_word_frame = []
                # 确保更新时不越界
                num_bodies_to_update = len(my_robot_bodies)
                num_points_available = len(positions)
                num_spheres_to_update = min(num_bodies_to_update - 3, num_points_available) # 减3是因为末端单独处理

                # 更新球体 (身体部分)
                for i in range(num_spheres_to_update):
                    pos_local = positions[i]
                    pos_world, orin_world = p.multiplyTransforms(my_base_pos + my_base_pos_offset,
                                                                 my_base_ori,
                                                                 pos_local,
                                                                 [0, 0, 0, 1])
                    pose_in_word_frame.append(np.concatenate((np.array(pos_world),np.array(orin_world))))
                    try:
                        p.resetBasePositionAndOrientation(my_robot_bodies[i], pos_world, orin_world)
                    except p.error: pass # 忽略可能的错误

                # 更新末端
                if num_points_available >= 3 and num_bodies_to_update >= 3:
                     try:
                         # 计算世界坐标的头部位置 (使用最后一个点)
                         head_pos_world = np.array(p.multiplyTransforms(my_base_pos + my_base_pos_offset, my_base_ori, positions[-1], [0,0,0,1])[0])
                         # 计算世界坐标的末端姿态
                         _tip_ori_local, tip_ori_euler = calculate_orientation(positions[-3], positions[-1])
                         _, tip_ori_world = p.multiplyTransforms(my_base_pos + my_base_pos_offset, my_base_ori, [0,0,0], _tip_ori_local)

                         # 计算世界坐标的夹爪位置
                         gripper_pos1_local_rotated = rotate_point_3d([0.02, -my_grasp_width, 0], tip_ori_euler)
                         gripper_pos2_local_rotated = rotate_point_3d([0.02, my_grasp_width, 0], tip_ori_euler)
                         gripper_pos1_world, _ = p.multiplyTransforms(head_pos_world, tip_ori_world, gripper_pos1_local_rotated, [0,0,0,1])
                         gripper_pos2_world, _ = p.multiplyTransforms(head_pos_world, tip_ori_world, gripper_pos2_local_rotated, [0,0,0,1])

                         # 更新 PyBullet 对象 (使用负索引访问末端)
                         p.resetBasePositionAndOrientation(my_robot_bodies[-3], head_pos_world, tip_ori_world)
                         p.resetBasePositionAndOrientation(my_robot_bodies[-2], gripper_pos1_world, tip_ori_world)
                         p.resetBasePositionAndOrientation(my_robot_bodies[-1], gripper_pos2_world, tip_ori_world)
                     except p.error: pass # 忽略可能的错误
                     except IndexError: pass # 忽略可能的索引错误
                # else:
                #      print(f"Warning: Not enough points ({num_points_available}) or bodies ({num_bodies_to_update}) to update robot tip.")
            elif sol is not None and sol.shape[1] < 2:
                 print("Warning: ODE solution has < 2 points. Skipping PyBullet update.")
            else: # sol is None
                 print("Warning: ODE solve failed. Skipping PyBullet update.")


            # --- 仿真步进和暂停 ---
            p.stepSimulation()
            time.sleep(dt)
            frame_count += 1
            current_time = time.time()
            if current_time - last_print_time >= 1.0:
                # print(f"FPS: {frame_count / (current_time - last_print_time):.2f}")
                frame_count = 0
                last_print_time = current_time

        except KeyboardInterrupt:
            print("\nSimulation interrupted by user.")
            break
        except p.error as e:
            print(f"PyBullet error occurred (is the simulation window still open?): {e}")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            import traceback
            traceback.print_exc()
            break

    print("Disconnecting from PyBullet.")
    try:
        if p.isConnected():
             p.disconnect()
    except p.error:
        pass # 可能已经断开