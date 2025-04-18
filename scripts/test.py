# 文件名: cable_control_test.py
# (建议放置在 SoftManiSim/scripts/ 目录下)

import sys
import os
import math # 确保导入 math
import numpy as np
import pybullet as p
import pybullet_data
import time
from pprint import pprint

# --- 动态添加 SoftManiSim 路径 ---
# 假设此脚本放在 SoftManiSim/scripts/ 目录下
softmanisim_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if softmanisim_path not in sys.path:
    sys.path.insert(0, softmanisim_path) # 使用 insert(0, ...) 确保优先搜索
    

# --- 辅助函数 (保持不变) ---
def calculate_orientation(point1, point2):
    """根据两点计算方向 (四元数和欧拉角)"""
    diff = np.array(point2) - np.array(point1)
    # 防止除零或无效输入
    if np.linalg.norm(diff[:2]) < 1e-6: # 水平投影接近零
        yaw = 0
    else:
        yaw = math.atan2(diff[1], diff[0])

    if np.linalg.norm(diff) < 1e-6:
        pitch = 0
    else:
        pitch = math.atan2(-diff[2], math.sqrt(diff[0] ** 2 + diff[1] ** 2))

    roll = 0 # 假设 Roll 为 0
    # 可选：将角度转为 0-2pi 范围
    # if pitch < 0: pitch += 2 * np.pi
    # if yaw < 0: yaw += 2 * np.pi
    return p.getQuaternionFromEuler([roll, pitch, yaw]), [roll, pitch, yaw]

def rotate_point_3d(point, rotation_angles):
    """根据欧拉角旋转一个3D点"""
    rx, ry, rz = rotation_angles
    # 检查 rotation_angles 是否包含非数值
    if any(math.isnan(angle) or math.isinf(angle) for angle in rotation_angles):
        print(f"Warning: Invalid rotation angles {rotation_angles}. Returning original point.")
        return tuple(point)
        
    # 避免在 math.cos/sin 中使用过大的值 (虽然通常没问题)
    rx = rx % (2 * math.pi)
    ry = ry % (2 * math.pi)
    rz = rz % (2 * math.pi)

    rotation_x = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    rotation_y = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    rotation_z = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    rotation_matrix = np.dot(rotation_z, np.dot(rotation_y, rotation_x))
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
    if abs(d) < 1e-9 or abs(L0_seg) < 1e-9: # 避免除零
        return 0.0, 0.0
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

    elif num_cables == 4: # 如果是4根绳索 (e.g., 0, 90, 180, 270度)
         dl1, dl2, dl3, dl4 = dl_segment[0], dl_segment[1], dl_segment[2], dl_segment[3]
         gain = 1.0 / (d * L0_seg) # gain 可能不同
         # uy (绕y轴弯曲) 由 dl1 vs dl3 决定 (假设1,3在y轴方向?) - 需要确认坐标系
         # ux (绕x轴弯曲) 由 dl2 vs dl4 决定 (假设2,4在x轴方向?) - 需要确认坐标系
         # 假设坐标系：1(+y), 2(+x), 3(-y), 4(-x)
         uy = gain * (dl1 - dl3) / 2.0 # 除以2是因为用了两根绳的差值
         ux = gain * (dl2 - dl4) / 2.0
    else:
        print(f"Warning: Curvature calculation for {num_cables} cables not implemented in this example.")
    # --- 替换结束 ---

    # 添加一些限制避免曲率过大导致数值问题 (可选)
    max_curvature = 10.0 # 示例限制值
    ux = np.clip(ux, -max_curvature, max_curvature)
    uy = np.clip(uy, -max_curvature, max_curvature)

    return ux, uy

# --- 新增：曲率 -> ODE抽象动作 的转换函数 ---
def convert_curvatures_to_ode_action(ux, uy, length_change, d, L0_seg):
    """
    将计算出的曲率转换为原始ODE.updateAction期望的抽象action格式。
    假设 action = [delta_L, uy_related, ux_related]
    """
    # 根据 ODE.updateAction 的反向逻辑:
    # uy = action[1] / (l * d)  => action[1] = uy * l * d
    # ux = action[2] / -(l * d) => action[2] = -ux * l * d
    # 假设 l = L0_seg (不可伸长)
    l = L0_seg
    action_ode = np.zeros(3)
    action_ode[0] = length_change # 假设长度变化量为输入，或直接设为0
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
    number_of_segment = 3 # 机器人分段数
    num_cables = 3        # !!! 你的机器人的绳索数量 !!!
    cable_distance = 7.5e-3 # !!! 你的机器人的绳索分布半径 d !!!
    initial_length = 0.6  # !!! 你的机器人的总初始长度 L0 !!!

    my_sphere_radius = body_sphere_radius
    my_number_of_sphere = number_of_sphere
    my_number_of_segment = number_of_segment
    my_head_color = head_color
    my_max_grasp_width = 0.02
    my_grasp_width = 1 * my_max_grasp_width
    
    if my_number_of_segment <= 0:
        print("Error: number_of_segment must be positive.")
        sys.exit(1)
    L0_seg = initial_length / my_number_of_segment # 计算每段的参考长度
    print(f"Segments: {my_number_of_segment}, Cables/Seg: {num_cables}, Total Length: {initial_length}, Seg Length: {L0_seg:.4f}")

    # --- PyBullet 初始化 ---
    print("--- Initializing PyBullet ---")
    simulationStepTime = 0.01 # 使用更小的时间步长可能更稳定
    try:
        physicsClientId = p.connect(p.GUI)
        if physicsClientId < 0:
            raise ConnectionError("Failed to connect to PyBullet GUI.")
        print(f"Connected to PyBullet with Client ID: {physicsClientId}")
    except Exception as e:
        print(f"Error connecting to PyBullet: {e}")
        sys.exit(1)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(simulationStepTime)
    try:
        planeId = p.loadURDF("plane.urdf")
        print(f"Loaded plane.urdf with ID: {planeId}")
    except p.error as e:
        print(f"Error loading plane.urdf: {e}")
        sys.exit(1)

    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
    p.resetDebugVisualizerCamera(cameraDistance=0.7, cameraYaw=90, cameraPitch=-30, cameraTargetPosition=[0, 0.2, 0.1]) # 调整视角

    # --- 初始化 ODE 对象 ---
    print("--- Initializing ODE Object ---")
    my_ode = ODE()
    print(f"Original my_ode.l0: {my_ode.l0}, Original my_ode.d: {my_ode.d}, Original my_ode.ds: {my_ode.ds}")
    my_ode.l0 = initial_length # 设置总初始长度
    my_ode.d = cable_distance  # 设置绳索距离
    # my_ode.ds = 0.0005 # 可以尝试更小的积分步长
    print(f"Modified my_ode.l0: {my_ode.l0}, Modified my_ode.d: {my_ode.d}, Current my_ode.ds: {my_ode.ds}")

    # --- 计算初始形态 ---
    print("--- Calculating Initial Shape ---")
    act0_segment = np.zeros(3)
    my_ode.y0 = np.array([0,0,0, 1,0,0, 0,1,0, 0,0,1]) # 确保初始 y0 正确
    current_y0_init = np.copy(my_ode.y0) # 保存全局初始 y0
    sol0_list = []
    for n in range(my_number_of_segment):
        my_ode.y0 = current_y0_init # 设置当前段的初始 y0
        my_ode.updateAction(act0_segment)
        sol_n = my_ode.odeStepFull()
        if sol_n is None or sol_n.shape[1] == 0:
             print(f"Error: ODE solve failed for segment {n} (initial shape)")
             sys.exit(1)
        current_y0_init = sol_n[:, -1] # 保存末端状态作为下一段的初始
        if n == 0:
             sol0_list.append(sol_n)
        elif sol_n.shape[1] > 1:
             sol0_list.append(sol_n[:, 1:]) # 移除重复点
        # else: 长度为0或只有一点，不添加

    if not sol0_list:
        print("Error: No valid ODE solution segments found for initial shape.")
        sys.exit(1)
    sol0 = np.concatenate(sol0_list, axis=1) # 拼接初始解
    print(f"Initial shape calculated. Sol0 shape: {sol0.shape}")

    my_base_pos_init = np.array([0, 0, 0.0])
    my_base_pos      = np.array([0, 0, 0.3]) # 稍微降低一点基座高度
    radius = my_sphere_radius

    # --- 创建 PyBullet 形状 ---
    print("--- Creating PyBullet Shapes ---")
    try:
        shape = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
        visualShapeId = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=body_color)
        dl_slider_1 = p.addUserDebugParameter("dl_seg2_c1", -0.01, 0.01, 0.0) # 添加滑块示例
        dl_slider_2 = p.addUserDebugParameter("dl_seg3_c3", -0.01, 0.01, 0.0) # 添加滑块示例
        visualShapeId_tip = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.02, 0.002, 0.001], rgbaColor=head_color)
        visualShapeId_tip_ = p.createVisualShape(p.GEOM_SPHERE, radius=radius+0.0025, rgbaColor=my_head_color)
    except p.error as e:
        print(f"Error creating shapes: {e}")
        sys.exit(1)

    # --- 创建 PyBullet 物体 ---
    print("--- Creating PyBullet Bodies ---")
    t = 0
    dt = 0.01 # 循环等待时间
    if sol0.shape[1] < 2:
        print("Error: Initial ODE solution has less than 2 points.")
        sys.exit(1)
    idx0 = np.linspace(0, sol0.shape[1] -1, my_number_of_sphere, dtype=int)
    positions0 = [(sol0[0, i], sol0[2, i], sol0[1, i]) for i in idx0] # 提取局部位置
    my_robot_bodies = []
    for i, pos in enumerate(positions0):
        try:
            body_id = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=shape,
                                          baseVisualShapeIndex=visualShapeId,
                                          basePosition=np.array(pos) + my_base_pos)
            my_robot_bodies.append(body_id)
        except p.error as e:
            print(f"Error creating body {i}: {e}")
            p.disconnect()
            sys.exit(1)

    # --- 创建末端物体 ---
    if len(positions0) >= 2 and len(my_robot_bodies) > 0:
        try:
            ori0, _ = calculate_orientation(positions0[-2], positions0[-1]) # 使用原始局部点计算局部方向

            # 将局部方向转到世界坐标系 (乘以基座方向，这里基座方向是 identity)
            # base_ori_q0 = p.getQuaternionFromEuler([0,0,0])
            # _, ori0_world = p.multiplyTransforms(my_base_pos, base_ori_q0, [0,0,0], ori0)
            ori0_world = ori0 # 基座初始方向为0，局部方向即世界方向

            last_pos_world = np.array(positions0[-1]) + my_base_pos

            tip_id1 = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=shape,
                                       baseVisualShapeIndex=visualShapeId_tip_,
                                       basePosition=last_pos_world, baseOrientation=ori0_world)
            my_robot_bodies.append(tip_id1)
            tip_id2 = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=shape,
                                       baseVisualShapeIndex=visualShapeId_tip,
                                       basePosition=last_pos_world + p.rotateVector(ori0_world, [-0.01, 0, 0]), baseOrientation=ori0_world)
            my_robot_bodies.append(tip_id2)
            tip_id3 = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=shape,
                                       baseVisualShapeIndex=visualShapeId_tip,
                                       basePosition=last_pos_world + p.rotateVector(ori0_world, [0.01,0,0]), baseOrientation=ori0_world)
            my_robot_bodies.append(tip_id3)
            print(f"Created {len(my_robot_bodies)} bodies in total.")
        except p.error as e:
            print(f"Error creating tip bodies: {e}")
            p.disconnect()
            sys.exit(1)
    else:
        print("Warning: Cannot create tip bodies due to insufficient initial points or body parts.")

    my_robot_line_ids = []
    print("--- Initialization Complete, Starting Simulation Loop ---")

    # --- 主循环 ---
    frame_count = 0
    last_print_time = time.time()
    while True:
        try:
            # --- 读取滑块值 ---
            dl_s2_c1 = p.readUserDebugParameter(dl_slider_1) # 读取滑块值
            dl_s3_c3 = p.readUserDebugParameter(dl_slider_2) # 读取滑块值

            # --- 定义目标绳长变化量 (现在由滑块控制部分) ---
            target_cable_changes = np.zeros(my_number_of_segment * num_cables)
            if my_number_of_segment >= 2:
                target_cable_changes[3] = dl_s2_c1 # 第2段第1根绳
                # target_cable_changes[4] = ... # 其他绳索可以保持0或添加更多滑块
                # target_cable_changes[5] = ...
            if my_number_of_segment >= 3:
                # target_cable_changes[6] = ...
                # target_cable_changes[7] = ...
                target_cable_changes[8] = dl_s3_c3 # 第3段第3根绳

            # --- 设定基座位姿 ---
            base_pos = np.array([0, 0, 0.3])
            base_ori_euler = np.array([0, 0, 0])

            # --- 分段计算新形态 ---
            my_ode._reset_y0() # 重置 y0 为初始状态 [0,0,0, I]
            sol = None
            current_y0 = np.copy(my_ode.y0) # 获取重置后的初始 y0

            for n in range(my_number_of_segment):
                # 1. 获取当前段 dl
                start_idx = n * num_cables
                end_idx = (n + 1) * num_cables
                dl_segment = target_cable_changes[start_idx:end_idx]

                # 2. 计算曲率 ux, uy
                ux, uy = calculate_curvatures_from_dl(dl_segment, cable_distance, L0_seg, num_cables)

                # 3. 转换为 ODE 动作
                action_ode_segment = convert_curvatures_to_ode_action(ux, uy, 0.0, cable_distance, L0_seg)

                # 4. 求解当前段
                my_ode.y0 = current_y0 # !!! 设置当前段的初始条件 !!!
                my_ode.updateAction(action_ode_segment)
                sol_n = my_ode.odeStepFull()

                # 检查求解是否成功
                if sol_n is None or sol_n.shape[1] == 0:
                    print(f"Warning: ODE solve failed for segment {n}. Skipping update for this frame.")
                    sol = None # 标记求解失败
                    break # 中断当前帧的计算

                # 5. 更新下一段的初始条件
                current_y0 = sol_n[:, -1]

                # 6. 拼接解
                if sol is None:
                    sol = np.copy(sol_n)
                elif sol_n.shape[1] > 1:
                    sol = np.concatenate((sol, sol_n[:, 1:]), axis=1)

            # --- 后续更新 PyBullet 对象 ---
            if sol is not None and sol.shape[1] > 0: # 仅在 ODE 求解成功时更新
                base_ori = p.getQuaternionFromEuler(base_ori_euler)
                my_base_pos, my_base_ori = base_pos, base_ori # 更新当前的基座位姿
                my_base_pos_offset = np.array([0,0,0])

                if sol.shape[1] < 2: # 检查点数是否足够
                     print(f"Warning: ODE solution has < 2 points ({sol.shape[1]}). Skipping PyBullet update.")
                     continue

                idx = np.linspace(0, sol.shape[1] -1, my_number_of_sphere, dtype=int)
                positions = [(sol[0, i], sol[2, i], sol[1, i]) for i in idx] # 提取局部位置
                my_robot_line_ids = [] # 清除旧线 (如果绘制了的话)

                pose_in_word_frame = []
                min_len = min(len(positions), len(my_robot_bodies))
                # 更新球体
                for i in range(min_len - 3): # 只更新身体部分，末端单独更新
                    pos_local = positions[i]
                    pos_world, orin_world = p.multiplyTransforms(my_base_pos + my_base_pos_offset,
                                                                 my_base_ori,
                                                                 pos_local,
                                                                 [0, 0, 0, 1])
                    pose_in_word_frame.append(np.concatenate((np.array(pos_world),np.array(orin_world))))
                    p.resetBasePositionAndOrientation(my_robot_bodies[i], pos_world, orin_world)

                # 更新末端
                if len(positions) >= 3 and len(my_robot_bodies) >= 3:
                    # 计算世界坐标的头部位置
                    head_pos_world = np.array(p.multiplyTransforms(my_base_pos + my_base_pos_offset, my_base_ori, positions[-1], [0,0,0,1])[0])
                    # 计算世界坐标的末端姿态
                    _tip_ori_local, tip_ori_euler = calculate_orientation(positions[-3], positions[-1])
                    _, tip_ori_world = p.multiplyTransforms(my_base_pos + my_base_pos_offset, my_base_ori, [0,0,0], _tip_ori_local) # 确保参考点是基座+偏移

                    # 计算世界坐标的夹爪位置
                    gripper_pos1_local_rotated = rotate_point_3d([0.02, -my_grasp_width, 0], tip_ori_euler)
                    gripper_pos2_local_rotated = rotate_point_3d([0.02, my_grasp_width, 0], tip_ori_euler)
                    gripper_pos1_world, _ = p.multiplyTransforms(head_pos_world, tip_ori_world, gripper_pos1_local_rotated, [0,0,0,1])
                    gripper_pos2_world, _ = p.multiplyTransforms(head_pos_world, tip_ori_world, gripper_pos2_local_rotated, [0,0,0,1])

                    # 更新 PyBullet 对象
                    p.resetBasePositionAndOrientation(my_robot_bodies[-3], head_pos_world, tip_ori_world)
                    # my_head_pose = [head_pos_world, tip_ori_world] # 不再需要 base_ori
                    p.resetBasePositionAndOrientation(my_robot_bodies[-2], gripper_pos1_world, tip_ori_world)
                    p.resetBasePositionAndOrientation(my_robot_bodies[-1], gripper_pos2_world, tip_ori_world)
                else:
                     print(f"Warning: Not enough points ({len(positions)}) or bodies ({len(my_robot_bodies)}) to update tip.")

            # --- 仿真步进和暂停 ---
            p.stepSimulation()
            time.sleep(dt)
            frame_count += 1

            # 打印帧率 (可选)
            current_time = time.time()
            if current_time - last_print_time >= 1.0:
                # print(f"FPS: {frame_count / (current_time - last_print_time):.2f}")
                frame_count = 0
                last_print_time = current_time

        except KeyboardInterrupt:
            print("Simulation interrupted by user.")
            break
        except p.error as e:
            print(f"PyBullet error occurred: {e}")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            import traceback
            traceback.print_exc()
            break

    print("Disconnecting from PyBullet.")
    p.disconnect()