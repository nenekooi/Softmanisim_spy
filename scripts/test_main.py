# -*- coding: utf-8 -*-
import sys
import os

# 动态添加 SoftManiSim 文件夹到 sys.path
# (假设此脚本位于 SoftManiSim/scripts/ 目录下)
softmanisim_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if softmanisim_path not in sys.path:
    sys.path.append(softmanisim_path)
    print(f"[调试] 已添加路径: {softmanisim_path}") # 打印添加的路径，方便调试

# --- 检查 visualizer 模块导入 ---
try:
    # 尝试直接导入，如果 visualizer 在 SoftManiSim 根目录
    from visualizer.visualizer import ODE
    print("[调试] 成功从 'visualizer.visualizer' 导入 ODE")
except ImportError:
    print("[错误] 无法直接从 'visualizer.visualizer' 导入 ODE。")
    print("[调试] sys.path:", sys.path)
    # 可以尝试其他可能的导入路径，但这通常表明项目结构或 PYTHONPATH 设置有问题
    # 例如: from SoftManiSim.visualizer.visualizer import ODE # 如果 SoftManiSim 在 PYTHONPATH 中
    # 如果还不行，需要用户检查环境和文件结构
    sys.exit("请检查 visualizer 模块的位置和 Python 路径设置。")


import numpy as np
import pybullet as p
import pybullet_data
import time
import math
from pprint import pprint
import pandas as pd # <<< 新增导入 pandas

# --- 辅助函数 (来自原代码，保持不变) ---
def calculate_orientation(point1, point2):
    # ... (代码同上) ...
    """根据两点计算方向 (四元数和欧拉角)"""
    diff = np.array(point2) - np.array(point1)
    if np.linalg.norm(diff) < 1e-6:
        return p.getQuaternionFromEuler([0,0,0]), [0,0,0]
    if np.linalg.norm(diff[:2]) < 1e-6:
        yaw = 0
    else:
        yaw = math.atan2(diff[1], diff[0])
    pitch = math.atan2(-diff[2], math.sqrt(diff[0] ** 2 + diff[1] ** 2))
    roll = 0
    return p.getQuaternionFromEuler([roll, pitch, yaw]), [roll, pitch, yaw]

def rotate_point_3d(point, rotation_angles):
    # ... (代码同上) ...
    """根据欧拉角旋转一个3D点"""
    rx, ry, rz = rotation_angles
    if any(math.isnan(angle) or math.isinf(angle) for angle in rotation_angles):
        print(f"Warning: Invalid rotation angles {rotation_angles}. Returning original point.")
        return tuple(point)
    rx = rx % (2 * math.pi)
    ry = ry % (2 * math.pi)
    rz = rz % (2 * math.pi)
    cos_rx, sin_rx = np.cos(rx), np.sin(rx)
    cos_ry, sin_ry = np.cos(ry), np.sin(ry)
    cos_rz, sin_rz = np.cos(rz), np.sin(rz)
    rotation_x = np.array([[1, 0, 0], [0, cos_rx, -sin_rx], [0, sin_rx, cos_rx]])
    rotation_y = np.array([[cos_ry, 0, sin_ry], [0, 1, 0], [-sin_ry, 0, cos_ry]])
    rotation_z = np.array([[cos_rz, -sin_rz, 0], [sin_rz, cos_rz, 0], [0, 0, 1]])
    rotation_matrix = np.dot(rotation_z, np.dot(rotation_y, rotation_x))
    rotated_point = np.dot(rotation_matrix, point)
    return tuple(rotated_point)

# --- 绳长变化量 -> 曲率 的转换函数 (来自原代码，可能需要用户根据实际模型修改) ---
def calculate_curvatures_from_dl(dl_segment, d, L0_seg, num_cables=3):
    """
    根据绳长变化量计算曲率 ux, uy。
    !!! 重要: 这是一个基于简化假设的示例函数 !!!
    !!! 你需要根据你的机器人几何结构和论文模型替换这里的计算逻辑 !!!
    """
    # ... (代码同上，但注意输入是 dl，不是 L) ...
    ux = 0.0
    uy = 0.0
    if abs(d) < 1e-9 or abs(L0_seg) < 1e-9:
        return 0.0, 0.0
    if len(dl_segment) != num_cables:
        print(f"警告: 输入的 dl_segment 长度 {len(dl_segment)} 与 num_cables {num_cables} 不符")
        return ux, uy

    # --- 在这里替换成你论文或推导的精确公式 ---
    if num_cables == 3:
        dl1, dl2, dl3 = dl_segment[0], dl_segment[1], dl_segment[2]
        # 简化比例因子 (可能需要标定或精确推导!)
        # gain = 1.0 / (d * L0_seg) # 简化近似 gain (原代码注释掉的)
        # 使用论文或物理模型推导出的关系，或者标定得到的系数
        # 这里的示例关系可能不适用于你的具体机器人
        gain = 1.0 / (d) # !!! 示例: 更简化的增益, 忽略 L0_seg 的影响 !!! 请务必检查这里的合理性

        # uy (绕y轴弯曲) 主要由绳1相对于绳2和绳3的平均长度变化引起
        uy = gain * (dl1 - (dl2 + dl3) / 2.0)
        # ux (绕x轴弯曲) 主要由绳2和绳3的长度差异引起 (注意符号和坐标系!)
        ux = gain * (math.sqrt(3.0) / 2.0) * (dl3 - dl2) # 乘以 sqrt(3)/2 是几何关系
    else:
        print(f"错误: 当前示例仅为3根绳索实现了曲率计算。")
        # 返回0或抛出错误
        return 0.0, 0.0
    # --- 替换结束 ---

    # 可选限制
    max_curvature = 50.0 # 示例限制值，根据需要调整
    ux = np.clip(ux, -max_curvature, max_curvature)
    uy = np.clip(uy, -max_curvature, max_curvature)

    return ux, uy

# --- 曲率 -> ODE抽象动作 的转换函数 (来自原代码) ---
def convert_curvatures_to_ode_action(ux, uy, length_change, d, L0_seg):
    """将计算出的曲率转换为原始ODE.updateAction期望的抽象action格式。"""
    # 根据 ODE.updateAction 的反向逻辑:
    # uy = action[1] / (l * d)  => action[1] = uy * l * d
    # ux = action[2] / -(l * d) => action[2] = -ux * l * d
    l = L0_seg # 假设长度不变
    action_ode = np.zeros(3)
    action_ode[0] = length_change # 长度变化量 (这里我们主要关心弯曲，设为0?)
                                  # 或者如果也需要模拟整体伸缩，需要提供总长度变化
    action_ode[1] = uy * l * d
    action_ode[2] = -ux * l * d
    return action_ode

# --- 主程序 ---
if __name__ == "__main__":

    # --- 数据文件路径和参数 (!!! 请仔细检查和修改 !!!) ---
    print("--- Setting Parameters & Loading Data ---")
    # 1. 数据文件路径
    DATA_FILE_PATH = 'c:/Users/11647/SoftManiSim/code/Modeling/Processed_Data3w_20250318.xlsx' # <<< 确认路径
    SHEET_NAME = 'Sheet1' # <<< 确认 Sheet 名称

    # 2. 机器人物理参数 (!!! 必须与你的机器人和数据匹配 !!!)
    num_cables = 3       # 绳索数量
    cable_distance = 7.5e-3 # 绳索到中心线的距离 d (单位: 米)
    initial_length = 0.6   # 机器人总初始/参考长度 L0 (单位: 米)
                           # 这个 L0 应该对应 cblen 数据接近初始值时的状态
                           # 注意：这里的 L0 用于计算 L0_seg，进而可能影响曲率计算和ODE转换
                           # 需要保证它与 calculate_curvatures_from_dl 函数中的 L0_seg 一致

    number_of_segment = 1 # !!! 重要：当前代码基于单段机器人 !!!
    if number_of_segment <= 0:
        print("错误: number_of_segment 必须是正数。"); sys.exit(1)
    L0_seg = initial_length / number_of_segment # 每段的参考长度 = 总长 (因为只有一段)
    print(f"机器人参数: 段数={number_of_segment}, 绳数={num_cables}, 总参考长度L0={initial_length:.4f}m, 每段参考长度L0_seg={L0_seg:.4f}m, 绳索半径d={cable_distance:.4f}m")

    # 3. 可视化参数
    body_color = [1, 0.0, 0.0, 1] # 红色身体
    head_color = [0.0, 0.0, 0.75, 1] # 蓝色头部
    body_sphere_radius = 0.02
    number_of_sphere = 30 # 可视化用的球体数量
    my_sphere_radius = body_sphere_radius
    my_number_of_sphere = number_of_sphere
    my_head_color = head_color

    # --- 加载并处理绳长数据 ---
    print(f"[信息] 正在加载绝对绳长数据文件: {DATA_FILE_PATH}")
    if not os.path.exists(DATA_FILE_PATH):
        print(f"[错误] 数据文件未找到: {DATA_FILE_PATH}"); sys.exit(1)

    try:
        # 读取 Excel 文件
        df_lengths = pd.read_excel(DATA_FILE_PATH, sheet_name=SHEET_NAME, engine='openpyxl')
        print(f"[成功] 已从 Excel 文件 '{os.path.basename(DATA_FILE_PATH)}' (Sheet: {SHEET_NAME}) 加载数据。")

        # 检查必需的列是否存在
        required_cols = ['cblen1', 'cblen2', 'cblen3']
        if not all(col in df_lengths.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in df_lengths.columns]
            raise ValueError(f"错误: 文件缺少必需的列: {missing_cols}")

        # 提取绝对绳长数据 (单位: mm，根据用户描述的数值推断)
        absolute_lengths_mm = df_lengths[required_cols].values
        print(f"[信息] 成功提取 {len(absolute_lengths_mm)} 行绝对绳长数据。")

        # --- 计算绳长变化量 (dl) ---
        # 假设第一行是初始/参考绳长 L0
        if len(absolute_lengths_mm) > 0:
            L0_cables_mm = absolute_lengths_mm[0] # 获取第一行的 L0 (mm)
            print(f"[假设] 使用数据文件的第一行作为初始绝对绳长 L0 (mm): {L0_cables_mm}")

            # 将所有绝对绳长转换为米
            absolute_lengths_m = absolute_lengths_mm / 1000.0
            # 将初始绳长也转换为米
            L0_cables_m = L0_cables_mm / 1000.0

            # 计算绳长变化量 dl (单位: 米)
            # dl(t) = L(t) - L0
            dl_sequence_m = absolute_lengths_m - L0_cables_m
            print(f"[信息] 已计算得到 {len(dl_sequence_m)} 行绳长变化量 dl (单位: m)。")
            # 打印前几行 dl 检查
            print("[调试] 前5行的 dl (m):")
            print(dl_sequence_m[:5])

        else:
            raise ValueError("错误: 数据文件为空，无法计算绳长变化量。")

    except FileNotFoundError:
        print(f"[错误] 文件未找到，请检查路径: {DATA_FILE_PATH}"); sys.exit(1)
    except ImportError:
        print("[错误] 读取 Excel 文件需要 'openpyxl' 库。请运行 'pip install openpyxl' 安装。"); sys.exit(1)
    except ValueError as e:
         print(f"[错误] 处理数据时出错: {e}"); sys.exit(1)
    except Exception as e:
        print(f"[错误] 加载或处理数据时发生意外错误: {e}"); sys.exit(1)

    # --- PyBullet 初始化 (与原代码类似) ---
    print("--- Initializing PyBullet ---")
    simulationStepTime = 0.01
    try:
        physicsClientId = p.connect(p.GUI)
        if physicsClientId < 0: raise ConnectionError("Failed to connect.")
        print(f"已连接 PyBullet, Client ID: {physicsClientId}")
    except Exception as e:
        print(f"连接 PyBullet 出错: {e}"); sys.exit(1)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(simulationStepTime)
    try:
        planeId = p.loadURDF("plane.urdf")
        print(f"已加载 plane.urdf, ID: {planeId}")
    except p.error as e:
        print(f"加载 plane.urdf 出错: {e}"); sys.exit(1)

    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
    p.resetDebugVisualizerCamera(cameraDistance=0.7, cameraYaw=90, cameraPitch=-30, cameraTargetPosition=[0, 0.2, 0.1])

    # --- 初始化 ODE 对象 (与原代码类似) ---
    print("--- Initializing ODE Object ---")
    my_ode = ODE()
    my_ode.l0 = initial_length # !!! 使用上面定义的总初始长度 !!!
    my_ode.d = cable_distance   # !!! 使用上面定义的绳索半径 !!!
    print(f"ODE 已初始化 L0={my_ode.l0:.4f}m, d={my_ode.d:.4f}m")

    # --- 计算并显示初始形态 (与原代码类似) ---
    print("--- Calculating Initial Shape ---")
    act0_segment = np.zeros(3) # 初始 action (对应 dl=[0,0,0])
    my_ode._reset_y0() # 确保从标准初始状态开始
    my_ode.updateAction(act0_segment)
    sol0 = my_ode.odeStepFull()
    if sol0 is None or sol0.shape[1] < 2: # 检查初始解是否有效
        print("错误: 初始 ODE 求解失败或点数不足。")
        p.disconnect(); sys.exit(1)
    print(f"初始形状计算完成。 Sol0 shape: {sol0.shape}")

    my_base_pos = np.array([0, 0, 0.3]) # 世界坐标基座位置
    radius = my_sphere_radius

    # --- 创建 PyBullet 形状 (移除滑块) ---
    print("--- Creating PyBullet Shapes (No Sliders) ---")
    try:
        shape = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
        visualShapeId = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=body_color)
        # --- 不再需要创建滑块 ---
        # dl_sliders = []
        # dl_sliders.append(p.addUserDebugParameter("dl_cable_1", -0.01, 0.01, 0.0))
        # ...
        # 末端的可视形状
        visualShapeId_tip = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.02, 0.002, 0.001], rgbaColor=head_color)
        visualShapeId_tip_ = p.createVisualShape(p.GEOM_SPHERE, radius=radius+0.0025, rgbaColor=my_head_color)
    except p.error as e:
        print(f"创建形状出错: {e}"); p.disconnect(); sys.exit(1)

    # --- 创建 PyBullet 物体 (与原代码类似) ---
    print("--- Creating PyBullet Bodies ---")
    t = 0
    dt = 0.01 # 注意：这里的 dt 与 simulationStepTime 不同步，仅用于 time.sleep
    if sol0.shape[1] < 2: print("错误: 初始 ODE 解的点数少于2。"); p.disconnect(); sys.exit(1)
    # 从初始解采样点
    idx0 = np.linspace(0, sol0.shape[1] - 1, my_number_of_sphere, dtype=int)
    positions0 = [(sol0[0, i], sol0[2, i], sol0[1, i]) for i in idx0] # 提取局部位置 (注意 YZ 交换)
    my_robot_bodies = []
    # 创建身体球体
    for i, pos in enumerate(positions0):
        try:
            current_pos_world = np.array(pos) + my_base_pos
            body_id = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=-1, # 通常不希望身体部分有碰撞
                                         baseVisualShapeIndex=visualShapeId,
                                         basePosition=current_pos_world)
            my_robot_bodies.append(body_id)
        except p.error as e:
            print(f"创建身体 {i} 出错: {e}"); p.disconnect(); sys.exit(1)

    # 创建末端物体 (头部+夹爪指示器)
    num_initial_bodies = len(my_robot_bodies)
    if len(positions0) >= 3 and num_initial_bodies >= 1: # 需要至少3个点算姿态
        try:
            ori0_local, _ = calculate_orientation(positions0[-3], positions0[-1]) # 使用倒数第三和最后一个点
            ori0_world = ori0_local # 初始基座姿态是 identity
            last_pos_world = np.array(positions0[-1]) + my_base_pos

            # 头部
            tip_id1 = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=-1,
                                        baseVisualShapeIndex=visualShapeId_tip_,
                                        basePosition=last_pos_world, baseOrientation=ori0_world)
            my_robot_bodies.append(tip_id1)
            # 夹爪指示器1
            offset1_local = [0.01, 0, 0] # 在尖端坐标系下的偏移
            pos1, _ = p.multiplyTransforms(last_pos_world, ori0_world, offset1_local, [0,0,0,1])
            tip_id2 = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=-1,
                                        baseVisualShapeIndex=visualShapeId_tip,
                                        basePosition=pos1, baseOrientation=ori0_world)
            my_robot_bodies.append(tip_id2)
            # 夹爪指示器2
            offset2_local = [-0.01, 0, 0] # 在尖端坐标系下的偏移
            pos2, _ = p.multiplyTransforms(last_pos_world, ori0_world, offset2_local, [0,0,0,1])
            tip_id3 = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1,
                                        baseVisualShapeIndex=visualShapeId_tip,
                                        basePosition=pos2, baseOrientation=ori0_world)
            my_robot_bodies.append(tip_id3)
            print(f"总共创建了 {len(my_robot_bodies)} 个物体 (包括末端)。")
        except p.error as e:
            print(f"创建末端物体出错: {e}"); p.disconnect(); sys.exit(1)
        except IndexError:
             print("错误: 初始点不足以计算末端姿态。"); p.disconnect(); sys.exit(1)
    else:
        print("警告: 无法创建末端物体，因为初始点或身体部件不足。")

    print("--- 初始化完成, 开始主仿真循环 ---")

    # --- 主循环 (使用 dl 数据驱动) ---
    play_mode = 'loop' # 播放模式: 'once' 或 'loop'
    current_row_index = 0
    num_data_rows = len(dl_sequence_m)
    target_cable_changes = np.zeros(num_cables) # 用于存储当前帧的 dl

    if num_data_rows == 0:
        print("[错误] 没有可用的绳长变化量数据来驱动仿真。")
        p.disconnect()
        sys.exit(1)

    print(f"将按顺序应用 {num_data_rows} 组绳长变化量数据。播放模式: {play_mode}")

    frame_count = 0
    last_print_time = time.time()

    try:
        while True:
            # --- 获取当前帧的绳长变化量 dl ---
            if current_row_index >= num_data_rows:
                if play_mode == 'loop':
                    # print("[信息] 数据播放完毕，重新开始循环。")
                    current_row_index = 0 # 重置索引
                else:
                    print("[信息] 数据播放完毕。")
                    break # 结束循环

            # 从计算好的序列中获取当前行的 dl (单位: 米)
            dl_segment = dl_sequence_m[current_row_index]
            target_cable_changes = dl_segment # 更新目标变化量

            # --- 设定基座位姿 (可以保持不变或根据需要修改) ---
            base_pos = np.array([0, 0, 0.3])
            base_ori_euler = np.array([0, 0, 0]) # (Roll, Pitch, Yaw)

            # --- 计算新形态 (基于当前 dl_segment) ---
            my_ode._reset_y0() # 每次都从基座状态开始重新积分

            # 1. 计算对应的曲率 ux, uy (使用脚本中定义的参数)
            ux, uy = calculate_curvatures_from_dl(dl_segment, cable_distance, L0_seg, num_cables)

            # 2. 将曲率转换为原始 ODE 的抽象 action
            # 注意: convert_curvatures_to_ode_action 第三个参数是整体长度变化量
            # 如果只关心弯曲，可以设为 0.0
            action_ode_segment = convert_curvatures_to_ode_action(ux, uy, 0.0, cable_distance, L0_seg)

            # 3. 调用原始 ODE 求解得到当前形状
            my_ode.updateAction(action_ode_segment)
            sol = my_ode.odeStepFull() # sol 包含形状信息 (12, k)

            # 打印调试信息 (可选，频率可以降低)
            if current_row_index % 200 == 0: # 每 200 帧打印一次
                print(f"帧 {current_row_index}: dl(m)={dl_segment}, -> ux={ux:.3f}, uy={uy:.3f} -> ODE Action={action_ode_segment}")
                if sol is None: print("  ODE 求解失败!")


            # --- 更新 PyBullet 可视化对象 ---
            if sol is not None and sol.shape[1] >= 3: # 检查解是否有效且至少有3个点(算姿态需要)
                base_ori = p.getQuaternionFromEuler(base_ori_euler)
                my_base_pos, my_base_ori = base_pos, base_ori # 当前基座位姿
                my_base_pos_offset = np.array([0,0,0]) # 简化偏移

                # 从解中采样点
                idx = np.linspace(0, sol.shape[1] -1, my_number_of_sphere, dtype=int)
                positions = [(sol[0, i], sol[2, i], sol[1, i]) for i in idx] # 提取局部位置 (YZ交换)

                # 确保更新时不越界
                num_bodies_total = len(my_robot_bodies)
                num_tip_bodies = 3 # 头部+两个夹爪指示器
                num_body_spheres = num_bodies_total - num_tip_bodies
                num_points_available = len(positions)
                num_spheres_to_update = min(num_body_spheres, num_points_available) # 要更新的身体球体数量

                # 更新球体 (身体部分)
                for i in range(num_spheres_to_update):
                    pos_local = positions[i]
                    # 将局部坐标转换为世界坐标
                    pos_world, orin_world = p.multiplyTransforms(my_base_pos + my_base_pos_offset,
                                                                 my_base_ori,
                                                                 pos_local,
                                                                 [0, 0, 0, 1]) # 局部姿态设为 identity
                    try:
                        # 重置每个球体的位置和姿态
                        p.resetBasePositionAndOrientation(my_robot_bodies[i], pos_world, orin_world)
                    except p.error: pass # 忽略可能的错误

                # 更新末端 (头部 + 夹爪指示器)
                if num_points_available >= 3 and num_bodies_total >= num_tip_bodies:
                    try:
                        # 计算世界坐标的头部位置 (使用最后一个采样点)
                        head_pos_world = np.array(p.multiplyTransforms(my_base_pos + my_base_pos_offset, my_base_ori, positions[-1], [0,0,0,1])[0])
                        # 计算世界坐标的末端姿态 (基于最后两/三个点)
                        _tip_ori_local, tip_ori_euler = calculate_orientation(positions[-3], positions[-1]) # 需要至少3个点
                        _, tip_ori_world = p.multiplyTransforms(my_base_pos + my_base_pos_offset, my_base_ori, [0,0,0], _tip_ori_local)

                        # 更新头部 (通常是 my_robot_bodies[-3])
                        p.resetBasePositionAndOrientation(my_robot_bodies[-3], head_pos_world, tip_ori_world)

                        # 更新夹爪指示器1 (通常是 my_robot_bodies[-2])
                        gripper_offset1_local = [0.01, 0, 0] # 相对于头部的局部偏移
                        gripper_pos1_world, _ = p.multiplyTransforms(head_pos_world, tip_ori_world, gripper_offset1_local, [0,0,0,1])
                        p.resetBasePositionAndOrientation(my_robot_bodies[-2], gripper_pos1_world, tip_ori_world)

                        # 更新夹爪指示器2 (通常是 my_robot_bodies[-1])
                        gripper_offset2_local = [-0.01, 0, 0] # 相对于头部的局部偏移
                        gripper_pos2_world, _ = p.multiplyTransforms(head_pos_world, tip_ori_world, gripper_offset2_local, [0,0,0,1])
                        p.resetBasePositionAndOrientation(my_robot_bodies[-1], gripper_pos2_world, tip_ori_world)

                    except p.error: pass # 忽略可能的错误
                    except IndexError: pass # 忽略可能的索引错误
                # else:
                #     if frame_count % 60 == 0: # 降低打印频率
                #          print(f"警告: 点数 ({num_points_available}) 或物体数 ({num_bodies_total}) 不足以更新末端。")

            elif sol is not None and sol.shape[1] < 3:
                 if frame_count % 60 == 0: # 降低打印频率
                     print(f"警告: 第 {current_row_index} 帧 ODE 解的点数 ({sol.shape[1]}) 少于3, 无法更新可视化。")
            elif sol is None: # sol is None
                 if frame_count % 60 == 0: # 降低打印频率
                     print(f"警告: 第 {current_row_index} 帧 ODE 求解失败, 无法更新可视化。")

            # --- 仿真步进和延时 ---
            p.stepSimulation()
            time.sleep(simulationStepTime) # 使用与仿真步长匹配的延时
            frame_count += 1
            current_time = time.time()
            # if current_time - last_print_time >= 1.0:
            #     # print(f"FPS: {frame_count / (current_time - last_print_time):.2f}")
            #     frame_count = 0
            #     last_print_time = current_time

            # 移动到数据文件的下一行
            current_row_index += 1

    except KeyboardInterrupt:
        print("\n[信息] 检测到键盘中断 (Ctrl+C)，正在停止仿真...")
    except p.error as e:
        print(f"\n[错误] PyBullet 发生错误 (图形窗口是否已关闭?): {e}")
    except Exception as e:
        print(f"\n[错误] 发生意外错误: {e}")
        import traceback
        traceback.print_exc() # 打印详细错误追踪信息
    finally:
        # 确保在退出时断开 PyBullet 连接
        print("[信息] 断开 PyBullet 连接。")
        if p.isConnected():
            try:
                p.disconnect()
            except p.error:
                pass # 可能已经因为错误断开了

    print("--- 仿真结束 ---")