# -*- coding: utf-8 -*-
# ==============================================================================
# 脚本功能:
# 读取包含绝对绳长数据的文件 (CSV 或 Excel)，
# 计算绳长变化量 (Delta L)，并使用基于简化模型的曲率将其转换为 ODE 输入，
# 驱动一个单段连续体机器人的 ODE 仿真，
# 并将结果可视化在 PyBullet 中，机器人初始状态为竖直向下。
# ==============================================================================

# 导入所需的库
import sys
import os
import numpy as np
import pybullet as p
import pybullet_data
import time
import math
from pprint import pprint
import pandas as pd
softmanisim_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if softmanisim_path not in sys.path:
    sys.path.append(softmanisim_path)

from visualizer.visualizer import ODE

def calculate_orientation(point1, point2):
    """根据两点计算方向 (四元数和欧拉角)"""
    diff = np.array(point2) - np.array(point1)
    if np.linalg.norm(diff) < 1e-6:
        return p.getQuaternionFromEuler([0,0,0]), [0,0,0] # 默认姿态
    # 计算 Yaw (绕 Z 轴) - 注意是在点的局部坐标系，可能需要调整
    if np.linalg.norm(diff[:2]) < 1e-6: # 垂直线
        yaw = 0
    else:
        yaw = math.atan2(diff[1], diff[0])
    # 计算 Pitch (绕 Y 轴)
    pitch = math.atan2(-diff[2], math.sqrt(diff[0]**2 + diff[1]**2))
    # 假设 Roll (绕 X 轴) 为 0
    roll = 0
    return p.getQuaternionFromEuler([roll, pitch, yaw]), [roll, pitch, yaw]

def rotate_point_3d(point, rotation_angles):
    """(此函数在当前代码中似乎未使用，但保留以防万一)"""
    """根据欧拉角旋转一个3D点"""
    rx, ry, rz = rotation_angles
    if any(math.isnan(angle) or math.isinf(angle) for angle in rotation_angles):
        print(f"警告: 无效的旋转角度 {rotation_angles}。返回原始点。")
        return tuple(point)
    # 确保角度在合理范围或由三角函数处理
    cos_rx, sin_rx = np.cos(rx), np.sin(rx)
    cos_ry, sin_ry = np.cos(ry), np.sin(ry)
    cos_rz, sin_rz = np.cos(rz), np.sin(rz)
    # 旋转矩阵定义可能需要根据坐标系约定调整
    rotation_x = np.array([[1, 0, 0], [0, cos_rx, -sin_rx], [0, sin_rx, cos_rx]])
    rotation_y = np.array([[cos_ry, 0, sin_ry], [0, 1, 0], [-sin_ry, 0, cos_ry]])
    rotation_z = np.array([[cos_rz, -sin_rz, 0], [sin_rz, cos_rz, 0], [0, 0, 1]])
    rotation_matrix = np.dot(rotation_z, np.dot(rotation_y, rotation_x)) # ZYX 顺序
    rotated_point = np.dot(rotation_matrix, point)
    return tuple(rotated_point)

# --- 绳长变化量 -> 曲率 的转换函数 (需要根据实际模型验证或修改) ---
# def calculate_curvatures_from_dl(dl_segment, d, L0_seg, num_cables=3):
#     """
#     根据绳长变化量计算曲率 ux, uy。
#     重要: 这是一个基于简化假设的示例函数
#     Args:
#         dl_segment (np.array): 当前段的绳长变化量 [dl1, dl2, ..., dln] (单位: 米)。
#         d (float): 绳索到中心线的距离 (半径) (单位: 米)。
#         L0_seg (float): 当前段的参考长度 (单位: 米)。
#         num_cables (int): 绳索数量。
#     Returns:
#         tuple: (ux, uy) 计算得到的曲率 (单位: 1/米)。
#     """
#     ux = 0.0
#     uy = 0.0
#     # 基本检查
#     if abs(d) < 1e-9: # L0_seg 可能为0如果初始长度为0，但也应避免除零
#         print("警告: 绳索半径 d 接近于零，无法计算曲率。")
#         return 0.0, 0.0
#     if len(dl_segment) != num_cables:
#         print(f"警告: 输入的 dl_segment 长度 {len(dl_segment)} 与 num_cables {num_cables} 不符")
#         return ux, uy

#     # --- 在这里替换成你论文或推导的精确公式 ---
#     # 以下是针对 3 根对称绳索 (0, 120, 240度) 的简化示例关系
#     # 这个关系可能需要根据您的机器人横截面几何、材料属性等进行调整或标定
#     if num_cables == 3:
#         dl1, dl2, dl3 = dl_segment[0], dl_segment[1], dl_segment[2]

#         # 示例关系：曲率与绳长变化量成正比，与半径d成反比
#         # 这里忽略了 L0_seg 的影响，假设增益为 1/d
#         # !!! 这只是一个非常简化的例子，实际关系可能更复杂 !!!
#         gain = 1.0 / d

#         # uy (绕y轴弯曲 - pitch) 通常与顶/底绳索相对于侧面绳索的变化有关
#         # 假设绳1在+X方向? (需要明确定义绳索编号和坐标系)
#         # 如果绳1在+X, 绳2在(-0.5X, +sqrt(3)/2 Y), 绳3在(-0.5X, -sqrt(3)/2 Y)
#         # 这里的 uy 计算假设绳1是主要影响 Pitch 的因素
#         uy = gain * (dl1 - (dl2 + dl3) / 2.0) # 绕 Y 轴的曲率

#         # ux (绕x轴弯曲 - yaw in local frame?) 通常与两侧绳索的差异有关
#         # 这里的 ux 计算假设绳2和绳3的差异影响 Yaw? (需要确认局部坐标系定义)
#         ux = gain * (math.sqrt(3.0) / 2.0) * (dl3 - dl2) # 绕 X 轴的曲率
#     else:
#         print(f"错误: 当前示例仅为3根绳索实现了曲率计算。")
#         return 0.0, 0.0 # 对于其他绳索数量，返回零曲率
#     # --- 替换结束 ---

#     # 可选：限制曲率大小防止数值问题
#     max_curvature = 50.0 # 设定一个最大允许曲率值，根据实际情况调整
#     ux = np.clip(ux, -max_curvature, max_curvature)
#     uy = np.clip(uy, -max_curvature, max_curvature)

#     return ux, uy


def calculate_curvatures_from_dl(dl_segment, d, L0_seg, num_cables=3):#根据折纸机器人的硕士论文反推的从绳长到曲率
    ux = 0.0
    uy = 0.0
    if abs(d) < 1e-9:
        print("警告: 绳索半径 d 接近于零。")
        return ux, uy

    if num_cables == 3:
        dl1, dl2, dl3 = dl_segment[0], dl_segment[1], dl_segment[2]

        # --- 基于论文模型反解的精确计算 ---
        # 计算 uy (基于 dl1)
        uy = -dl1 / d

        # 计算 ux (基于 dl3 和 dl2 的差异)
        if abs(math.sqrt(3.0)) > 1e-9: # 避免除零 (虽然 sqrt(3) 不会是零)
             ux = (dl3 - dl2) / (d * math.sqrt(3.0))
        else:
             ux = 0.0 # 或者其他处理方式

    else:
        print(f"错误: 此处仅实现了3根绳索的反解。")
        return 0.0, 0.0
    # ... (曲率限制 np.clip) ...
    max_curvature = 50.0 # 示例限制值
    ux = np.clip(ux, -max_curvature, max_curvature)
    uy = np.clip(uy, -max_curvature, max_curvature)

    return ux, uy

# --- 曲率 -> ODE抽象动作 的转换函数 (与原代码匹配) ---
def convert_curvatures_to_ode_action(ux, uy, length_change, d, L0_seg):
    """将计算出的曲率转换为原始ODE.updateAction期望的抽象action格式。"""
    l = L0_seg # 使用每段的参考长度
    action_ode = np.zeros(3)
    action_ode[0] = length_change # 整体轴向伸缩量 (如果数据中没有，通常设为0)
    # 根据 visualizer.py 中 ODE 类的 updateAction 方法反推:
    # uy = self.action[1] / (self.l0 * self.d) => action[1] = uy * l * d
    # ux = self.action[2] /-(self.l0 * self.d) => action[2] = -ux * l * d
    action_ode[1] = uy * l * d
    action_ode[2] = -ux * l * d
    return action_ode

# --- 主程序 ---
if __name__ == "__main__":

    # 1. 数据文件路径
    DATA_FILE_PATH = 'c:/Users/11647/SoftManiSim/code/Modeling/Processed_Data3w_20250318.xlsx' # <<< 确认 Excel 文件路径
    SHEET_NAME = 'Sheet1' # <<< 确认 Excel 中的工作表名称

    # 2. 机器人物理参数 
    num_cables = 3           # 绳索数量
    cable_distance = 5e-3 # 绳索到中心线的距离 d (单位: 米)
    initial_length = 0.15   # 机器人总初始/参考长度 L0 (单位: 米)   

    number_of_segment = 1 # !!! 重要：当前代码和ODE模型基于单段 !!!
    if number_of_segment <= 0: print("错误: number_of_segment 必须是正数。"); sys.exit(1)
    L0_seg = initial_length / number_of_segment # 每段的参考长度 = 总长
    print(f"机器人参数: 段数={number_of_segment}, 绳数={num_cables}, 总参考长度L0={initial_length:.4f}m, 每段参考长度L0_seg={L0_seg:.4f}m, 绳索半径d={cable_distance:.4f}m")

    # 3. 可视化参数
    body_color = [1, 0.0, 0.0, 1]; head_color = [0.0, 0.0, 0.75, 1]
    body_sphere_radius = 0.02; number_of_sphere = 50
    my_sphere_radius = body_sphere_radius; my_number_of_sphere = number_of_sphere
    my_head_color = head_color

    # --- 加载并处理绳长数据 ---
    print(f"[信息] 正在加载绝对绳长数据文件: {DATA_FILE_PATH}")
    if not os.path.exists(DATA_FILE_PATH): print(f"[错误] 数据文件未找到: {DATA_FILE_PATH}"); sys.exit(1)
    try:
        # 读取 Excel 文件
        df_lengths = pd.read_excel(DATA_FILE_PATH, sheet_name=SHEET_NAME, engine='openpyxl')
        print(f"[成功] 已从 Excel 文件 '{os.path.basename(DATA_FILE_PATH)}' (Sheet: {SHEET_NAME}) 加载数据。")
        # 检查必需的列
        required_cols = ['cblen1', 'cblen2', 'cblen3']
        if not all(col in df_lengths.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in df_lengths.columns]
            raise ValueError(f"错误: 文件缺少必需的列: {missing_cols}")
        # 提取绝对绳长 (假定单位是 mm)
        absolute_lengths_mm = df_lengths[required_cols].values
        print(f"[信息] 成功提取 {len(absolute_lengths_mm)} 行绝对绳长数据。")

        # --- 计算绳长变化量 (dl) ---
        if len(absolute_lengths_mm) > 0:
            L0_cables_mm = absolute_lengths_mm[0] # 取第一行作为参考 L0 (mm)
            print(f"[假设] 使用第一行作为初始绝对绳长 L0 (mm): {L0_cables_mm}")
            # 转换单位为米 (m)
            absolute_lengths_m = absolute_lengths_mm / 1000.0
            L0_cables_m = L0_cables_mm / 1000.0
            # 计算 dl 序列 (m)
            dl_sequence_m = absolute_lengths_m - L0_cables_m
            print(f"[信息] 已计算得到 {len(dl_sequence_m)} 行绳长变化量 dl (单位: m)。")
            print("[调试] 前5行的 dl (m):"); print(dl_sequence_m[:5])
        else:
            raise ValueError("错误: 数据文件为空。")
    except Exception as e: print(f"[错误] 加载或处理数据时出错: {e}"); sys.exit(1)

    # --- PyBullet 初始化 (修改相机和添加参数检查) ---
    print("--- 初始化 PyBullet ---")
    simulationStepTime = 0.01
    try:
        physicsClientId = p.connect(p.GUI)
        if physicsClientId < 0: raise ConnectionError("未能连接到 PyBullet。")
        print(f"已连接 PyBullet, Client ID: {physicsClientId}")
    except Exception as e: print(f"连接 PyBullet 出错: {e}"); sys.exit(1)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(simulationStepTime)
    try:
        planeId = p.loadURDF("plane.urdf")
        print(f"已加载 plane.urdf, ID: {planeId}")
    except p.error as e: print(f"加载 plane.urdf 出错: {e}"); sys.exit(1)

    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
    # --- 修改相机视角 ---
    p.resetDebugVisualizerCamera(
        cameraDistance=0.8,      # 稍微拉远一点距离
        cameraYaw=45,           # 从侧前方看
        cameraPitch=-20,         # 稍微俯视的角度
        cameraTargetPosition=[0, 0, 0.3] # 观察目标点设在基座稍靠下的位置
    )
    print("[信息] 已设置相机视角以观察垂直机器人。")

    # --- 初始化 ODE 对象 ---
    print("--- 初始化 ODE 对象 ---")
    my_ode = ODE()
    my_ode.l0 = initial_length
    my_ode.d = cable_distance
    print(f"ODE 已初始化 L0={my_ode.l0:.4f}m, d={my_ode.d:.4f}m")

    # --- 计算并显示初始形态 (对应 dl=[0,0,0]) ---
    print("--- 计算初始形状 (对应 dl=0) ---")
    act0_segment = np.zeros(3) # 初始 action (dl=0)
    my_ode._reset_y0()         # 重置 ODE 内部状态
    my_ode.updateAction(act0_segment)
    sol0 = my_ode.odeStepFull() # 求解初始形状
    if sol0 is None or sol0.shape[1] < 2:
        print("错误: 初始 ODE 求解失败或点数不足。"); p.disconnect(); sys.exit(1)
    print(f"初始形状计算完成。Sol0 shape: {sol0.shape}")

    # --- 设置基座位置和姿态 ---
    base_pos = np.array([0, 0, 0.6]) # !!! 将基座 Z 坐标抬高到 0.6m !!!
    # !!! 设置基座姿态: 绕 X 轴旋转 -90 度使其竖直向下 !!!
    base_ori_euler = np.array([-math.pi / 2.0, 0, 0]) # [Roll, Pitch, Yaw]
    base_ori = p.getQuaternionFromEuler(base_ori_euler) # 转换为四元数
    print(f"[设置] 机器人基座世界坐标: {base_pos}")
    print(f"[设置] 机器人基座世界姿态 (Euler): {base_ori_euler}")

    radius = my_sphere_radius

    # --- 创建 PyBullet 形状 ---
    print("--- 创建 PyBullet 形状 ---")
    try:
        shape = p.createCollisionShape(p.GEOM_SPHERE, radius=radius) # 复用球体形状
        visualShapeId = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=body_color)
        visualShapeId_tip_body = p.createVisualShape(p.GEOM_SPHERE, radius=radius+0.0025, rgbaColor=my_head_color) # 末端球体
        visualShapeId_tip_gripper = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.002, 0.01, 0.002], rgbaColor=head_color) # 细长方体代表夹爪
    except p.error as e: print(f"创建形状出错: {e}"); p.disconnect(); sys.exit(1)

    # --- 创建 PyBullet 物体 (基于初始形态 sol0 和设置好的基座位姿) ---
    print("--- 创建 PyBullet 物体 (基于初始形态和竖直姿态) ---")
    t = 0
    dt = 0.01 # 可视化更新间隔
    if sol0.shape[1] < 2: print("错误: 初始 ODE 解点数不足2。"); p.disconnect(); sys.exit(1)
    # 从初始解采样点
    idx0 = np.linspace(0, sol0.shape[1] - 1, my_number_of_sphere, dtype=int)
    positions0_local = [(sol0[0, i], sol0[2, i], sol0[1, i]) for i in idx0] # 提取局部位置 (YZ交换)
    my_robot_bodies = []
    # 创建身体球体
    for i, pos_local in enumerate(positions0_local):
        try:
            # !!! 使用设定的 base_pos 和 base_ori 将局部点转换到世界坐标 !!!
            pos_world, ori_world = p.multiplyTransforms(base_pos, base_ori,
                                                        pos_local, [0, 0, 0, 1]) # 假设局部姿态为单位四元数
            body_id = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=-1,
                                         baseVisualShapeIndex=visualShapeId,
                                         basePosition=pos_world, baseOrientation=ori_world) # 使用世界姿态
            my_robot_bodies.append(body_id)
        except p.error as e: print(f"创建身体 {i} 出错: {e}"); p.disconnect(); sys.exit(1)

    # 创建末端物体 (头部+夹爪指示器)
    num_initial_bodies = len(my_robot_bodies)
    if len(positions0_local) >= 3 and num_initial_bodies >= 1:
        try:
            # 计算末端的局部姿态
            ori_tip_local, _ = calculate_orientation(positions0_local[-3], positions0_local[-1])
            # 将末端局部位置和姿态转换到世界坐标
            pos_tip_world, ori_tip_world = p.multiplyTransforms(base_pos, base_ori,
                                                                positions0_local[-1], ori_tip_local)

            # 头部 (末端球体)
            tip_id_body = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=-1,
                                            baseVisualShapeIndex=visualShapeId_tip_body,
                                            basePosition=pos_tip_world, baseOrientation=ori_tip_world)
            my_robot_bodies.append(tip_id_body)
            # 夹爪指示器1
            gripper_offset1_local = [0, 0.01, 0] # 在末端局部坐标系下 Y 方向偏移
            pos1, ori1 = p.multiplyTransforms(pos_tip_world, ori_tip_world, gripper_offset1_local, [0,0,0,1])
            tip_id_grip1 = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=-1,
                                             baseVisualShapeIndex=visualShapeId_tip_gripper,
                                             basePosition=pos1, baseOrientation=ori_tip_world)
            my_robot_bodies.append(tip_id_grip1)
            # 夹爪指示器2
            gripper_offset2_local = [0, -0.01, 0] # 在末端局部坐标系下 负Y 方向偏移
            pos2, ori2 = p.multiplyTransforms(pos_tip_world, ori_tip_world, gripper_offset2_local, [0,0,0,1])
            tip_id_grip2 = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1,
                                             baseVisualShapeIndex=visualShapeId_tip_gripper,
                                             basePosition=pos2, baseOrientation=ori_tip_world)
            my_robot_bodies.append(tip_id_grip2)
            print(f"总共创建了 {len(my_robot_bodies)} 个物体 (包括末端)。")
        except Exception as e: print(f"创建末端物体出错: {e}"); p.disconnect(); sys.exit(1)
    else: print("警告: 无法创建末端物体，因为初始点或身体部件不足。")

    print("--- 初始化完成, 开始主仿真循环 ---")

    # --- 主循环 (使用 dl 数据驱动) ---
    play_mode = 'loop' # 播放模式: 'once' 或 'loop'
    current_row_index = 0
    num_data_rows = len(dl_sequence_m)

    if num_data_rows == 0: print("[错误] 没有可用的绳长变化量数据。"); p.disconnect(); sys.exit(1)
    print(f"将按顺序应用 {num_data_rows} 组绳长变化量数据。播放模式: {play_mode}")

    frame_count = 0
    last_print_time = time.time()

    try:
        while True:
            # --- 获取当前帧的绳长变化量 dl ---
            if current_row_index >= num_data_rows:
                if play_mode == 'loop': current_row_index = 0
                else: print("[信息] 数据播放完毕。"); break

            dl_segment = dl_sequence_m[current_row_index]

            # --- 计算新形态 (基于当前 dl_segment) ---
            my_ode._reset_y0() # 从初始状态开始积分
            ux, uy = calculate_curvatures_from_dl(dl_segment, cable_distance, L0_seg, num_cables)
            action_ode_segment = convert_curvatures_to_ode_action(ux, uy, 0.0, cable_distance, L0_seg) # 假设轴向伸缩为0
            my_ode.updateAction(action_ode_segment)
            sol = my_ode.odeStepFull() # 求解当前 dl 对应的形状

            # 打印调试信息 (可选)
            if current_row_index % 200 == 0:
                print(f"帧 {current_row_index}: dl(m)={np.round(dl_segment, 4)}, -> ux={ux:.3f}, uy={uy:.3f}")
                if sol is None: print("  ODE 求解失败!")

            # --- 更新 PyBullet 可视化对象 ---
            if sol is not None and sol.shape[1] >= 3:
                # 基座位姿保持不变 (已经在初始化时设定为垂直向下)
                # my_base_pos, my_base_ori 已在循环外设定为最终值

                # 从解中采样点
                idx = np.linspace(0, sol.shape[1] -1, my_number_of_sphere, dtype=int)
                positions_local = [(sol[0, i], sol[2, i], sol[1, i]) for i in idx] # 提取局部位置 (YZ交换)

                # 检查点和物体数量
                num_bodies_total = len(my_robot_bodies)
                num_tip_bodies = 3
                num_body_spheres = num_bodies_total - num_tip_bodies
                num_points_available = len(positions_local)
                num_spheres_to_update = min(num_body_spheres, num_points_available)

                # 更新身体球体位置和姿态
                for i in range(num_spheres_to_update):
                    pos_local = positions_local[i]
                    # 将局部坐标转换为世界坐标 (使用固定的垂直基座位姿)
                    pos_world, ori_world = p.multiplyTransforms(base_pos, base_ori,
                                                                pos_local, [0, 0, 0, 1])
                    try:
                        p.resetBasePositionAndOrientation(my_robot_bodies[i], pos_world, ori_world)
                    except p.error: pass # 忽略可能的错误

                # 更新末端位置和姿态
                if num_points_available >= 3 and num_bodies_total >= num_tip_bodies:
                    try:
                        # 计算末端局部姿态
                        ori_tip_local, _ = calculate_orientation(positions_local[-3], positions_local[-1])
                        # 将末端局部位置和姿态转换为世界坐标
                        pos_tip_world, ori_tip_world = p.multiplyTransforms(base_pos, base_ori,
                                                                            positions_local[-1], ori_tip_local)

                        # 更新头部 (通常是 my_robot_bodies[-3])
                        p.resetBasePositionAndOrientation(my_robot_bodies[-3], pos_tip_world, ori_tip_world)

                        # 更新夹爪指示器1 (通常是 my_robot_bodies[-2])
                        gripper_offset1_local = [0, 0.01, 0] # 局部 Y 轴偏移
                        pos1, _ = p.multiplyTransforms(pos_tip_world, ori_tip_world, gripper_offset1_local, [0,0,0,1])
                        p.resetBasePositionAndOrientation(my_robot_bodies[-2], pos1, ori_tip_world)

                        # 更新夹爪指示器2 (通常是 my_robot_bodies[-1])
                        gripper_offset2_local = [0, -0.01, 0] # 局部 负Y 轴偏移
                        pos2, _ = p.multiplyTransforms(pos_tip_world, ori_tip_world, gripper_offset2_local, [0,0,0,1])
                        p.resetBasePositionAndOrientation(my_robot_bodies[-1], pos2, ori_tip_world)
                    except Exception as e:
                        if frame_count % 60 == 0: print(f"警告: 更新末端时出错 - {e}")

            # ... (处理解无效或点数不足的警告，代码同上，省略) ...
            elif sol is not None and sol.shape[1] < 3:
                 if frame_count % 60 == 0: print(f"警告: 第 {current_row_index} 帧 ODE 解的点数 ({sol.shape[1]}) 少于3, 无法更新可视化。")
            elif sol is None:
                 if frame_count % 60 == 0: print(f"警告: 第 {current_row_index} 帧 ODE 求解失败, 无法更新可视化。")

            # --- 仿真步进和延时 ---
            p.stepSimulation()
            time.sleep(simulationStepTime) # 使用与仿真时间步长匹配的延时
            frame_count += 1

            # 移动到数据文件的下一行
            current_row_index += 1

    except KeyboardInterrupt: print("\n[信息] 检测到键盘中断 (Ctrl+C)，正在停止仿真...")
    except p.error as e: print(f"\n[错误] PyBullet 发生错误 (图形窗口是否已关闭?): {e}")
    except Exception as e: print(f"\n[错误] 发生意外错误: {e}"); import traceback; traceback.print_exc()
    finally:
        # 清理工作
        print("[信息] 断开 PyBullet 连接。")
        if p.isConnected():
            try: p.disconnect()
            except p.error: pass

    print("--- 仿真结束 ---")