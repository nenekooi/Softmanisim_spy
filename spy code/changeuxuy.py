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
sys.path.append(softmanisim_path)
from visualizer.visualizer import ODE
print("[调试] 成功从 'visualizer.visualizer' 导入 ODE")

def load_and_preprocess_data(file_path, sheet_name):
    """
    加载 Excel 文件，提取所需数据，并进行预处理 (无错误检查简化版)。

    Args:
        file_path (str): 输入 Excel 文件的路径。
        sheet_name (str): 要读取的工作表名称。

    Returns:
        tuple: 包含以下元素的元组:
            - absolute_lengths_mm (np.ndarray): 绝对绳长数组 (N x 3), 单位 mm。
            - real_xyz_mm (np.ndarray): 真实末端坐标数组 (N x 3), 单位 mm。
            - dl_sequence_m (np.ndarray): 绳长变化量数组 (N x 3), 单位 m。
            - L0_cables_mm (np.ndarray): 初始绳长数组 (1 x 3), 单位 mm。
    """
    print(f"[信息] 正在加载数据文件: {file_path}")
    df_input = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')
    print(f"[成功] 已加载数据，共 {len(df_input)} 行。")

    absolute_lengths_mm = df_input[['cblen1', 'cblen2', 'cblen3']].values
    real_xyz_mm = df_input[['X', 'Y', 'Z']].values
    print(f"[信息] 成功提取 {len(absolute_lengths_mm)} 行绳长和真实 XYZ 坐标。")

    L0_cables_mm = absolute_lengths_mm[0]
    print(f"[假设] 使用第一行 L0(mm): {L0_cables_mm}")
    absolute_lengths_m = absolute_lengths_mm / 1000.0
    L0_cables_m = L0_cables_mm / 1000.0
    dl_sequence_m = absolute_lengths_m - L0_cables_m
    print(f"[信息] 已计算 {len(dl_sequence_m)} 行 dl (m)。")

    return absolute_lengths_mm, real_xyz_mm, dl_sequence_m, L0_cables_mm

def initialize_results_storage():
    return {
        'cblen1_mm': [], 'cblen2_mm': [], 'cblen3_mm': [],
        'X_real_mm': [], 'Y_real_mm': [], 'Z_real_mm': [],
        'sim_X_mm': [], 'sim_Y_mm': [], 'sim_Z_mm': []
    }

def append_result(results_dict, cblen_mm, real_xyz_mm, sim_xyz_m):
    """
    将单步的输入和仿真结果追加到结果字典中。

    Args:
        results_dict (dict): 要追加到的结果字典。
        cblen_mm (np.ndarray): 当前步的输入绳长 (1x3), 单位 mm。
        real_xyz_mm (np.ndarray): 当前步的真实坐标 (1x3), 单位 mm。
        sim_xyz_m (np.ndarray): 当前步的仿真末端坐标 (1x3), 单位 m。
    """
    results_dict['cblen1_mm'].append(cblen_mm[0])
    results_dict['cblen2_mm'].append(cblen_mm[1])
    results_dict['cblen3_mm'].append(cblen_mm[2])
    results_dict['X_real_mm'].append(real_xyz_mm[0])
    results_dict['Y_real_mm'].append(real_xyz_mm[1])
    results_dict['Z_real_mm'].append(real_xyz_mm[2])

    sim_x_mm = sim_xyz_m[0] * 1000.0 if not np.isnan(sim_xyz_m[0]) else np.nan
    sim_y_mm = sim_xyz_m[1] * -1000.0 if not np.isnan(sim_xyz_m[1]) else np.nan
    sim_z_mm = sim_xyz_m[2] * 1000.0 - 480 if not np.isnan(sim_xyz_m[2]) else np.nan
    results_dict['sim_X_mm'].append(sim_x_mm)
    results_dict['sim_Y_mm'].append(sim_y_mm)
    results_dict['sim_Z_mm'].append(sim_z_mm)

def save_results_to_excel(results_dict, output_path):
    """
    将结果字典保存到 Excel 文件 (无错误检查简化版)。

    Args:
        results_dict (dict): 包含所有仿真结果的字典。
        output_path (str): 输出 Excel 文件的路径。
    """
    print("\n--- 保存仿真结果 ---")
    output_column_order = [
        'cblen1_mm', 'cblen2_mm', 'cblen3_mm',
        'X_real_mm', 'Y_real_mm', 'Z_real_mm',
        'sim_X_mm', 'sim_Y_mm', 'sim_Z_mm'
    ]
    output_results_df = pd.DataFrame(results_data)
    output_results_df = output_results_df[output_column_order]

    print(f"[信息] 正在将 {len(output_results_df)} 条结果 ({len(output_column_order)} 列) 保存到: {output_path}")
    output_results_df.to_excel(output_path, index=False, engine='openpyxl')
    print(f"[成功] 结果已保存至: {os.path.abspath(output_path)}")

def calculate_orientation(point1, point2):
    """根据两点计算方向 (四元数)"""
    diff = np.array(point2) - np.array(point1)
    norm_diff = np.linalg.norm(diff)
    # 保留对重合点的检查以避免数学错误
    if norm_diff < 1e-6:
        return p.getQuaternionFromEuler([0, 0, 0]), [0, 0, 0]
    # 保留对垂直情况的检查以计算 Yaw
    if np.linalg.norm(diff[:2]) < 1e-6:
        yaw = 0
    else:
        yaw = math.atan2(diff[1], diff[0])
    pitch = math.atan2(-diff[2], math.sqrt(diff[0]**2 + diff[1]**2))
    roll = 0
    return p.getQuaternionFromEuler([roll, pitch, yaw]), [roll, pitch, yaw]

def calculate_curvatures_from_dl(dl_segment, d, L0_seg, num_cables=3):
    """根据绳长变化量计算曲率 ux, uy (假设输入有效且为3绳索)。"""
    ux = 0.0
    uy = 0.0
    # 保留对半径 d 的检查以避免除零
    if abs(d) < 1e-9:
        print("警告: 绳索半径 d 接近于零。")
        return ux, uy
    # 移除对 dl_segment 长度的检查
    # 移除对 num_cables == 3 的检查，直接按 3 绳索处理
    dl1 = dl_segment[0] # 如果 dl_segment 长度不足会在此处中断
    dl2 = dl_segment[1]
    dl3 = dl_segment[2]
    uy = -dl1 / d
    denominator_ux = d * math.sqrt(3.0)
    # 保留对 ux 分母的检查以避免除零
    if abs(denominator_ux) > 1e-9:
        ux = (dl3 - dl2) / denominator_ux
    else:
        ux = 0.0 # 分母为零时 ux 定义为 0
        print("警告: 计算 ux 时分母接近零。")
    return ux, uy

def convert_curvatures_to_ode_action(ux, uy, length_change, d, L0_seg):
    """将曲率转换为 ODE action。"""
    l = L0_seg
    action_ode = np.zeros(3)
    action_ode[0] = length_change
    action_ode[1] = uy * l * d
    action_ode[2] = -ux * l * d
    return action_ode

def calculate_curvatures_from_dl_v2(dl_segment, d, L0_seg):
    """
    使用针对 3 根对称缆绳的标准公式，根据不同的缆绳长度变化量计算曲率 ux, uy。

    假设缆绳 1 位于 0 度，缆绳 2 位于 120 度，缆绳 3 位于 240 度，
    相对于与 uy 弯曲相关的轴。

    Args:
        dl_segment (np.ndarray): 包含缆绳长度变化量 [dl1, dl2, dl3] 的数组 (单位：米)。
        d (float): 缆绳到中心骨干的径向距离 (单位：米)。
        L0_seg (float): 段的初始长度 (单位：米)。用于比例缩放。

    Returns:
        tuple: (ux, uy) 曲率 (单位：1/米)。
               ux: 绕局部 y 轴的曲率。
               uy: 绕局部 x 轴的曲率。
    """
    ux = 0.0
    uy = 0.0

    # 对有效输入进行基本检查
    if abs(d) < 1e-9:
        print("[警告] calculate_curvatures_from_dl_v2: 缆绳距离 'd' 接近于零。")
        return ux, uy
    if abs(L0_seg) < 1e-9:
        print("[警告] calculate_curvatures_from_dl_v2: 段长度 'L0_seg' 接近于零。")
        return ux, uy

    dl1 = dl_segment[0]
    dl2 = dl_segment[1]
    dl3 = dl_segment[2]

    # 分母部分 L*d
    # 我们使用 L0_seg 作为参考长度 L 来计算曲率贡献
    Ld = L0_seg * d

    # 计算 ux (绕 y 轴弯曲)
    ux_denominator = Ld * math.sqrt(3.0)
    if abs(ux_denominator) > 1e-9:
        # 注意: 原始代码是 dl3 - dl2。根据缆绳 2/3 相对于 y 轴的定义，
        # 可能是 dl2 - dl3。我们暂时先保持 dl3 - dl2。
        ux = (dl3 - dl2) / ux_denominator
    else:
        ux = 0.0

    # 计算 uy (绕 x 轴弯曲)
    uy_denominator = 3.0 * Ld
    if abs(uy_denominator) > 1e-9:
         uy = (2.0 * dl1 - dl2 - dl3) / uy_denominator
    else:
         uy = 0.0

    # --- 关于坐标系的重要说明 ---
    # ODE 类 (`visualizer.py`) 在其 u_hat 矩阵中使用了 ux, uy：
    # u_hat = np.array([[0, 0, self.uy], [0, 0, -self.ux],[-self.uy, self.ux, 0]])
    # 这意味着：
    # - uy 导致绕 x 轴的旋转 (dR ~ R@[[0,0,uy],[0,0,0],[-uy,0,0]])
    # - ux 导致绕 y 轴的旋转 (dR ~ R@[[0,0,0],[0,0,-ux],[0,ux,0]])
    # 你需要确保计算出的 ux, uy 与这个定义相匹配，这取决于你的物理设置
    # 和缆绳编号（哪根缆绳对应 dl1, dl2, dl3）。
    # 例如，如果你的设置中 1 号缆绳主要导致绕物理 Y 轴的弯曲，
    # 你可能需要交换计算出的 ux 和 uy，或者调整它们的符号。

    # 目前，我们基于上面的标准推导返回 ux, uy。
    return ux, uy

def calculate_curvatures_from_dl_v3(dl_segment, d, L0_seg, AXIAL_ACTION_SCALE=1.0):
    """
    计算曲率 ux, uy，使用标准公式，但使用当前估计长度而非初始长度进行缩放。

    Args:
        dl_segment (np.ndarray): 包含缆绳长度变化量 [dl1, dl2, dl3] 的数组 (单位：米)。
        d (float): 缆绳到中心骨干的径向距离 (单位：米)。
        L0_seg (float): 段的初始长度 (单位：米)。
        axial_scale (float): 应用于 avg_dl 以估计长度变化的比例因子。默认为 1.0。

    Returns:
        tuple: (ux, uy) 曲率 (单位：1/米)。
               ux: 绕局部 y 轴的曲率。
               uy: 绕局部 x 轴的曲率。
    """
    ux = 0.0
    uy = 0.0

    # 基本检查
    if abs(d) < 1e-9:
        print("[警告] calculate_curvatures_from_dl_v3: 缆绳距离 'd' 接近于零。")
        return ux, uy
    if abs(L0_seg) < 1e-9:
        print("[警告] calculate_curvatures_from_dl_v3: 初始段长度 'L0_seg' 接近于零。")
        # 如果 L0 为零，无法估计当前长度
        return ux, uy

    dl1 = dl_segment[0]
    dl2 = dl_segment[1]
    dl3 = dl_segment[2]

    # 根据平均 dl 和比例因子估计当前长度 L
    avg_dl = (dl1 + dl2 + dl3) / 3.0
    L_current_estimate = L0_seg + avg_dl * AXIAL_ACTION_SCALE
    # 确保估计长度为正
    if L_current_estimate <= 1e-9:
         print(f"[警告] calculate_curvatures_from_dl_v3: 估计的当前长度 ({L_current_estimate:.4f}) 接近零或为负。将使用 L0_seg 代替。")
         L_current_estimate = L0_seg

    # 使用当前估计长度计算分母部分 L*d
    Ld = L_current_estimate * d
    if abs(Ld) < 1e-9:
        print("[警告] calculate_curvatures_from_dl_v3: 乘积 L_current*d 接近于零。")
        # 如果 L_current 估计有问题或为零，则回退到使用 L0
        Ld = L0_seg * d
        if abs(Ld) < 1e-9:
             return 0.0, 0.0 # 如果 L0*d 也为零，则无法计算

    # 计算 ux (绕 y 轴弯曲)
    ux_denominator = Ld * math.sqrt(3.0)
    if abs(ux_denominator) > 1e-9:
        ux = (dl3 - dl2) / ux_denominator
    else:
        ux = 0.0

    # 计算 uy (绕 x 轴弯曲)
    uy_denominator = 3.0 * Ld
    if abs(uy_denominator) > 1e-9:
         uy = (2.0 * dl1 - dl2 - dl3) / uy_denominator
    else:
         uy = 0.0

    # 关于坐标系的警告同样适用于这里
    return ux, uy
if __name__ == "__main__":

    print("--- 设置参数 ---")
    DATA_FILE_PATH = 'D:/data/load_data/random_data.xlsx'
    # DATA_FILE_PATH = 'D:/data/load_data/circle.xlsx'
    SHEET_NAME = 'Sheet1'
    OUTPUT_RESULTS_PATH = 'D:/data/save_data/4(u_new_3,cab=0.04,k=-200,a=0.6).xlsx'
    # OUTPUT_RESULTS_PATH = 'D:/data/save_data/circle(u_new_2,cab=0.04,k=-200,a=0.8).xlsx'
    

    num_cables = 3 
    cable_distance = 0.04
    initial_length = 0.12
    number_of_segment = 1
    L0_seg = initial_length / number_of_segment
    print(f"机器人参数: L0={initial_length:.4f}m, d={cable_distance:.4f}m")
    axial_strain_coefficient = -200
    AXIAL_ACTION_SCALE = 0.6

    body_color = [1, 0.0, 0.0, 1]
    head_color = [0.0, 0.0, 0.75, 1]
    body_sphere_radius = 0.02
    number_of_sphere = 30
    my_sphere_radius = body_sphere_radius
    my_number_of_sphere = number_of_sphere
    my_head_color = head_color

    # --- 加载数据 ---
    absolute_lengths_mm, real_xyz_mm, dl_sequence_m, L0_cables_mm = load_and_preprocess_data(DATA_FILE_PATH, SHEET_NAME)

    # --- 初始化结果存储 ---
    results_data = initialize_results_storage()

    # --- PyBullet 初始化 ---
    print("--- 初始化 PyBullet ---")
    simulationStepTime = 0.0001
    physicsClientId = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(simulationStepTime)
    planeId = p.loadURDF("plane.urdf")
    print(f"加载 plane.urdf, ID: {planeId}")
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.resetDebugVisualizerCamera(cameraDistance=0.8, cameraYaw=45, cameraPitch=-20, cameraTargetPosition=[0, 0, 0.3])
    print("[信息] 已设置相机视角。")

    # --- 初始化 ODE 对象 ---
    print("--- 初始化 ODE 对象 ---")
    my_ode = ODE()
    my_ode.l0 = initial_length
    my_ode.d = cable_distance
    print(f"ODE 已初始化 L0={my_ode.l0:.4f}m, d={my_ode.d:.4f}m")

    # --- 计算初始形态 ---
    print("--- 计算初始形状 (dl=0) ---")
    act0_segment = np.zeros(3)
    my_ode._reset_y0()
    my_ode.updateAction(act0_segment)
    sol0 = my_ode.odeStepFull()
    print(f"初始形状计算完成。") 

    # --- 设置基座 ---
    base_pos = np.array([0, 0, 0.6])
    base_ori_euler = np.array([-math.pi / 2.0, 0, 0])
    base_ori = p.getQuaternionFromEuler(base_ori_euler)
    print(f"[设置] 基座世界坐标: {base_pos}")
    print(f"[设置] 基座世界姿态 (Euler): {base_ori_euler}")
    radius = my_sphere_radius

    # --- 创建 PyBullet 形状 ---
    print("--- 创建 PyBullet 形状 ---")
    shape = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
    visualShapeId = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=body_color)
    visualShapeId_tip_body = p.createVisualShape(p.GEOM_SPHERE, radius=radius+0.0025, rgbaColor=my_head_color)
    visualShapeId_tip_gripper = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.002, 0.01, 0.002], rgbaColor=head_color)

    # --- 创建 PyBullet 物体 ---
    print("--- 创建 PyBullet 物体 ---")
    idx0 = np.linspace(0, sol0.shape[1] - 1, my_number_of_sphere, dtype=int)
    positions0_local = [(sol0[0, i], sol0[2, i], sol0[1, i]) for i in idx0]
    my_robot_bodies = []
    for i, pos_local in enumerate(positions0_local):
        pos_world, ori_world = p.multiplyTransforms(base_pos, base_ori, pos_local, [0,0,0,1])
        my_robot_bodies.append(p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=visualShapeId, basePosition=pos_world, baseOrientation=ori_world))

    ori_tip_local, _ = calculate_orientation(positions0_local[-3], positions0_local[-1]) 
    pos_tip_world, ori_tip_world = p.multiplyTransforms(base_pos, base_ori, positions0_local[-1], ori_tip_local)
    my_robot_bodies.append(p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=visualShapeId_tip_body, basePosition=pos_tip_world, baseOrientation=ori_tip_world))
    gripper_offset1 = [0, 0.01, 0]
    pos1, _ = p.multiplyTransforms(pos_tip_world, ori_tip_world, gripper_offset1, [0,0,0,1])
    my_robot_bodies.append(p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=visualShapeId_tip_gripper, basePosition=pos1, baseOrientation=ori_tip_world))
    gripper_offset2 = [0,-0.01, 0]
    pos2, _ = p.multiplyTransforms(pos_tip_world, ori_tip_world, gripper_offset2, [0,0,0,1])
    my_robot_bodies.append(p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=visualShapeId_tip_gripper, basePosition=pos2, baseOrientation=ori_tip_world))
    print(f"总共创建了 {len(my_robot_bodies)} 个物体。")

    print("--- 初始化完成, 开始主仿真循环 ---")

    # --- 主循环 (改为 for 循环) ---
    num_data_rows = len(dl_sequence_m)
    print(f"将按顺序应用 {num_data_rows} 组绳长变化量数据。")
    for current_row_index in range(num_data_rows):
        dl_segment = dl_sequence_m[current_row_index]
        current_cblen_mm = absolute_lengths_mm[current_row_index]
        current_real_xyz_mm = real_xyz_mm[current_row_index]

        # --- 计算新形态 ---
        my_ode._reset_y0()
        ux, uy = calculate_curvatures_from_dl_v3(dl_segment, cable_distance, L0_seg, AXIAL_ACTION_SCALE)
        avg_dl = np.mean(dl_segment)
        commanded_length_change = avg_dl * AXIAL_ACTION_SCALE
        action_ode_segment = convert_curvatures_to_ode_action(ux, uy, commanded_length_change, cable_distance, L0_seg)
        my_ode.updateAction(action_ode_segment)
        sol = my_ode.odeStepFull()

        # --- 更新可视化并获取仿真末端位置 ---
        pos_tip_world_m = np.array([np.nan, np.nan, np.nan])
        if sol is not None and sol.shape[1] >= 3:
            idx = np.linspace(0, sol.shape[1] -1, my_number_of_sphere, dtype=int)
            positions_local = [(sol[0, i], sol[2, i], sol[1, i]) for i in idx]
            num_bodies_total = len(my_robot_bodies)
            num_tip_bodies = 3
            num_body_spheres = num_bodies_total - num_tip_bodies
            num_points_available = len(positions_local)
            num_spheres_to_update = min(num_body_spheres, num_points_available)

            for i in range(num_spheres_to_update):
                pos_world, ori_world = p.multiplyTransforms(base_pos, base_ori, positions_local[i], [0,0,0,1])
                p.resetBasePositionAndOrientation(my_robot_bodies[i], pos_world, ori_world)

            ori_tip_local, _ = calculate_orientation(positions_local[-3], positions_local[-1])
            pos_tip_world_tuple, ori_tip_world = p.multiplyTransforms(base_pos, base_ori, positions_local[-1], ori_tip_local)
            pos_tip_world_m = np.array(pos_tip_world_tuple)

            p.resetBasePositionAndOrientation(my_robot_bodies[-3], pos_tip_world_m, ori_tip_world)
            gripper_offset1 = [0, 0.01, 0]
            pos1, _ = p.multiplyTransforms(pos_tip_world_m, ori_tip_world, gripper_offset1, [0,0,0,1])
            p.resetBasePositionAndOrientation(my_robot_bodies[-2], pos1, ori_tip_world)
            gripper_offset2 = [0,-0.01, 0]
            pos2, _ = p.multiplyTransforms(pos_tip_world_m, ori_tip_world, gripper_offset2, [0,0,0,1])
            p.resetBasePositionAndOrientation(my_robot_bodies[-1], pos2, ori_tip_world)



        append_result(results_data, current_cblen_mm, current_real_xyz_mm, pos_tip_world_m)


        p.stepSimulation()
        time.sleep(simulationStepTime)


    print("[信息] 所有数据已播放完毕。") 

    # --- 仿真结束，清理 ---
    print("[信息] 断开 PyBullet 连接。")
    p.disconnect(physicsClientId)

    # --- 保存结果 ---
    save_results_to_excel(results_data, OUTPUT_RESULTS_PATH)

    print("--- 仿真结束 ---")