# -*- coding: utf-8 -*-
# ... (保留现有导入) ...
import pandas as pd
import numpy as np
import math
import pybullet as p
import pybullet_data
import time
import os
import sys

# --- 确保正确的导入路径 ---
softmanisim_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if softmanisim_path not in sys.path:
    sys.path.append(softmanisim_path)


from visualizer.kinematic_model import ODE_PhysicsBased # <<< 根据你的文件结构调整路径


def calculate_inputs_from_dl_physics(dl_segment, d, L0_seg, k_c, EA, EI, num_cables=3):
    ux = 0.0; uy = 0.0; epsilon = 0.0
    if num_cables != 3: print(f"错误: 此物理模型目前仅支持3根绳索。"); return ux, uy, epsilon
    if len(dl_segment) != num_cables: print(f"警告: dl_segment 长度 {len(dl_segment)} != num_cables {num_cables}"); return ux, uy, epsilon
    if abs(d) < 1e-9 or k_c <= 0 or EA <= 0 or EI <= 0: print(f"警告: 无效的物理参数 (d={d}, k_c={k_c}, EA={EA}, EI={EI})。"); return ux, uy, epsilon
    tau = k_c * np.array(dl_segment); tau1, tau2, tau3 = tau[0], tau[1], tau[2]
    N = tau1 + tau2 + tau3
    Mx = (tau3 - tau2) * d * (math.sqrt(3.0) / 2.0)
    My = ( (tau2 + tau3)/2.0 - tau1 ) * d
    epsilon = N / EA
    ux = My / EI
    uy = -Mx / EI
    return ux, uy, epsilon
# --- 粘贴函数结束 ---

# --- 保留辅助函数，如 calculate_orientation ---
def calculate_orientation(point1, point2):
    # ... (保留现有代码) ...
    diff = np.array(point2) - np.array(point1); norm_diff = np.linalg.norm(diff)
    if norm_diff < 1e-6: return p.getQuaternionFromEuler([0,0,0]), [0,0,0]
    if np.linalg.norm(diff[:2]) < 1e-6: yaw = 0
    else: yaw = math.atan2(diff[1], diff[0])
    pitch = math.atan2(-diff[2], math.sqrt(diff[0]**2 + diff[1]**2)); roll = 0
    return p.getQuaternionFromEuler([roll, pitch, yaw]), [roll, pitch, yaw]

def calculate_inputs_from_dl_physics(dl_segment, d, L0_seg, k_c, EA, EI_base, k_stiffen, num_cables=3):
    """
    根据拉线长度变化量(dl)，基于估算的张力和刚度属性(EA, EI)计算输入。
    包含非线性弯曲刚度: EI_eff = EI_base + k_stiffen * (ux^2 + uy^2)。

    参数:
        dl_segment (np.array): 拉线长度变化量数组 [dl1, dl2, dl3] (米)。
        d (float): 拉线径向距离 (米)。
        L0_seg (float): 段参考长度 (米)。
        k_c (float): 拉线刚度 (N/m)。
        EA (float): 段轴向刚度 (N)。
        EI_base (float): 段的基础弯曲刚度 (零曲率时) (Nm^2)。
        k_stiffen (float): 弯曲硬化系数 (单位: Nm^2 / (rad/m)^2, 如果p=2)。非负。
        num_cables (int): 拉线数量 (目前支持 3)。

    返回:
        tuple: (ux, uy, epsilon) 基于物理计算得到的结果。
    """
    ux = 0.0
    uy = 0.0
    epsilon = 0.0

    # --- 输入验证 ---
    if num_cables != 3:
        print(f"错误: 此物理模型目前仅支持3根绳索。")
        return ux, uy, epsilon
    if len(dl_segment) != num_cables:
        print(f"警告: dl_segment 长度 {len(dl_segment)} != num_cables {num_cables}")
        return ux, uy, epsilon
    # 允许 k_stiffen 为零 (即无硬化效应)
    if abs(d) < 1e-9 or k_c <= 0 or EA <= 0 or EI_base <= 0 or k_stiffen < 0:
        print(f"警告: 无效的物理参数 (d={d}, k_c={k_c}, EA={EA}, EI_base={EI_base}, k_stiffen={k_stiffen})。")
        return ux, uy, epsilon

    # --- 1. 估算拉线张力 ---
    tau = k_c * np.array(dl_segment)
    tau1, tau2, tau3 = tau[0], tau[1], tau[2]

    # --- 2. 估算合力与合力矩 (N, Mx, My) ---
    # (保持之前的几何假设和计算)
    N = tau1 + tau2 + tau3
    Mx = (tau3 - tau2) * d * (math.sqrt(3.0) / 2.0)
    My = ( (tau2 + tau3)/2.0 - tau1 ) * d

    # --- 3. 计算应变 (包含非线性 EI) ---
    # 轴向应变 (保持不变)
    epsilon = N / EA

    # 曲率 (u = M / EI_eff) - 近似计算方法
    # 步骤 A: 使用 EI_base 估算初始曲率
    if abs(EI_base) < 1e-9: # 避免除零
        ux_initial = 0.0
        uy_initial = 0.0
    else:
        ux_initial = My / EI_base
        uy_initial = -Mx / EI_base # 注意符号约定

    # 步骤 B: 计算初始总曲率的平方 (假设 p=2)
    curvature_sq_initial = ux_initial**2 + uy_initial**2

    # 步骤 C: 计算有效弯曲刚度 EI_effective
    EI_eff = EI_base + k_stiffen * curvature_sq_initial

    # 确保 EI_eff > 0
    EI_eff = max(EI_eff, 1e-9) # 防止除零或刚度为负

    # 步骤 D: 使用 EI_effective 计算最终曲率
    ux = My / EI_eff
    uy = -Mx / EI_eff # 注意符号约定

    # (可选: 添加限制)
    # ...

    return ux, uy, epsilon

# --- 主程序 ---
if __name__ == "__main__":

    # 1. 输入数据文件路径
    DATA_FILE_PATH = 'c:/Users/11647/Desktop/data/Processed_Data3w_20250318.xlsx' # <<< 确认 Excel 文件路径
    SHEET_NAME = 'Sheet1'
    # 2. 输出结果文件路径
    output_dir = os.path.dirname(DATA_FILE_PATH) if os.path.dirname(DATA_FILE_PATH) else '.'
    OUTPUT_RESULTS_PATH = os.path.join(output_dir, 'sim_vs_real_physics_model_3w(7).xlsx') # 修改输出文件名

    # 3. 机器人基本物理参数
    num_cables = 3
    cable_distance = 0.004 # (米) 拉线径向距离
    initial_length = 0.12 # (米) 段的参考长度
    number_of_segment = 1 # 假设目前为单段
    L0_seg = initial_length / number_of_segment

    # --- <<< 新增: 定义模型的物理参数 (需要仔细调试!) >>> ---
    # 以下值仅为示例，**必须**根据你的机器人材料和结构进行估计或调试。
    CABLE_STIFFNESS_Kc = 1000.0 # (N/m) 示例值 - 每米拉伸需要多少牛的力?
    AXIAL_STIFFNESS_EA = 200.0  # (N) 示例值 - 抗拉/压刚度
    BENDING_STIFFNESS_EI_BASE = 0.045 # (Nm^2) 示例值 - 抗弯刚度
    BENDING_STIFFENING_K = 0.001 # (Nm^2 / (rad/m)^2, if p=2) <<< 从一个较小正值开始尝试
    # --- <<< 新参数定义结束 >>> ---

  

    # 4. 可视化参数 (保持不变)
    body_color = [1, 0.0, 0.0, 1]; head_color = [0.0, 0.0, 0.75, 1]
    body_sphere_radius = 0.02; number_of_sphere = 30
    my_sphere_radius = body_sphere_radius; my_number_of_sphere = number_of_sphere; my_head_color = head_color

    # --- 加载和处理数据 (基本保持不变) ---
    print("--- 加载和处理数据 ---")
    # ... (读取 excel, 检查列, 计算 dl_sequence_m 的代码) ...
    absolute_lengths_mm = None; dl_sequence_m = None; real_xyz_mm = None
    if not os.path.exists(DATA_FILE_PATH): print(f"[错误] 文件未找到: {DATA_FILE_PATH}"); sys.exit(1)
    try:
        df_input = pd.read_excel(DATA_FILE_PATH, sheet_name=SHEET_NAME, engine='openpyxl')
        print(f"[成功] 已加载数据，共 {len(df_input)} 行。")
        required_cols = ['cblen1', 'cblen2', 'cblen3', 'X', 'Y', 'Z']
        if not all(col in df_input.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in df_input.columns]
            raise ValueError(f"错误: 文件缺少必需的列: {missing_cols}")
        absolute_lengths_mm = df_input[['cblen1', 'cblen2', 'cblen3']].values
        real_xyz_mm = df_input[['X', 'Y', 'Z']].values
        print(f"[信息] 成功提取 {len(absolute_lengths_mm)} 行绳长和真实 XYZ 坐标。")
        if len(absolute_lengths_mm) > 1:
            L0_cables_mm = absolute_lengths_mm[1] # <<< 使用第二行的数据作为参考 L0
            print(f"[假设] 使用第二行 L0(mm): {L0_cables_mm}")
            absolute_lengths_m = absolute_lengths_mm / 1000.0
            L0_cables_m = L0_cables_mm / 1000.0
            # 计算 dl 时，所有行的绝对长度都减去这个参考长度
            dl_sequence_m = absolute_lengths_m - L0_cables_m
            print(f"[信息] 已计算 {len(dl_sequence_m)} 行 dl (m)，相对于第二行。")
             # 注意：dl_sequence_m 的第一行现在将是非零值（除非第一行和第二行绳长相同）
            # 对应的，第一行 real_xyz 可能也不是 (0,0,0)，这没关系。
        else:
            raise ValueError("错误: 数据文件行数不足2，无法使用第二行作为参考 L0。")
    except Exception as e: print(f"[错误] 加载或处理数据时出错: {e}"); sys.exit(1)
    if dl_sequence_m is None: print("[错误] 未能计算 dl 序列。"); sys.exit(1)


    # --- PyBullet 初始化 (保持不变) ---
    print("--- 初始化 PyBullet ---")
    simulationStepTime = 0.001; physicsClientId = -1
    try: physicsClientId = p.connect(p.GUI);
    except Exception as e: print(f"连接 PyBullet 出错: {e}"); sys.exit(1)
    p.setAdditionalSearchPath(pybullet_data.getDataPath()); p.setGravity(0, 0, -9.81); p.setTimeStep(simulationStepTime)
    try: planeId = p.loadURDF("plane.urdf"); print(f"加载 plane.urdf, ID: {planeId}")
    except p.error as e: print(f"加载 plane.urdf 出错: {e}"); p.disconnect(physicsClientId); sys.exit(1)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
    p.resetDebugVisualizerCamera(cameraDistance=0.8, cameraYaw=45, cameraPitch=-20, cameraTargetPosition=[0, 0, 0.3])


    # --- 初始化 *新* 的 ODE 对象 ---
    print("--- 初始化 ODE_PhysicsBased 对象 ---")
    my_ode = ODE_PhysicsBased(initial_length_m=L0_seg,
                              cable_distance_m=cable_distance,
                              ode_step_ds=0.001) # 可以尝试减小步长提高精度
    print(f"ODE 已初始化 L0={my_ode.l0:.4f}m, d={my_ode.d:.4f}m")

    # --- 计算初始形状 (dl=0 -> 理想情况下 ux=0, uy=0, epsilon=0) ---
    print("--- 计算初始形状 (dl=0) ---")
    my_ode._reset_y0()
    # 对于 dl=0, 基于物理的计算应该得到 ux=0, uy=0, epsilon=0 (如果无预紧)
    ux0, uy0, eps0 = calculate_inputs_from_dl_physics(
        np.zeros(3), cable_distance, L0_seg,
        CABLE_STIFFNESS_Kc, AXIAL_STIFFNESS_EA,
        BENDING_STIFFNESS_EI_BASE, # 传递 EI_base
        BENDING_STIFFENING_K       # 传递 k_stiffen
    )
    my_ode.update_inputs(ux=ux0, uy=uy0, epsilon=eps0, delta_l=0.0) # 使用新的更新方法
    sol0 = my_ode.odeStepFull()
    if sol0 is None or sol0.shape[1] < 3: print("错误: 初始 ODE 求解失败或点数不足(<3)。"); p.disconnect(physicsClientId); sys.exit(1)
    print(f"初始形状计算完成，末端状态 (r): {my_ode.states[:3]}") # 打印初始末端位置

    # --- 设置基座姿态并创建 PyBullet 形状/物体 (保持不变) ---
    print("--- 设置基座并创建 PyBullet 物体 ---")
    base_pos = np.array([0, 0, 0.6]); base_ori_euler = np.array([-math.pi / 2.0, 0, 0]); base_ori = p.getQuaternionFromEuler(base_ori_euler)
    radius = my_sphere_radius
    try: shape = p.createCollisionShape(p.GEOM_SPHERE, radius=radius); visualShapeId = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=body_color); visualShapeId_tip_body = p.createVisualShape(p.GEOM_SPHERE, radius=radius+0.0025, rgbaColor=my_head_color); visualShapeId_tip_gripper = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.002, 0.01, 0.002], rgbaColor=head_color)
    except p.error as e: print(f"创建形状出错: {e}"); p.disconnect(physicsClientId); sys.exit(1)
    if sol0.shape[1] < 3: print("错误: 初始解点数不足3。"); p.disconnect(physicsClientId); sys.exit(1)
    idx0 = np.linspace(0, sol0.shape[1] - 1, my_number_of_sphere, dtype=int); positions0_local = [(sol0[0, i], sol0[2, i], sol0[1, i]) for i in idx0] # YZ 交换以适应可视化
    my_robot_bodies = [];
    try:
        for i, pos_local in enumerate(positions0_local): pos_world, ori_world = p.multiplyTransforms(base_pos, base_ori, pos_local, [0,0,0,1]); my_robot_bodies.append(p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=visualShapeId, basePosition=pos_world, baseOrientation=ori_world))
        num_initial_bodies = len(my_robot_bodies); num_tip_bodies_expected = 3
        if len(positions0_local) >= 3 and num_initial_bodies == my_number_of_sphere:
            ori_tip_local, _ = calculate_orientation(positions0_local[-3], positions0_local[-1]); pos_tip_world, ori_tip_world = p.multiplyTransforms(base_pos, base_ori, positions0_local[-1], ori_tip_local)
            my_robot_bodies.append(p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=visualShapeId_tip_body, basePosition=pos_tip_world, baseOrientation=ori_tip_world))
            gripper_offset1 = [0, 0.01, 0]; pos1, _ = p.multiplyTransforms(pos_tip_world, ori_tip_world, gripper_offset1, [0,0,0,1]); my_robot_bodies.append(p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=visualShapeId_tip_gripper, basePosition=pos1, baseOrientation=ori_tip_world))
            gripper_offset2 = [0,-0.01, 0]; pos2, _ = p.multiplyTransforms(pos_tip_world, ori_tip_world, gripper_offset2, [0,0,0,1]); my_robot_bodies.append(p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=visualShapeId_tip_gripper, basePosition=pos2, baseOrientation=ori_tip_world))
            print(f"总共创建了 {len(my_robot_bodies)} 个物体。")
        else: print("警告: 无法创建末端物体。")
    except Exception as e: print(f"创建物体时出错: {e}"); p.disconnect(physicsClientId); sys.exit(1)

    print("--- 初始化完成, 开始主仿真循环 ---")

    # --- 初始化结果存储字典 (保持不变, 可选增加计算出的输入) ---
    results_data = {
        'cblen1_mm': [], 'cblen2_mm': [], 'cblen3_mm': [],
        'X_real_mm': [], 'Y_real_mm': [], 'Z_real_mm': [],
        'sim_X_mm': [], 'sim_Y_mm': [], 'sim_Z_mm': [],
        # 可选: 记录计算出的输入值以供分析
        'calc_ux': [], 'calc_uy': [], 'calc_epsilon': []
    }

    # --- 主循环 ---
    play_mode = 'single'; current_row_index = 0; num_data_rows = len(dl_sequence_m)
    if num_data_rows == 0: print("[错误] 没有可用的绳长变化量数据。"); p.disconnect(physicsClientId); sys.exit(1)
    print(f"将按顺序应用 {num_data_rows} 组绳长变化量数据。播放模式: {play_mode}")
    frame_count = 0; last_print_time = time.time(); simulation_running = True

    try:
        while simulation_running:
            # --- 获取当前行的 dl ---
            if current_row_index >= num_data_rows:
                print("[信息] 所有数据已播放完毕。"); simulation_running = False; continue
            if not p.isConnected(physicsClientId): print("[警告] PyBullet 连接已断开。"); simulation_running = False; continue
            dl_segment = dl_sequence_m[current_row_index]

            # --- <<< 新增: 使用物理模型计算输入 >>> ---
            my_ode._reset_y0() # 如果每一步都从 s=0 开始积分，则需要重置
            ux, uy, epsilon = calculate_inputs_from_dl_physics(
                dl_segment,
                cable_distance,
                L0_seg,
                CABLE_STIFFNESS_Kc,
                AXIAL_STIFFNESS_EA,
                BENDING_STIFFNESS_EI_BASE, # 传递 EI_base
                BENDING_STIFFENING_K,      # 传递 k_stiffen
                num_cables
            )

            # --- <<< 新增: 使用计算得到的输入更新 ODE 对象 >>> ---
            # 假设 delta_l (指令性长度变化) 为零，除非你的 dl_segment 包含了这部分信息。
            # 目前假设 dl_segment 只用于弯曲/张力计算。
            commanded_delta_l = 0.0 # 如果需要，可以从 dl_segment 推导
            my_ode.update_inputs(ux, uy, epsilon, delta_l=commanded_delta_l)

            # --- 求解 ODE ---
            sol = my_ode.odeStepFull()

            # --- 更新可视化并记录结果 ---
            pos_tip_world_m = np.array([np.nan, np.nan, np.nan]) # 初始化为 NaN
            if sol is not None and sol.shape[1] >= 3:
                # --- 更新可视化 (保持现有逻辑) ---
                idx = np.linspace(0, sol.shape[1] -1, my_number_of_sphere, dtype=int)
                positions_local = [(sol[0, i], sol[2, i], sol[1, i]) for i in idx] # YZ 交换
                num_bodies_total = len(my_robot_bodies); num_tip_bodies = 3
                num_body_spheres = num_bodies_total - num_tip_bodies; num_points_available = len(positions_local)
                num_spheres_to_update = min(num_body_spheres, num_points_available)
                for i in range(num_spheres_to_update): # 更新躯干球体
                    pos_world, ori_world = p.multiplyTransforms(base_pos, base_ori, positions_local[i], [0,0,0,1])
                    try:
                        if i < len(my_robot_bodies): p.resetBasePositionAndOrientation(my_robot_bodies[i], pos_world, ori_world)
                    except p.error: pass # 忽略可能的 PyBullet 错误

                # --- 获取末端位置 (确保使用解的最后一个点) ---
                if num_points_available >= 3 and num_bodies_total >= num_tip_bodies:
                    try:
                        # 使用解的最后一个点作为末端位置
                        final_pos_local = (sol[0, -1], sol[2, -1], sol[1, -1]) # YZ 交换
                        # 如果需要精确的末端姿态，可能需要使用解中的 R 矩阵
                        # 这里简化处理，使用临近点计算姿态
                        ori_tip_local, _ = calculate_orientation(positions_local[-3] if len(positions_local)>2 else positions_local[-1], positions_local[-1])
                        # 计算世界坐标系下的末端位姿
                        pos_tip_world_tuple, ori_tip_world = p.multiplyTransforms(base_pos, base_ori, final_pos_local, ori_tip_local)
                        pos_tip_world_m = np.array(pos_tip_world_tuple) # (米)

                        # 更新末端可视化 (保持现有逻辑)
                        if len(my_robot_bodies) >= num_tip_bodies:
                           p.resetBasePositionAndOrientation(my_robot_bodies[-3], pos_tip_world_m, ori_tip_world)
                           gripper_offset1 = [0, 0.01, 0]; pos1, _ = p.multiplyTransforms(pos_tip_world_m, ori_tip_world, gripper_offset1, [0,0,0,1])
                           p.resetBasePositionAndOrientation(my_robot_bodies[-2], pos1, ori_tip_world)
                           gripper_offset2 = [0,-0.01, 0]; pos2, _ = p.multiplyTransforms(pos_tip_world_m, ori_tip_world, gripper_offset2, [0,0,0,1])
                           p.resetBasePositionAndOrientation(my_robot_bodies[-1], pos2, ori_tip_world)
                    except Exception as e:
                        if frame_count % 120 == 0: print(f"警告: 更新末端或记录位置时出错 - {e}")
                        pos_tip_world_m = np.array([np.nan, np.nan, np.nan]) # 出错时记录 NaN
                # 处理 ODE 求解失败的情况
            elif sol is None:
                 if frame_count % 120 == 0: print(f"警告: 第 {current_row_index} 帧 ODE 求解失败。")
                 pos_tip_world_m = np.array([np.nan, np.nan, np.nan]) # 记录 NaN

            # --- 记录数据 (可选增加计算出的输入) ---
            current_cblen_mm = absolute_lengths_mm[current_row_index]
            results_data['cblen1_mm'].append(current_cblen_mm[0])
            results_data['cblen2_mm'].append(current_cblen_mm[1])
            results_data['cblen3_mm'].append(current_cblen_mm[2])
            current_real_xyz_mm = real_xyz_mm[current_row_index]
            results_data['X_real_mm'].append(current_real_xyz_mm[0])
            results_data['Y_real_mm'].append(current_real_xyz_mm[1])
            results_data['Z_real_mm'].append(current_real_xyz_mm[2])
            # 对仿真结果应用坐标变换
            sim_x_mm = pos_tip_world_m[0] * -1000.0 if not np.isnan(pos_tip_world_m[0]) else np.nan
            sim_y_mm = pos_tip_world_m[1] * 1000.0 if not np.isnan(pos_tip_world_m[1]) else np.nan # <<< 检查此变换是否正确
            sim_z_mm = pos_tip_world_m[2] * 1000.0 - 480 if not np.isnan(pos_tip_world_m[2]) else np.nan # <<< 检查此变换是否正确 (基座高度?)
            results_data['sim_X_mm'].append(sim_x_mm)
            results_data['sim_Y_mm'].append(sim_y_mm)
            results_data['sim_Z_mm'].append(sim_z_mm)
            # 可选: 记录计算出的输入值
            results_data['calc_ux'].append(ux)
            results_data['calc_uy'].append(uy)
            results_data['calc_epsilon'].append(epsilon)

            # --- 仿真步进 ---
            p.stepSimulation()
            time.sleep(simulationStepTime) # 如果需要实时观察，可以调整延时
            frame_count += 1
            current_row_index += 1

    # --- 异常处理和退出清理 (保持不变) ---
    except KeyboardInterrupt: print("\n[信息] 检测到键盘中断 (Ctrl+C)...")
    except p.error as e: print(f"\n[错误] PyBullet 发生错误: {e}")
    except Exception as e: print(f"\n[错误] 发生意外错误: {e}"); import traceback; traceback.print_exc()
    finally:
        print("[信息] 断开 PyBullet 连接。")
        if 'physicsClientId' in locals() and p.isConnected(physicsClientId):
            try: p.disconnect(physicsClientId)
            except p.error: pass

    # --- 保存结果 (如果需要，更新列顺序) ---
    print("\n--- 保存仿真结果 ---")
    try:
        if len(results_data['cblen1_mm']) > 0:
            # 定义输出列的顺序
            output_column_order = [
                'cblen1_mm', 'cblen2_mm', 'cblen3_mm',
                'X_real_mm', 'Y_real_mm', 'Z_real_mm',
                'sim_X_mm', 'sim_Y_mm', 'sim_Z_mm',
                'calc_ux', 'calc_uy', 'calc_epsilon' # 添加了可选列
            ]
            output_results_df = pd.DataFrame(results_data)
            # 确保列按指定顺序排列
            output_results_df = output_results_df[output_column_order]
            print(f"[信息] 正在将 {len(output_results_df)} 条结果 ({len(output_column_order)} 列) 保存到: {OUTPUT_RESULTS_PATH}")
            output_results_df.to_excel(OUTPUT_RESULTS_PATH, index=False, engine='openpyxl')
            print(f"[成功] 结果已保存至: {os.path.abspath(OUTPUT_RESULTS_PATH)}")
        else:
            print("[警告] 没有记录到任何仿真结果，未生成输出文件。")
    except Exception as e: print(f"[错误] 保存结果到 Excel 文件时出错: {e}")

    print("--- 仿真结束 ---")