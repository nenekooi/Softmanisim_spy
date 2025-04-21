# -*- coding: utf-8 -*-
# ==============================================================================
# 脚本功能: (同上)
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

# --- 路径设置 ---
# 假设脚本正常运行，__file__ 是定义的
softmanisim_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if softmanisim_path not in sys.path:
    sys.path.append(softmanisim_path)
    print(f"[调试] 已添加路径: {softmanisim_path}")

# --- 导入 ODE 类 ---
# 如果导入失败，程序会直接抛出 ImportError 并停止
from visualizer.visualizer import ODE
print("[调试] 成功导入 ODE 类")

# --- 辅助函数 ---
def calculate_orientation(point1, point2):
    """根据两点计算方向 (四元数)"""
    diff = np.array(point2) - np.array(point1); norm_diff = np.linalg.norm(diff)
    if norm_diff < 1e-6: return p.getQuaternionFromEuler([0,0,0]), [0,0,0]
    if np.linalg.norm(diff[:2]) < 1e-6: yaw = 0
    else: yaw = math.atan2(diff[1], diff[0])
    pitch = math.atan2(-diff[2], math.sqrt(diff[0]**2 + diff[1]**2)); roll = 0
    return p.getQuaternionFromEuler([roll, pitch, yaw]), [roll, pitch, yaw]

# --- 曲率计算函数 (保留其中一个) ---
# def calculate_curvatures_from_dl(...): # 保留这个或下面的
#     # ... (代码同前) ...

def calculate_curvatures_with_simple_stiffness(dl_segment, d, L0_seg, bending_stiffness_param, num_cables=3):
    # ... (代码同前) ...
    ux = 0.0; uy = 0.0
    if abs(d) < 1e-9: print("警告: d 接近零。"); return ux, uy
    if len(dl_segment) != num_cables: print(f"警告: dl 长度错误"); return ux,uy
    if num_cables != 3: print(f"错误: 仅支持3缆绳"); return 0.0, 0.0
    dl1, dl2, dl3 = dl_segment[0], dl_segment[1], dl_segment[2]
    uy_kinematic = -dl1 / d
    denominator_ux = d * math.sqrt(3.0)
    if abs(denominator_ux) > 1e-9: ux_kinematic = (dl3 - dl2) / denominator_ux
    else: ux_kinematic = 0.0; print("警告: 计算 ux 时分母接近零。")
    stiffness_param = max(0.0, bending_stiffness_param)
    reduction_factor = 1.0 / (1.0 + stiffness_param)
    ux = ux_kinematic * reduction_factor
    uy = uy_kinematic * reduction_factor
    return ux, uy

# --- 曲率 -> ODE action 转换函数 ---
def convert_curvatures_to_ode_action(ux, uy, length_change, d, L0_seg):
    """将曲率转换为 ODE action。"""
    l = L0_seg; action_ode = np.zeros(3); action_ode[0] = length_change
    action_ode[1] = uy * l * d; action_ode[2] = -ux * l * d
    return action_ode

# --- 主程序 ---
if __name__ == "__main__":

    # --- 参数设置 ---
    DATA_FILE_PATH = 'c:/Users/11647/SoftManiSim//code/Modeling/Processed_Data3w_20250318.xlsx' 
    SHEET_NAME = 'Sheet1'
    output_dir = os.path.dirname(DATA_FILE_PATH) if os.path.dirname(DATA_FILE_PATH) else '.'
    OUTPUT_RESULTS_PATH = os.path.join(output_dir, 'Sim2Real_Results_Simplified.xlsx') # 改个名

    num_cables = 3; 
    cable_distance = 0.004; 
    initial_length = 0.12; 
    number_of_segment = 1
    L0_seg = initial_length / number_of_segment
    
    # --- 调试参数 ---
    axial_strain_coefficient = -0.05 # 示例: 修正尺度后的值 (需要重调!)
    EFFECTIVE_BENDING_STIFFNESS = 0.5  # 示例值

    # --- 可视化参数 ---
    body_color = [1, 0.0, 0.0, 1]; head_color = [0.0, 0.0, 0.75, 1]
    body_sphere_radius = 0.02; number_of_sphere = 30
    my_sphere_radius = body_sphere_radius; my_number_of_sphere = number_of_sphere; my_head_color = head_color

    # --- 加载数据 (如果失败会直接报错) ---
    print(f"[信息] 加载数据文件: {DATA_FILE_PATH}")
    if not os.path.exists(DATA_FILE_PATH):
        # 使用 raise 而不是 print + sys.exit
        raise FileNotFoundError(f"错误: 文件未找到: {DATA_FILE_PATH}") 
        
    df_input = pd.read_excel(DATA_FILE_PATH, sheet_name=SHEET_NAME, engine='openpyxl')
    print(f"[成功] 已加载数据 {len(df_input)} 行。")

    required_cols = ['cblen1', 'cblen2', 'cblen3', 'X', 'Y', 'Z']
    if not all(col in df_input.columns for col in required_cols):
        missing_cols = [col for col in required_cols if col not in df_input.columns]
        raise ValueError(f"错误: 文件缺少必需列: {missing_cols}")

    absolute_lengths_mm = df_input[['cblen1', 'cblen2', 'cblen3']].values
    real_xyz_mm = df_input[['X', 'Y', 'Z']].values
    print(f"[信息] 成功提取 {len(absolute_lengths_mm)} 行绳长和真实 XYZ。")

    if len(absolute_lengths_mm) == 0: 
        raise ValueError("错误: 数据文件为空。")
        
    L0_cables_mm = absolute_lengths_mm[0]; print(f"[假设] 使用第一行 L0(mm): {L0_cables_mm}")
    absolute_lengths_m = absolute_lengths_mm / 1000.0; L0_cables_m = L0_cables_mm / 1000.0
    dl_sequence_m = absolute_lengths_m - L0_cables_m; print(f"[信息] 已计算 {len(dl_sequence_m)} 行 dl (m)。")
    
    if dl_sequence_m is None or len(dl_sequence_m) == 0: 
         raise ValueError("[错误] 未能计算 dl 序列或序列为空。")

    # --- PyBullet 初始化 (如果失败会直接报错) ---
    print("--- 初始化 PyBullet ---"); 
    simulationStepTime = 0.01; 
    physicsClientId = p.connect(p.GUI) # 直接连接，失败会抛异常
    print(f"已连接 PyBullet, Client ID: {physicsClientId}")

    p.setAdditionalSearchPath(pybullet_data.getDataPath()); 
    p.setGravity(0, 0, -9.81); 
    p.setTimeStep(simulationStepTime)
    planeId = p.loadURDF("plane.urdf"); print(f"加载 plane.urdf, ID: {planeId}") # 直接加载
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.resetDebugVisualizerCamera(cameraDistance=0.8, cameraYaw=45, cameraPitch=-20, cameraTargetPosition=[0, 0, 0.3])
    print("[信息] 已设置相机视角。")

    # --- 初始化 ODE 对象 ---
    print("--- 初始化 ODE 对象 ---"); 
    my_ode = ODE(initial_length_m=initial_length, 
                 cable_distance_m=cable_distance,
                 axial_coupling_coefficient=axial_strain_coefficient) # 假设 ODE 类已修改
    print(f"ODE 已初始化 L0={my_ode.l0:.4f}m, d={my_ode.d:.4f}m, StrainCoeff={my_ode.k_strain}") # 假设 k_strain 存在

    # --- 计算初始形态 ---
    print("--- 计算初始形状 (dl=0) ---"); 
    act0_segment = np.zeros(3); 
    my_ode._reset_y0(); 
    my_ode.updateAction(act0_segment) # 包含应变计算
    sol0 = my_ode.odeStepFull()
    if sol0 is None or sol0.shape[1] < 3: 
        raise RuntimeError("错误: 初始 ODE 求解失败或点数不足(<3)。")
    print(f"初始形状计算完成。")

    # --- 设置基座和创建形状 (如果失败会报错) ---
    base_pos = np.array([0, 0, 0.6]); 
    base_ori_euler = np.array([-math.pi / 2.0, 0, 0]); 
    base_ori = p.getQuaternionFromEuler(base_ori_euler)
    print(f"[设置] 基座: Pos={base_pos}, Ori(Euler)={base_ori_euler}")
    radius = my_sphere_radius

    print("--- 创建 PyBullet 形状 ---"); 
    shape = p.createCollisionShape(p.GEOM_SPHERE, radius=radius); 
    visualShapeId = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=body_color); 
    visualShapeId_tip_body = p.createVisualShape(p.GEOM_SPHERE, radius=radius+0.0025, rgbaColor=my_head_color); 
    visualShapeId_tip_gripper = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.002, 0.01, 0.002], rgbaColor=head_color)

    # --- 创建 PyBullet 物体 (如果失败会报错) ---
    print("--- 创建 PyBullet 物体 ---"); 
    if sol0.shape[1] < 3: raise RuntimeError("错误: 初始解点数不足3。")
    idx0 = np.linspace(0, sol0.shape[1] - 1, my_number_of_sphere, dtype=int); 
    positions0_local = [(sol0[0, i], sol0[2, i], sol0[1, i]) for i in idx0]
    my_robot_bodies = []; 
    for i, pos_local in enumerate(positions0_local): 
        pos_world, ori_world = p.multiplyTransforms(base_pos, base_ori, pos_local, [0,0,0,1]); 
        my_robot_bodies.append(p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=visualShapeId, basePosition=pos_world, baseOrientation=ori_world))
    num_initial_bodies = len(my_robot_bodies); num_tip_bodies_expected = 3
    if len(positions0_local) >= 3 and num_initial_bodies == my_number_of_sphere:
        ori_tip_local, _ = calculate_orientation(positions0_local[-3], positions0_local[-1]); 
        pos_tip_world, ori_tip_world = p.multiplyTransforms(base_pos, base_ori, positions0_local[-1], ori_tip_local)
        my_robot_bodies.append(p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=visualShapeId_tip_body, basePosition=pos_tip_world, baseOrientation=ori_tip_world))
        gripper_offset1 = [0, 0.01, 0]; pos1, _ = p.multiplyTransforms(pos_tip_world, ori_tip_world, gripper_offset1, [0,0,0,1]); my_robot_bodies.append(p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=visualShapeId_tip_gripper, basePosition=pos1, baseOrientation=ori_tip_world))
        gripper_offset2 = [0,-0.01, 0]; pos2, _ = p.multiplyTransforms(pos_tip_world, ori_tip_world, gripper_offset2, [0,0,0,1]); my_robot_bodies.append(p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=visualShapeId_tip_gripper, basePosition=pos2, baseOrientation=ori_tip_world))
        print(f"总共创建了 {len(my_robot_bodies)} 个物体。")
    else: print("警告: 无法创建末端物体。")

    print("--- 初始化完成, 开始主仿真循环 ---")

    # --- 初始化结果列表 ---
    results_data = { 'cblen1_mm': [], 'cblen2_mm': [], 'cblen3_mm': [], 'X_real_mm': [], 'Y_real_mm': [], 'Z_real_mm': [], 'sim_X_mm': [], 'sim_Y_mm': [], 'sim_Z_mm': [] }

    # --- 主循环 (保留 try...finally 用于断开连接) ---
    play_mode = 'single'; current_row_index = 0; num_data_rows = len(dl_sequence_m)
    print(f"将按顺序应用 {num_data_rows} 组绳长变化量数据。播放模式: {play_mode}")
    frame_count = 0; simulation_running = True
    
    try: # <<< 保留这个 try >>>
        while simulation_running:
            if current_row_index >= num_data_rows:
                print("[信息] 所有数据已处理完毕。")
                simulation_running = False; 
                continue # 使用 continue 而不是 break，以确保 finally 执行

            if not p.isConnected(physicsClientId): 
                print("[警告] PyBullet 连接已断开，提前终止。")
                simulation_running = False; 
                continue

            dl_segment = dl_sequence_m[current_row_index]

            # 计算曲率 (选择一个函数)
            my_ode._reset_y0()
            # --- 选择使用哪个曲率计算函数 ---
            # 选项1: 简单刚度模型
            ux, uy = calculate_curvatures_with_simple_stiffness(dl_segment, cable_distance, L0_seg, EFFECTIVE_BENDING_STIFFNESS, num_cables)
            # 选项2: 原始运动学模型 (如果想对比)
            # ux, uy = calculate_curvatures_from_dl(dl_segment, cable_distance, L0_seg, num_cables)
            # ---------------------------------
            
            action_ode_segment = convert_curvatures_to_ode_action(ux, uy, 0.0, cable_distance, L0_seg)
            my_ode.updateAction(action_ode_segment) # updateAction 内部会计算应变 epsilon
            sol = my_ode.odeStepFull()

            # 更新和记录
            pos_tip_world_m = np.array([np.nan, np.nan, np.nan]) 
            if sol is not None and sol.shape[1] >= 3:
                idx = np.linspace(0, sol.shape[1] -1, my_number_of_sphere, dtype=int)
                positions_local = [(sol[0, i], sol[2, i], sol[1, i]) for i in idx] 
                num_bodies_total = len(my_robot_bodies); num_tip_bodies = 3
                num_body_spheres = num_bodies_total - num_tip_bodies; num_points_available = len(positions_local)
                num_spheres_to_update = min(num_body_spheres, num_points_available)

                for i in range(num_spheres_to_update): # 更新身体
                    pos_world, ori_world = p.multiplyTransforms(base_pos, base_ori, positions_local[i], [0,0,0,1])
                    # 不再捕捉 p.error，如果出错让它自然抛出
                    if i < len(my_robot_bodies): p.resetBasePositionAndOrientation(my_robot_bodies[i], pos_world, ori_world)
                
                if num_points_available >= 3 and num_bodies_total >= num_tip_bodies:
                    # 不再捕捉这里的 Exception，让错误暴露出来
                    ori_tip_local, _ = calculate_orientation(positions_local[-3], positions_local[-1])
                    pos_tip_world_tuple, ori_tip_world = p.multiplyTransforms(base_pos, base_ori, positions_local[-1], ori_tip_local)
                    pos_tip_world_m = np.array(pos_tip_world_tuple)
                    
                    if len(my_robot_bodies) >= num_tip_bodies: # 更新末端可视化
                       p.resetBasePositionAndOrientation(my_robot_bodies[-3], pos_tip_world_m, ori_tip_world)
                       gripper_offset1 = [0, 0.01, 0]; pos1, _ = p.multiplyTransforms(pos_tip_world_m, ori_tip_world, gripper_offset1, [0,0,0,1])
                       p.resetBasePositionAndOrientation(my_robot_bodies[-2], pos1, ori_tip_world)
                       gripper_offset2 = [0,-0.01, 0]; pos2, _ = p.multiplyTransforms(pos_tip_world_m, ori_tip_world, gripper_offset2, [0,0,0,1])
                       p.resetBasePositionAndOrientation(my_robot_bodies[-1], pos2, ori_tip_world)
                
                # 如果上面的 try 块（现在没了）出错，pos_tip_world_m 会保持 NaN

            elif sol is None or sol.shape[1] < 3:
                 if frame_count % 120 == 0: print(f"警告: 第 {current_row_index} 帧无法获取有效形状。")
                 # pos_tip_world_m 保持 NaN
            
            # --- 记录数据 ---
            current_cblen_mm = absolute_lengths_mm[current_row_index]
            results_data['cblen1_mm'].append(current_cblen_mm[0])
            results_data['cblen2_mm'].append(current_cblen_mm[1])
            results_data['cblen3_mm'].append(current_cblen_mm[2])
            current_real_xyz_mm = real_xyz_mm[current_row_index]
            results_data['X_real_mm'].append(current_real_xyz_mm[0])
            results_data['Y_real_mm'].append(current_real_xyz_mm[1])
            results_data['Z_real_mm'].append(current_real_xyz_mm[2])
            sim_x_mm = pos_tip_world_m[0] * 1000.0 if not np.isnan(pos_tip_world_m[0]) else np.nan
            sim_y_mm = pos_tip_world_m[1] * -1000.0 if not np.isnan(pos_tip_world_m[1]) else np.nan # Y轴反转
            sim_z_mm = pos_tip_world_m[2] * 1000.0 - 480 if not np.isnan(pos_tip_world_m[2]) else np.nan # Z轴偏移
            results_data['sim_X_mm'].append(sim_x_mm)
            results_data['sim_Y_mm'].append(sim_y_mm)
            results_data['sim_Z_mm'].append(sim_z_mm)

            # --- 步进与延时 ---
            p.stepSimulation()
            time.sleep(simulationStepTime)
            frame_count += 1
            current_row_index += 1
    
    # <<< 保留 finally 以确保断开连接 >>>
    finally:
        print("[信息] 仿真循环结束或出错，断开 PyBullet 连接。")
        # 检查 physicsClientId 是否已定义且 PyBullet 仍然连接
        if 'physicsClientId' in locals() and isinstance(physicsClientId, int) and p.isConnected(physicsClientId): 
            p.disconnect(physicsClientId)
        elif 'physicsClientId' in locals():
             print("[调试] physicsClientId 存在但 PyBullet 未连接或无效。")
        else:
             print("[调试] physicsClientId 未定义。")


    # --- 保存结果 (如果失败会直接报错) ---
    print("\n--- 保存仿真结果 ---")
    if len(results_data['cblen1_mm']) > 0:
        output_column_order = ['cblen1_mm', 'cblen2_mm', 'cblen3_mm', 'X_real_mm', 'Y_real_mm', 'Z_real_mm', 'sim_X_mm', 'sim_Y_mm', 'sim_Z_mm']
        output_results_df = pd.DataFrame(results_data)
        
        # 检查列是否存在再排序，以防万一
        cols_to_order = [col for col in output_column_order if col in output_results_df.columns]
        output_results_df = output_results_df[cols_to_order] 

        print(f"[信息] 正在将 {len(output_results_df)} 条结果保存到: {OUTPUT_RESULTS_PATH}")
        # 需要安装 openpyxl: pip install openpyxl
        output_results_df.to_excel(OUTPUT_RESULTS_PATH, index=False, engine='openpyxl')
        print(f"[成功] 结果已保存至: {os.path.abspath(OUTPUT_RESULTS_PATH)}")
    else:
        print("[警告] 没有记录到任何仿真结果，未生成输出文件。")

    print("--- 脚本执行完毕 ---")