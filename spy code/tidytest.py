# -*- coding: utf-8 -*-
# ==============================================================================
# 脚本功能: 
# 使用 SoftRobotBasicEnvironment 进行 PyBullet 设置和可视化管理，
# 但保留手动 ODE 求解以使用自定义参数（轴向应变、刚度），
# 读取包含绝对绳长和真实坐标的文件，驱动仿真，保存结果。
# ==============================================================================

# 导入所需的库
import sys
import os
import numpy as np
# 导入 pybullet 但可能主要通过 env.bullet 访问
import pybullet as p 
import pybullet_data
import time
import math
from pprint import pprint
import pandas as pd 

# --- 路径设置 ---
softmanisim_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if softmanisim_path not in sys.path:
    sys.path.append(softmanisim_path)
    print(f"[调试] 已添加路径: {softmanisim_path}")

# --- 导入所需的类 ---
# 导入环境类
try:
    # *** 请确保这个路径是正确的！*** # 它可能在 'pybullet_env' 或其他地方，例如 'SoftManiSim.pybullet_env.BasicEnvironment'
    # 假设它在 pybullet_env 目录下
    from pybullet_env.BasicEnvironment import SoftRobotBasicEnvironment 
    print("[调试] 成功导入 SoftRobotBasicEnvironment")
except ImportError as e:
    print(f"[错误] 无法导入 SoftRobotBasicEnvironment: {e}")
    print("[提示] 请检查 SoftRobotBasicEnvironment 类的确切位置和导入路径。")
    sys.exit(1)

# 导入你修改过的 ODE 类
from visualizer.visualizer import ODE
print("[调试] 成功导入 ODE 类")


# --- 辅助函数 & 曲率计算 (保持不变) ---
def calculate_orientation(point1, point2):
    # ... (代码同前) ...
    diff = np.array(point2) - np.array(point1); norm_diff = np.linalg.norm(diff)
    if norm_diff < 1e-6: return p.getQuaternionFromEuler([0,0,0]), [0,0,0]
    if np.linalg.norm(diff[:2]) < 1e-6: yaw = 0
    else: yaw = math.atan2(diff[1], diff[0])
    pitch = math.atan2(-diff[2], math.sqrt(diff[0]**2 + diff[1]**2)); roll = 0
    return p.getQuaternionFromEuler([roll, pitch, yaw]), [roll, pitch, yaw]


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

def convert_curvatures_to_ode_action(ux, uy, length_change, d, L0_seg):
    # ... (代码同前) ...
    l = L0_seg; action_ode = np.zeros(3); action_ode[0] = length_change
    action_ode[1] = uy * l * d; action_ode[2] = -ux * l * d
    return action_ode
# -----------------------------

# --- 主程序 ---
if __name__ == "__main__":

    # --- 参数设置 ---
    DATA_FILE_PATH = 'c:/Users/11647/SoftManiSim//code/Modeling/Processed_Data3w_20250318.xlsx' 
    SHEET_NAME = 'Sheet1'
    output_dir = os.path.dirname(DATA_FILE_PATH) if os.path.dirname(DATA_FILE_PATH) else '.'
    OUTPUT_RESULTS_PATH = os.path.join(output_dir, 'Sim2Real_Results_EnvSimplified.xlsx') 

    # 机器人参数 (需要与 Environment 匹配或传入)
    num_cables = 3; 
    cable_distance = 0.004; 
    initial_length = 0.12; 
    number_of_segment = 1 # <<< 确保与 Env 设置一致
    L0_seg = initial_length / number_of_segment
    
    # ODE 和刚度调试参数
    axial_strain_coefficient = -0.05 # (需要调试!)
    EFFECTIVE_BENDING_STIFFNESS = 0.5  # (需要调试!)

    # 可视化参数 (传入 Environment)
    body_color = [1, 0.0, 0.0, 1]; 
    head_color = [0.0, 0.0, 0.75, 1]
    body_sphere_radius = 0.02; 
    number_of_sphere = 30 # <<< 确保这个值与下面 Env 初始化一致
    
    simulationStepTime = 0.01 # 仿真步长时间

    # --- 使用 SoftRobotBasicEnvironment 初始化 PyBullet 和基础可视化 ---
    print("--- 初始化 SoftRobotBasicEnvironment ---")
    # 确保传入的参数与你的脚本逻辑一致
    env = SoftRobotBasicEnvironment(
        gui=True, 
        number_of_sphere=number_of_sphere,
        number_of_segment=number_of_segment, # 确保 Env 知道段数（虽然我们只用一段）
        body_color=body_color,
        head_color=head_color,
        body_sphere_radius=body_sphere_radius
        # 注意: Env 会创建自己的 self._ode 实例，我们不用它
    )
    env.bullet.setTimeStep(simulationStepTime) # 覆盖 Env 可能的默认步长
    # env.bullet 就是 PyBullet 实例 (p)
    # env 已经处理了 connect, gravity, plane, camera 基础设置
    print(f"PyBullet 已通过 Environment 连接")

    # --- 初始化我们自己的、带参数的 ODE 对象 ---
    print("--- 初始化自定义 ODE 对象 ---"); 
    my_ode = ODE(initial_length_m=initial_length, 
                 cable_distance_m=cable_distance,
                 axial_coupling_coefficient=axial_strain_coefficient) # 传入自定义参数
    # 确认参数设置 (假设 ODE 类有 k_strain 属性)
    if hasattr(my_ode, 'k_strain'):
         print(f"自定义 ODE 已初始化 L0={my_ode.l0:.4f}m, d={my_ode.d:.4f}m, StrainCoeff={my_ode.k_strain}")
    else:
         print(f"自定义 ODE 已初始化 L0={my_ode.l0:.4f}m, d={my_ode.d:.4f}m (无应变系数)")

    # --- 加载数据 ---
    print(f"[信息] 加载数据文件: {DATA_FILE_PATH}")
    # ... (加载数据的代码保持不变) ...
    if not os.path.exists(DATA_FILE_PATH): raise FileNotFoundError(f"错误: 文件未找到: {DATA_FILE_PATH}") 
    df_input = pd.read_excel(DATA_FILE_PATH, sheet_name=SHEET_NAME, engine='openpyxl')
    print(f"[成功] 已加载数据 {len(df_input)} 行。")
    required_cols = ['cblen1', 'cblen2', 'cblen3', 'X', 'Y', 'Z']
    if not all(col in df_input.columns for col in required_cols):
        missing_cols = [col for col in required_cols if col not in df_input.columns]
        raise ValueError(f"错误: 文件缺少必需列: {missing_cols}")
    absolute_lengths_mm = df_input[['cblen1', 'cblen2', 'cblen3']].values
    real_xyz_mm = df_input[['X', 'Y', 'Z']].values
    if len(absolute_lengths_mm) == 0: raise ValueError("错误: 数据文件为空。")
    L0_cables_mm = absolute_lengths_mm[0]; print(f"[假设] 使用第一行 L0(mm): {L0_cables_mm}")
    absolute_lengths_m = absolute_lengths_mm / 1000.0; L0_cables_m = L0_cables_mm / 1000.0
    dl_sequence_m = absolute_lengths_m - L0_cables_m; print(f"[信息] 已计算 {len(dl_sequence_m)} 行 dl (m)。")
    if dl_sequence_m is None or len(dl_sequence_m) == 0: raise ValueError("[错误] 未能计算 dl 序列或序列为空。")


    # --- 获取由 Environment 创建的可视化物体 ID ---
    print("--- 获取 Environment 创建的机器人 Body ID ---")
    if hasattr(env, '_robot_bodies') and isinstance(env._robot_bodies, list) and len(env._robot_bodies) > 0:
        # !!! 直接访问 env 的保护成员 _robot_bodies !!!
        robot_body_ids = env._robot_bodies 
        num_total_bodies = len(robot_body_ids)
        # 根据 env.create_robot 的逻辑，最后3个是末端
        num_tip_bodies = 3 
        num_body_spheres = num_total_bodies - num_tip_bodies
        print(f"成功获取 {num_body_spheres} 个身体球体 + {num_tip_bodies} 个末端物体 ID。")
        # 检查球体数量是否匹配
        if num_body_spheres != number_of_sphere:
             print(f"[警告] 脚本设置的球体数({number_of_sphere})与环境创建的不同({num_body_spheres})，可视化可能不完整或出错。")
             # 可能需要调整 number_of_sphere 参数以匹配环境，或者修改环境代码
    else:
        raise RuntimeError("错误: 无法从 Environment 获取 _robot_bodies 列表，或列表为空。请检查 SoftRobotBasicEnvironment 的实现。")

    # --- 设置基座姿态 ---
    # (保持手动设置，因为 Env 的 move_robot_ori/calc_tip_pos 不被使用)
    base_pos = np.array([0, 0, 0.6]); 
    base_ori_euler = np.array([-math.pi / 2.0, 0, 0]); 
    base_ori = env.bullet.getQuaternionFromEuler(base_ori_euler) # 使用 env.bullet
    print(f"[设置] 基座: Pos={base_pos}, Ori(Euler)={base_ori_euler}")

    # --- 计算并应用自定义 ODE 的初始形状到 Env 创建的物体上 ---
    # (这一步确保可视化物体与我们自定义 ODE 的初始状态对齐)
    print("--- 计算并应用自定义 ODE 的初始形状 ---"); 
    act0_segment = np.zeros(3); 
    my_ode._reset_y0(); 
    my_ode.updateAction(act0_segment) 
    sol0 = my_ode.odeStepFull()
    if sol0 is None or sol0.shape[1] < 3: 
        raise RuntimeError("错误: 初始 ODE 求解失败或点数不足(<3)。")
        
    idx0 = np.linspace(0, sol0.shape[1] - 1, number_of_sphere, dtype=int); # 使用 number_of_sphere
    positions0_local = [(sol0[0, i], sol0[2, i], sol0[1, i]) for i in idx0] # YZ Swap
    
    # 更新身体部分
    spheres_to_update_init = min(num_body_spheres, len(positions0_local))
    for i in range(spheres_to_update_init): 
        pos_world, ori_world = env.bullet.multiplyTransforms(base_pos, base_ori, positions0_local[i], [0,0,0,1]); 
        env.bullet.resetBasePositionAndOrientation(robot_body_ids[i], pos_world, ori_world)
        
    # 更新末端部分 (假设末端物体是最后三个)
    if len(positions0_local) >= 3 and num_total_bodies >= num_tip_bodies:
        ori_tip_local, _ = calculate_orientation(positions0_local[-3], positions0_local[-1]); 
        pos_tip_world, ori_tip_world = env.bullet.multiplyTransforms(base_pos, base_ori, positions0_local[-1], ori_tip_local)
        
        env.bullet.resetBasePositionAndOrientation(robot_body_ids[-3], pos_tip_world, ori_tip_world)
        gripper_offset1 = [0, 0.01, 0]; pos1, _ = env.bullet.multiplyTransforms(pos_tip_world, ori_tip_world, gripper_offset1, [0,0,0,1])
        env.bullet.resetBasePositionAndOrientation(robot_body_ids[-2], pos1, ori_tip_world)
        gripper_offset2 = [0,-0.01, 0]; pos2, _ = env.bullet.multiplyTransforms(pos_tip_world, ori_tip_world, gripper_offset2, [0,0,0,1])
        env.bullet.resetBasePositionAndOrientation(robot_body_ids[-1], pos2, ori_tip_world)
    print(f"自定义 ODE 初始形状已应用到可视化物体。")


    print("--- 初始化完成, 开始主仿真循环 ---")

    # --- 初始化结果列表 ---
    results_data = { 'cblen1_mm': [], 'cblen2_mm': [], 'cblen3_mm': [], 'X_real_mm': [], 'Y_real_mm': [], 'Z_real_mm': [], 'sim_X_mm': [], 'sim_Y_mm': [], 'sim_Z_mm': [] }

    # --- 主循环 ---
    play_mode = 'single'; current_row_index = 0; num_data_rows = len(dl_sequence_m)
    print(f"将按顺序应用 {num_data_rows} 组绳长变化量数据。播放模式: {play_mode}")
    frame_count = 0; simulation_running = True
    
    # --- <<< 保留 try...finally 用于确保断开连接 >>> ---
    try: 
        while simulation_running:
            if current_row_index >= num_data_rows:
                print("[信息] 所有数据已处理完毕。")
                simulation_running = False; 
                continue 

            # 使用 env.bullet 检查连接
            if not env.bullet.isConnected(env.bullet.getClient(0)): 
                print("[警告] PyBullet 连接已断开，提前终止。")
                simulation_running = False; 
                continue

            dl_segment = dl_sequence_m[current_row_index]

            # --- <<< 核心计算：使用我们自己的 ODE 对象 >>> ---
            my_ode._reset_y0()
            ux, uy = calculate_curvatures_with_simple_stiffness(dl_segment, cable_distance, L0_seg, EFFECTIVE_BENDING_STIFFNESS, num_cables)
            action_ode_segment = convert_curvatures_to_ode_action(ux, uy, 0.0, cable_distance, L0_seg)
            my_ode.updateAction(action_ode_segment) # updateAction 内部会计算应变 epsilon
            sol = my_ode.odeStepFull() # 使用我们自己的 ODE 求解

            # --- <<< 可视化更新：手动更新 Env 创建的物体 >>> ---
            pos_tip_world_m = np.array([np.nan, np.nan, np.nan]) 
            if sol is not None and sol.shape[1] >= 3:
                # 从我们自己的 ODE 解 sol 中获取点
                idx = np.linspace(0, sol.shape[1] -1, number_of_sphere, dtype=int) 
                positions_local = [(sol[0, i], sol[2, i], sol[1, i]) for i in idx] # YZ 交换
                
                num_points_available = len(positions_local)
                num_spheres_to_update = min(num_body_spheres, num_points_available)

                # 更新身体球体 (使用 robot_body_ids)
                for i in range(num_spheres_to_update): 
                    pos_world, ori_world = env.bullet.multiplyTransforms(base_pos, base_ori, positions_local[i], [0,0,0,1]); 
                    env.bullet.resetBasePositionAndOrientation(robot_body_ids[i], pos_world, ori_world)
                
                # 获取并更新末端 (使用 robot_body_ids)
                if num_points_available >= 3 and num_total_bodies >= num_tip_bodies:
                    ori_tip_local, _ = calculate_orientation(positions_local[-3], positions_local[-1]); 
                    pos_tip_world_tuple, ori_tip_world = env.bullet.multiplyTransforms(base_pos, base_ori, positions_local[-1], ori_tip_local)
                    pos_tip_world_m = np.array(pos_tip_world_tuple) # 获取末端世界坐标 (m)
                    
                    # 更新末端可视化
                    env.bullet.resetBasePositionAndOrientation(robot_body_ids[-3], pos_tip_world_m, ori_tip_world)
                    gripper_offset1 = [0, 0.01, 0]; pos1, _ = env.bullet.multiplyTransforms(pos_tip_world_m, ori_tip_world, gripper_offset1, [0,0,0,1])
                    env.bullet.resetBasePositionAndOrientation(robot_body_ids[-2], pos1, ori_tip_world)
                    gripper_offset2 = [0,-0.01, 0]; pos2, _ = env.bullet.multiplyTransforms(pos_tip_world_m, ori_tip_world, gripper_offset2, [0,0,0,1])
                    env.bullet.resetBasePositionAndOrientation(robot_body_ids[-1], pos2, ori_tip_world)
                
            elif sol is None or sol.shape[1] < 3:
                 if frame_count % 120 == 0: print(f"警告: 第 {current_row_index} 帧无法获取有效形状。")
                 # pos_tip_world_m 保持 NaN
            
            # --- 记录数据 (保持不变) ---
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

            # --- <<< 步进仿真 (仍然需要手动调用) >>> ---
            env.bullet.stepSimulation() 
            time.sleep(simulationStepTime) 
            frame_count += 1
            current_row_index += 1
    
    # --- <<< finally 块使用 env.bullet.disconnect() >>> ---
    finally:
        print("[信息] 仿真循环结束或出错，断开 PyBullet 连接。")
        # 检查 env 是否已定义且 PyBullet 仍然连接
        if 'env' in locals() and hasattr(env, 'bullet') and env.bullet.isConnected(env.bullet.getClient(0)): 
            env.bullet.disconnect()
            print("[调试] 已调用 env.bullet.disconnect()")
        else:
             print("[调试] 未调用 env.bullet.disconnect() (env 未定义或未连接)")


    # --- 保存结果 (保持不变) ---
    print("\n--- 保存仿真结果 ---")
    if len(results_data['cblen1_mm']) > 0:
        # ... (保存逻辑) ...
        output_column_order = ['cblen1_mm', 'cblen2_mm', 'cblen3_mm', 'X_real_mm', 'Y_real_mm', 'Z_real_mm', 'sim_X_mm', 'sim_Y_mm', 'sim_Z_mm']
        output_results_df = pd.DataFrame(results_data)
        cols_to_order = [col for col in output_column_order if col in output_results_df.columns]
        output_results_df = output_results_df[cols_to_order] 
        print(f"[信息] 正在将 {len(output_results_df)} 条结果保存到: {OUTPUT_RESULTS_PATH}")
        output_results_df.to_excel(OUTPUT_RESULTS_PATH, index=False, engine='openpyxl')
        print(f"[成功] 结果已保存至: {os.path.abspath(OUTPUT_RESULTS_PATH)}")
    else:
        print("[警告] 没有记录到任何仿真结果，未生成输出文件。")

    print("--- 脚本执行完毕 ---")