# -*- coding: utf-8 -*-
# ==============================================================================
# 脚本功能:
# 读取包含绝对绳长(...)和真实坐标(...)的文件，
# 驱动 ODE 仿真，对仿真结果应用 EMA 滤波，记录滤波后的仿真坐标(...)，
# 将输入绳长、真实坐标(mm)、滤波后的仿真坐标(mm)保存到新的 Excel 文件。
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

# 动态添加 SoftManiSim 文件夹到 sys.path
# ... (保持不变) ...
try:
    softmanisim_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if softmanisim_path not in sys.path: sys.path.append(softmanisim_path); print(f"[调试] 已添加路径: {softmanisim_path}")
except NameError: print("[警告] 无法自动确定项目根目录，请确保 visualizer 模块在 Python 路径中。")

# --- 检查 visualizer 模块导入 ---
try:
    # 使用你代码中指定的 ODE 类路径
    from visualizer.visualizer import ODE
    print("[调试] 成功从 'visualizer.visualizer' 导入 ODE")
except ImportError as e: print(f"[错误] 无法导入 ODE 类: {e}"); print("[调试] 当前 sys.path:", sys.path); sys.exit("请检查 visualizer 模块和路径。")


# --- 辅助函数 (保持不变) ---
def calculate_orientation(point1, point2):
    """根据两点计算方向 (四元数)"""
    # ... (保持不变) ...
    diff = np.array(point2) - np.array(point1); norm_diff = np.linalg.norm(diff)
    if norm_diff < 1e-6: return p.getQuaternionFromEuler([0,0,0]), [0,0,0]
    if np.linalg.norm(diff[:2]) < 1e-6: yaw = 0
    else: yaw = math.atan2(diff[1], diff[0])
    pitch = math.atan2(-diff[2], math.sqrt(diff[0]**2 + diff[1]**2)); roll = 0
    return p.getQuaternionFromEuler([roll, pitch, yaw]), [roll, pitch, yaw]

# --- 绳长变化量 -> 曲率 的转换函数 (保持不变) ---
def calculate_curvatures_from_dl(dl_segment, d, L0_seg, num_cables=3):
    """根据绳长变化量计算曲率 ux, uy (基于论文反解)。"""
    # ... (保持不变) ...
    ux = 0.0; uy = 0.0
    if abs(d) < 1e-9: print("警告: 绳索半径 d 接近于零。"); return ux, uy
    if len(dl_segment) != num_cables: print(f"警告: dl_segment 长度 {len(dl_segment)} != num_cables {num_cables}"); return ux,uy
    if num_cables == 3:
        dl1, dl2, dl3 = dl_segment[0], dl_segment[1], dl_segment[2]
        uy = -dl1 / d # 计算 uy
        denominator_ux = d * math.sqrt(3.0)
        if abs(denominator_ux) > 1e-9: ux = (dl3 - dl2) / denominator_ux # 计算 ux
        else: ux = 0.0; print("警告: 计算 ux 时分母接近零。")
    else: print(f"错误: 此处仅实现了3根绳索的反解。"); return 0.0, 0.0
    return ux, uy

# --- 曲率 -> ODE抽象动作 的转换函数 (保持不变) ---
def convert_curvatures_to_ode_action(ux, uy, length_change, d, L0_seg):
    """将曲率转换为 ODE action。"""
    # ... (保持不变) ...
    l = L0_seg; action_ode = np.zeros(3); action_ode[0] = length_change
    action_ode[1] = uy * l * d; action_ode[2] = -ux * l * d
    return action_ode

# --- 主程序 ---
if __name__ == "__main__":

    # --- 数据文件路径和参数 ---
    print("--- 设置参数 & 加载数据 ---")
    DATA_FILE_PATH = 'c:/Users/11647/Desktop/data/circle2.xlsx'
    SHEET_NAME = 'Sheet1'
    output_dir = os.path.dirname(DATA_FILE_PATH) if os.path.dirname(DATA_FILE_PATH) else '.'
    # <<< 修改输出文件名以反映滤波效果 >>>
    OUTPUT_RESULTS_PATH = os.path.join(output_dir, 'changznewcircle2_with_EMA_filter.xlsx')

    # ... (机器人物理参数，k_strain, AXIAL_ACTION_SCALE 保持你之前的设定) ...
    num_cables = 3; cable_distance = 0.004; initial_length = 0.12; number_of_segment = 1
    L0_seg = initial_length / number_of_segment
    axial_strain_coefficient = -2000
    AXIAL_ACTION_SCALE = 0.8 # 使用你上一版的值

    # --- <<< 新增：定义 EMA 平滑因子 (需要调试!) >>> ---
    # --- 0 < ALPHA <= 1 ---
    # --- 值越小，平滑/滞后越多，越能解决“超前”问题 ---
    # --- 但过小的值会使响应过于迟钝 ---
    # --- 尝试从 0.8, 0.6, 0.4, 0.2 等开始 ---
    ALPHA = 0.6 # <<< 示例值，需要调试！

    # ... (可视化参数) ...
    body_color = [1, 0.0, 0.0, 1]; head_color = [0.0, 0.0, 0.75, 1]
    body_sphere_radius = 0.02; number_of_sphere = 30
    my_sphere_radius = body_sphere_radius; my_number_of_sphere = number_of_sphere; my_head_color = head_color

    # --- 加载并处理绳长和真实坐标数据 ---
    # ... (加载代码, dl 计算逻辑保持不变) ...
    print(f"[信息] 正在加载数据文件: {DATA_FILE_PATH}")
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
        # 沿用基于第一行计算 dl 的逻辑
        if len(absolute_lengths_mm) > 0:
            L0_cables_mm = absolute_lengths_mm[0]; print(f"[假设] 使用第一行 L0(mm): {L0_cables_mm}")
            absolute_lengths_m = absolute_lengths_mm / 1000.0; L0_cables_m = L0_cables_mm / 1000.0
            dl_sequence_m = absolute_lengths_m - L0_cables_m; print(f"[信息] 已计算 {len(dl_sequence_m)} 行 dl (m)。")
        else: raise ValueError("错误: 数据文件为空。")
    except Exception as e: print(f"[错误] 加载或处理数据时出错: {e}"); sys.exit(1)
    if dl_sequence_m is None: print("[错误] 未能计算 dl 序列。"); sys.exit(1)


    # --- PyBullet 初始化 ---
    # ... (初始化代码) ...
    print("--- 初始化 PyBullet ---");
    simulationStepTime = 0.001;
    physicsClientId = -1
    try: physicsClientId = p.connect(p.GUI); # ... (连接检查) ...
    except Exception as e: print(f"连接 PyBullet 出错: {e}"); sys.exit(1)
    p.setAdditionalSearchPath(pybullet_data.getDataPath()); p.setGravity(0, 0, -9.81); p.setTimeStep(simulationStepTime)
    try: planeId = p.loadURDF("plane.urdf"); print(f"加载 plane.urdf, ID: {planeId}")
    except p.error as e: print(f"加载 plane.urdf 出错: {e}"); p.disconnect(physicsClientId); sys.exit(1)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.resetDebugVisualizerCamera(cameraDistance=0.8, cameraYaw=45, cameraPitch=-20, cameraTargetPosition=[0, 0, 0.3])
    print("[信息] 已设置相机视角。")

    # --- 初始化 ODE 对象 ---
    # ... (ODE 初始化代码) ...
    print("--- 初始化 ODE 对象 ---");
    my_ode = ODE(initial_length_m=initial_length,
                 cable_distance_m=cable_distance,
                 ode_step_ds=simulationStepTime,
                 axial_coupling_coefficient=axial_strain_coefficient)
    print(f"ODE 已初始化 L0={my_ode.l0:.4f}m, d={my_ode.d:.4f}m, k_strain={my_ode.k_strain}")

    # --- 计算初始形态 (dl=0) ---
    # ... (计算 sol0 代码) ...
    print("--- 计算初始形状 (dl=0) ---");
    act0_segment = np.zeros(3);
    my_ode._reset_y0();
    my_ode.updateAction(act0_segment)
    sol0 = my_ode.odeStepFull()
    if sol0 is None or sol0.shape[1] < 3: print("错误: 初始 ODE 求解失败或点数不足(<3)。"); p.disconnect(physicsClientId); sys.exit(1)
    print(f"初始形状计算完成。")


    # --- 设置基座和创建 PyBullet 物体 ---
    # ... (创建物体代码) ...
    base_pos = np.array([0, 0, 0.6]); base_ori_euler = np.array([-math.pi / 2.0, 0, 0]); base_ori = p.getQuaternionFromEuler(base_ori_euler)
    radius = my_sphere_radius
    try: shape = p.createCollisionShape(p.GEOM_SPHERE, radius=radius); visualShapeId = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=body_color); visualShapeId_tip_body = p.createVisualShape(p.GEOM_SPHERE, radius=radius+0.0025, rgbaColor=my_head_color); visualShapeId_tip_gripper = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.002, 0.01, 0.002], rgbaColor=head_color)
    except p.error as e: print(f"创建形状出错: {e}"); p.disconnect(physicsClientId); sys.exit(1)
    if sol0.shape[1] < 3: print("错误: 初始解点数不足3。"); p.disconnect(physicsClientId); sys.exit(1)
    idx0 = np.linspace(0, sol0.shape[1] - 1, my_number_of_sphere, dtype=int); positions0_local = [(sol0[0, i], sol0[2, i], sol0[1, i]) for i in idx0]
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

    # --- 初始化用于存储结果的字典 (保持不变) ---
    results_data = {
        'cblen1_mm': [], 'cblen2_mm': [], 'cblen3_mm': [],
        'X_real_mm': [], 'Y_real_mm': [], 'Z_real_mm': [],
        'sim_X_mm': [], 'sim_Y_mm': [], 'sim_Z_mm': []
    }

    # --- 主循环 ---
    play_mode = 'single';
    # current_row_index = 0; # 使用 for 循环索引
    num_data_rows = len(dl_sequence_m)
    if num_data_rows == 0: print("[错误] 没有可用的绳长变化量数据。"); p.disconnect(physicsClientId); sys.exit(1)
    print(f"将按顺序应用 {num_data_rows} 组绳长变化量数据。播放模式: {play_mode}")
    frame_count = 0; last_print_time = time.time(); simulation_running = True

    # --- <<< 新增：初始化 EMA 状态列表 >>> ---
    sim_x_ema_history = []
    sim_y_ema_history = []
    sim_z_ema_history = []

    try:
        # 使用 for 循环遍历所有时间步
        for current_row_index in range(num_data_rows):
            if not p.isConnected(physicsClientId):
                print("[警告] PyBullet 连接已断开。")
                simulation_running = False;
                break
            if not simulation_running: # 如果需要外部停止逻辑
                break

            dl_segment = dl_sequence_m[current_row_index]

            # --- 计算当前步的“原始”仿真形态 ---
            my_ode._reset_y0()
            ux, uy = calculate_curvatures_from_dl(dl_segment, cable_distance, L0_seg, num_cables)
            avg_dl = np.mean(dl_segment)
            commanded_length_change = avg_dl * AXIAL_ACTION_SCALE
            action_ode_segment = convert_curvatures_to_ode_action(
                ux, uy, commanded_length_change, cable_distance, L0_seg
            )
            my_ode.updateAction(action_ode_segment)
            sol = my_ode.odeStepFull()

            # --- 获取原始仿真末端位置 ---
            pos_tip_world_m = np.array([np.nan, np.nan, np.nan])
            ori_tip_world = p.getQuaternionFromEuler([0,0,0]) # 默认姿态
            if sol is not None and sol.shape[1] >= 3:
                try:
                    final_pos_local = (sol[0, -1], sol[2, -1], sol[1, -1]) # YZ 交换
                    idx = np.linspace(0, sol.shape[1] -1, min(my_number_of_sphere, sol.shape[1]), dtype=int)
                    positions_local = [(sol[0, i], sol[2, i], sol[1, i]) for i in idx]
                    ori_tip_local = p.getQuaternionFromEuler([0,0,0])
                    if len(positions_local) >= 3:
                        ori_tip_local, _ = calculate_orientation(positions_local[-3], positions_local[-1])

                    pos_tip_world_tuple, ori_tip_world_calc = p.multiplyTransforms(base_pos, base_ori, final_pos_local, ori_tip_local)
                    pos_tip_world_m = np.array(pos_tip_world_tuple)
                    ori_tip_world = ori_tip_world_calc # 更新姿态
                except Exception as e:
                     if frame_count % 120 == 0: print(f"警告: 处理仿真结果时出错 - {e}")
                     pos_tip_world_m = np.array([np.nan, np.nan, np.nan]) # 出错时结果无效

            # --- 计算当前步原始仿真坐标 (mm) ---
            current_sim_x_mm = pos_tip_world_m[0] * 1000.0 if not np.isnan(pos_tip_world_m[0]) else np.nan
            current_sim_y_mm = pos_tip_world_m[1] * -1000.0 if not np.isnan(pos_tip_world_m[1]) else np.nan # 检查变换
            current_sim_z_mm = pos_tip_world_m[2] * 1000.0 - 480 if not np.isnan(pos_tip_world_m[2]) else np.nan # 检查变换

            # --- <<< 新增：应用 EMA 滤波 >>> ---
            if current_row_index == 0 or np.isnan(current_sim_x_mm) or np.isnan(current_sim_y_mm) or np.isnan(current_sim_z_mm):
                # 第一步或遇到 NaN 时，直接使用当前值（或保持 NaN）
                filtered_x = current_sim_x_mm
                filtered_y = current_sim_y_mm
                filtered_z = current_sim_z_mm
            else:
                # 获取上一步的滤波值，如果上一步是 NaN，则使用当前原始值作为替代
                prev_x = sim_x_ema_history[-1] if len(sim_x_ema_history) > 0 and not np.isnan(sim_x_ema_history[-1]) else current_sim_x_mm
                prev_y = sim_y_ema_history[-1] if len(sim_y_ema_history) > 0 and not np.isnan(sim_y_ema_history[-1]) else current_sim_y_mm
                prev_z = sim_z_ema_history[-1] if len(sim_z_ema_history) > 0 and not np.isnan(sim_z_ema_history[-1]) else current_sim_z_mm

                # EMA 公式
                filtered_x = ALPHA * current_sim_x_mm + (1 - ALPHA) * prev_x
                filtered_y = ALPHA * current_sim_y_mm + (1 - ALPHA) * prev_y
                filtered_z = ALPHA * current_sim_z_mm + (1 - ALPHA) * prev_z

            # --- <<< 新增：记录当前步的滤波值到历史列表 >>> ---
            sim_x_ema_history.append(filtered_x)
            sim_y_ema_history.append(filtered_y)
            sim_z_ema_history.append(filtered_z)

            # --- <<< 修改：将 *滤波后* 的值存入最终结果字典 >>> ---
            results_data['sim_X_mm'].append(filtered_x)
            results_data['sim_Y_mm'].append(filtered_y)
            results_data['sim_Z_mm'].append(filtered_z)

            # --- 记录输入和真实数据 (移到循环外) ---
            # results_data['cblen1_mm'].append(absolute_lengths_mm[current_row_index][0])
            # ...

            # --- 更新 PyBullet 可视化 (可选，使用滤波后的坐标或原始坐标) ---
            # 为了可视化平滑效果，可以用滤波后的值来更新位置
            # 注意：滤波后的值是 mm，需要转回米，并且应用反向坐标变换
            if not np.isnan(filtered_x) and not np.isnan(filtered_y) and not np.isnan(filtered_z):
                vis_pos_m = np.array([
                    filtered_x / 1000.0,
                    filtered_y / -1000.0, # 反向变换
                    (filtered_z + 480) / 1000.0 # 反向变换
                ])
                # 更新末端物体位置 (姿态用原始计算的 ori_tip_world)
                if len(my_robot_bodies) >= 3:
                     try:
                         p.resetBasePositionAndOrientation(my_robot_bodies[-3], vis_pos_m, ori_tip_world)
                         # 更新夹爪位置（可选）
                         gripper_offset1 = [0, 0.01, 0]; pos1, _ = p.multiplyTransforms(vis_pos_m, ori_tip_world, gripper_offset1, [0,0,0,1])
                         p.resetBasePositionAndOrientation(my_robot_bodies[-2], pos1, ori_tip_world)
                         gripper_offset2 = [0,-0.01, 0]; pos2, _ = p.multiplyTransforms(vis_pos_m, ori_tip_world, gripper_offset2, [0,0,0,1])
                         p.resetBasePositionAndOrientation(my_robot_bodies[-1], pos2, ori_tip_world)
                         # 如果需要更新躯干... 但躯干可能需要用原始 sol 数据插值更新
                     except p.error: pass


            # --- 仿真步进 ---
            p.stepSimulation()
            time.sleep(0.0001) # 短暂延时
            frame_count += 1

    # --- 循环结束后 ---
    except KeyboardInterrupt: print("\n[信息] 检测到键盘中断 (Ctrl+C)...")
    except p.error as e: print(f"\n[错误] PyBullet 发生错误: {e}")
    except Exception as e: print(f"\n[错误] 发生意外错误: {e}"); import traceback; traceback.print_exc()
    finally:
        print("[信息] 断开 PyBullet 连接。")
        if 'physicsClientId' in locals() and p.isConnected(physicsClientId):
            try: p.disconnect(physicsClientId)
            except p.error: pass

    # --- 在循环结束后，填充输入和真实数据列 ---
    print("[信息] 正在整理最终结果...")
    results_data['cblen1_mm'] = absolute_lengths_mm[:, 0].tolist()
    results_data['cblen2_mm'] = absolute_lengths_mm[:, 1].tolist()
    results_data['cblen3_mm'] = absolute_lengths_mm[:, 2].tolist()
    results_data['X_real_mm'] = real_xyz_mm[:, 0].tolist()
    results_data['Y_real_mm'] = real_xyz_mm[:, 1].tolist()
    results_data['Z_real_mm'] = real_xyz_mm[:, 2].tolist()
    # sim_X/Y/Z_mm 已经在循环中填充好了

    # --- 保存结果 (保持不变) ---
    print("\n--- 保存仿真结果 ---")
    try:
        if len(results_data['cblen1_mm']) > 0:
            output_column_order = [
                'cblen1_mm', 'cblen2_mm', 'cblen3_mm',
                'X_real_mm', 'Y_real_mm', 'Z_real_mm',
                'sim_X_mm', 'sim_Y_mm', 'sim_Z_mm'
            ]
            output_results_df = pd.DataFrame(results_data)
            # 处理可能因滤波产生的 NaN 值，如果需要保存干净数据，可以在此 dropna
            # output_results_df.dropna(subset=['sim_X_mm', 'sim_Y_mm', 'sim_Z_mm'], inplace=True)
            output_results_df = output_results_df[output_column_order]
            print(f"[信息] 正在将 {len(output_results_df)} 条结果 ({len(output_column_order)} 列) 保存到: {OUTPUT_RESULTS_PATH}")
            output_results_df.to_excel(OUTPUT_RESULTS_PATH, index=False, engine='openpyxl')
            print(f"[成功] 结果已保存至: {os.path.abspath(OUTPUT_RESULTS_PATH)}")
        else:
            print("[警告] 没有记录到任何仿真结果，未生成输出文件。")
    except ImportError as e:
        if 'openpyxl' in str(e).lower():
             print("[错误] 保存 Excel 文件需要 'openpyxl' 库。请运行 'pip install openpyxl' 安装。")
        else:
             print(f"[错误] 缺少必要的库 (pandas): {e}。请运行 'pip install pandas openpyxl'");
        sys.exit(1)
    except KeyError as e: print(f"[错误] 尝试保存结果时发生列名错误: {e}。请检查 'results_data' 字典和 'output_column_order' 列表。")
    except Exception as e: print(f"[错误] 保存结果到 Excel 文件时出错: {e}")

    print("--- 仿真结束 ---")