# -*- coding: utf-8 -*-
# ==============================================================================
# 脚本功能:
# 读取包含绝对绳长(cblen1-3, mm)和真实末端坐标(X,Y,Z, mm)的文件，
# 驱动 ODE 仿真，记录仿真的末端坐标(sim_X/Y/Z_mm)，
# 将输入绳长、真实坐标(mm)、仿真坐标(mm)保存到新的 Excel 文件。
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
import pandas as pd # 用于读写 Excel/CSV

# 动态添加 SoftManiSim 文件夹到 sys.path
# ... (与上一版本相同) ...
try:
    softmanisim_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if softmanisim_path not in sys.path: sys.path.append(softmanisim_path); print(f"[调试] 已添加路径: {softmanisim_path}")
except NameError: print("[警告] 无法自动确定项目根目录，请确保 visualizer 模块在 Python 路径中。")

# --- 检查 visualizer 模块导入 ---
try:
    from visualizer.visualizer import ODE
    print("[调试] 成功从 'visualizer.visualizer' 导入 ODE")
except ImportError as e: print(f"[错误] 无法导入 ODE 类: {e}"); print("[调试] 当前 sys.path:", sys.path); sys.exit("请检查 visualizer 模块和路径。")


# --- 辅助函数 (仅保留需要的) ---
def calculate_orientation(point1, point2):
    """根据两点计算方向 (四元数)"""
    diff = np.array(point2) - np.array(point1); norm_diff = np.linalg.norm(diff)
    if norm_diff < 1e-6: return p.getQuaternionFromEuler([0,0,0]), [0,0,0]
    if np.linalg.norm(diff[:2]) < 1e-6: yaw = 0
    else: yaw = math.atan2(diff[1], diff[0])
    pitch = math.atan2(-diff[2], math.sqrt(diff[0]**2 + diff[1]**2)); roll = 0
    return p.getQuaternionFromEuler([roll, pitch, yaw]), [roll, pitch, yaw]

# --- 绳长变化量 -> 曲率 的转换函数 (基于论文反解) ---
def calculate_curvatures_from_dl(dl_segment, d, L0_seg, num_cables=3):
    """根据绳长变化量计算曲率 ux, uy (基于论文反解)。"""
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
    # max_curvature = 50.0; ux = np.clip(ux, -max_curvature, max_curvature); uy = np.clip(uy, -max_curvature, max_curvature) # 可选限制
    return ux, uy

# --- 曲率 -> ODE抽象动作 的转换函数 ---
def convert_curvatures_to_ode_action(ux, uy, length_change, d, L0_seg):
    """将曲率转换为 ODE action。"""
    l = L0_seg; action_ode = np.zeros(3); action_ode[0] = length_change
    action_ode[1] = uy * l * d; action_ode[2] = -ux * l * d
    return action_ode

# --- 主程序 ---
if __name__ == "__main__":

    # --- 数据文件路径和参数 ---
    print("--- 设置参数 & 加载数据 ---")
    # 1. 输入数据文件路径
    DATA_FILE_PATH = 'c:/Users/11647/Desktop/data/circle2.xlsx' # <<< 确认 Excel 文件路径
    SHEET_NAME = 'Sheet1'
    # 2. 输出结果文件路径
    output_dir = os.path.dirname(DATA_FILE_PATH) if os.path.dirname(DATA_FILE_PATH) else '.'
    # <<< 修改输出文件名，更清晰地反映内容 >>>
    OUTPUT_RESULTS_PATH = os.path.join(output_dir, 'changzcircle2test.xlsx')

    # 3. 机器人物理参数
    num_cables = 3; cable_distance = 0.004; initial_length = 0.12; number_of_segment = 1
    if number_of_segment <= 0: print("错误: 段数必须为正。"); sys.exit(1)
    L0_seg = initial_length / number_of_segment
    print(f"机器人参数: L0={initial_length:.4f}m, d={cable_distance:.4f}m")
    # <<< 新增：定义轴向耦合系数 (需要仔细调试！) >>>
    # --- 这个值需要你根据仿真结果和真实数据的对比来反复调整 ---
    # --- 从 0 开始，然后尝试小的负值，比如 -0.01, -0.05, -0.1, -0.2 ... ---
    # --- 直到仿真结果的 Z 轴变化趋势接近真实数据 ---
    axial_strain_coefficient = -2000 # <--- 示例值，需要调试！  
    # 4. 可视化参数
    body_color = [1, 0.0, 0.0, 1]; head_color = [0.0, 0.0, 0.75, 1]
    body_sphere_radius = 0.02; number_of_sphere = 30
    my_sphere_radius = body_sphere_radius; my_number_of_sphere = number_of_sphere; my_head_color = head_color

    # --- 加载并处理绳长和真实坐标数据 ---
    print(f"[信息] 正在加载数据文件: {DATA_FILE_PATH}")
    absolute_lengths_mm = None; dl_sequence_m = None; real_xyz_mm = None
    if not os.path.exists(DATA_FILE_PATH): print(f"[错误] 文件未找到: {DATA_FILE_PATH}"); sys.exit(1)
    try:
        df_input = pd.read_excel(DATA_FILE_PATH, sheet_name=SHEET_NAME, engine='openpyxl')
        print(f"[成功] 已加载数据，共 {len(df_input)} 行。")
        # --- <<< 修改：同时检查 cblen 和真实 XYZ 列 >>> ---
        required_cols = ['cblen1', 'cblen2', 'cblen3', 'X', 'Y', 'Z']
        if not all(col in df_input.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in df_input.columns]
            raise ValueError(f"错误: 文件缺少必需的列: {missing_cols}")
        # 提取绝对绳长 (mm)
        absolute_lengths_mm = df_input[['cblen1', 'cblen2', 'cblen3']].values
        # --- <<< 新增：提取真实 XYZ 坐标 (假设单位是 mm) >>> ---
        real_xyz_mm = df_input[['X', 'Y', 'Z']].values
        print(f"[信息] 成功提取 {len(absolute_lengths_mm)} 行绳长和真实 XYZ 坐标。")

        # 计算 dl 序列 (m)
        if len(absolute_lengths_mm) > 0:
            L0_cables_mm = absolute_lengths_mm[0]; print(f"[假设] 使用第一行 L0(mm): {L0_cables_mm}")
            absolute_lengths_m = absolute_lengths_mm / 1000.0; L0_cables_m = L0_cables_mm / 1000.0
            dl_sequence_m = absolute_lengths_m - L0_cables_m; print(f"[信息] 已计算 {len(dl_sequence_m)} 行 dl (m)。")
        else: raise ValueError("错误: 数据文件为空。")
    except Exception as e: print(f"[错误] 加载或处理数据时出错: {e}"); sys.exit(1)
    if dl_sequence_m is None: print("[错误] 未能计算 dl 序列。"); sys.exit(1)

    # --- PyBullet 初始化 ---
    print("--- 初始化 PyBullet ---"); simulationStepTime = 0.01; physicsClientId = -1
    try: physicsClientId = p.connect(p.GUI); # ... (连接检查) ...
    except Exception as e: print(f"连接 PyBullet 出错: {e}"); sys.exit(1)
    # ... (设置路径、重力、时间步长、加载地面) ...
    p.setAdditionalSearchPath(pybullet_data.getDataPath()); p.setGravity(0, 0, -9.81); p.setTimeStep(simulationStepTime)
    try: planeId = p.loadURDF("plane.urdf"); print(f"加载 plane.urdf, ID: {planeId}")
    except p.error as e: print(f"加载 plane.urdf 出错: {e}"); p.disconnect(physicsClientId); sys.exit(1)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
    p.resetDebugVisualizerCamera(cameraDistance=0.8, cameraYaw=45, cameraPitch=-20, cameraTargetPosition=[0, 0, 0.3])
    print("[信息] 已设置相机视角。")

    # --- 初始化 ODE 对象 ---
    print("--- 初始化 ODE 对象 ---"); my_ode = ODE(); my_ode.l0 = initial_length; my_ode.d = cable_distance
    print(f"ODE 已初始化 L0={my_ode.l0:.4f}m, d={my_ode.d:.4f}m")

    # --- 计算初始形态 (dl=0) ---
    print("--- 计算初始形状 (dl=0) ---"); act0_segment = np.zeros(3); my_ode._reset_y0(); my_ode.updateAction(act0_segment)
    sol0 = my_ode.odeStepFull()
    if sol0 is None or sol0.shape[1] < 3: print("错误: 初始 ODE 求解失败或点数不足(<3)。"); p.disconnect(physicsClientId); sys.exit(1)
    print(f"初始形状计算完成。")

    # --- 设置基座位置和姿态 (竖直向下 - 绕 X 轴旋转) ---
    base_pos = np.array([0, 0, 0.6]); base_ori_euler = np.array([-math.pi / 2.0, 0, 0]); base_ori = p.getQuaternionFromEuler(base_ori_euler)
    print(f"[设置] 基座世界坐标: {base_pos}"); print(f"[设置] 基座世界姿态 (Euler): {base_ori_euler}")
    radius = my_sphere_radius

    # --- 创建 PyBullet 形状 ---
    print("--- 创建 PyBullet 形状 ---"); # ... (与上一版本相同，代码省略) ...
    try: shape = p.createCollisionShape(p.GEOM_SPHERE, radius=radius); visualShapeId = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=body_color); visualShapeId_tip_body = p.createVisualShape(p.GEOM_SPHERE, radius=radius+0.0025, rgbaColor=my_head_color); visualShapeId_tip_gripper = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.002, 0.01, 0.002], rgbaColor=head_color)
    except p.error as e: print(f"创建形状出错: {e}"); p.disconnect(physicsClientId); sys.exit(1)

    # --- 创建 PyBullet 物体 (基于初始形态和竖直姿态) ---
    print("--- 创建 PyBullet 物体 ---"); # ... (与上一版本相同，代码省略) ...
    if sol0.shape[1] < 3: print("错误: 初始解点数不足3。"); p.disconnect(physicsClientId); sys.exit(1)
    idx0 = np.linspace(0, sol0.shape[1] - 1, my_number_of_sphere, dtype=int); positions0_local = [(sol0[0, i], sol0[2, i], sol0[1, i]) for i in idx0]
    my_robot_bodies = []; # ... (创建身体和末端物体的循环，代码省略) ...
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

    # --- <<< 修改：初始化用于存储结果的列表 (调整列名和预期单位) >>> ---
    results_data = {
        'cblen1_mm': [],       # 输入绳长1 (mm)
        'cblen2_mm': [],       # 输入绳长2 (mm)
        'cblen3_mm': [],       # 输入绳长3 (mm)
        'X_real_mm': [],       # 真实 X (mm)
        'Y_real_mm': [],       # 真实 Y (mm)
        'Z_real_mm': [],       # 真实 Z (mm)
        'sim_X_mm': [],        # 仿真 X (mm) <--- 单位将转换
        'sim_Y_mm': [],        # 仿真 Y (mm) <--- 单位将转换
        'sim_Z_mm': []         # 仿真 Z (mm) <--- 单位将转换
    }
    # --- <<< 修改结束 >>> ---


    # --- 主循环 (使用 dl 数据驱动) ---
    play_mode = 'single'; current_row_index = 0; num_data_rows = len(dl_sequence_m)
    if num_data_rows == 0: print("[错误] 没有可用的绳长变化量数据。"); p.disconnect(physicsClientId); sys.exit(1)
    print(f"将按顺序应用 {num_data_rows} 组绳长变化量数据。播放模式: {play_mode}")
    frame_count = 0; last_print_time = time.time(); simulation_running = True

    try:
        while simulation_running:
            # --- 获取当前帧的绳长变化量 dl ---
            if current_row_index >= num_data_rows:
                if play_mode == 'stop_after_one_run': current_row_index = 0
                else: print("[信息] 所有数据已播放完毕。"); simulation_running = False; continue
            if not p.isConnected(physicsClientId): print("[警告] PyBullet 连接已断开。"); simulation_running = False; continue
            dl_segment = dl_sequence_m[current_row_index]

            # --- 计算新形态 ---
            my_ode._reset_y0()
            ux, uy = calculate_curvatures_from_dl(dl_segment, cable_distance, L0_seg, num_cables)
            action_ode_segment = convert_curvatures_to_ode_action(ux, uy, 0.0, cable_distance, L0_seg)
            my_ode.updateAction(action_ode_segment)
            sol = my_ode.odeStepFull()

            # --- 更新 PyBullet 可视化并记录末端位置 ---
            pos_tip_world_m = np.array([np.nan, np.nan, np.nan]) # 初始化为 NaN (单位：米)
            if sol is not None and sol.shape[1] >= 3:
                # --- 更新可视化 ---
                idx = np.linspace(0, sol.shape[1] -1, my_number_of_sphere, dtype=int)
                positions_local = [(sol[0, i], sol[2, i], sol[1, i]) for i in idx] # YZ交换
                num_bodies_total = len(my_robot_bodies); num_tip_bodies = 3
                num_body_spheres = num_bodies_total - num_tip_bodies; num_points_available = len(positions_local)
                num_spheres_to_update = min(num_body_spheres, num_points_available)
                for i in range(num_spheres_to_update): # 更新身体
                    pos_world, ori_world = p.multiplyTransforms(base_pos, base_ori, positions_local[i], [0,0,0,1])
                    try:
                        if i < len(my_robot_bodies): p.resetBasePositionAndOrientation(my_robot_bodies[i], pos_world, ori_world)
                    except p.error: pass
                # --- 获取末端位置 (世界坐标，单位：米) ---
                if num_points_available >= 3 and num_bodies_total >= num_tip_bodies:
                    try:
                        ori_tip_local, _ = calculate_orientation(positions_local[-3], positions_local[-1])
                        pos_tip_world_tuple, ori_tip_world = p.multiplyTransforms(base_pos, base_ori, positions_local[-1], ori_tip_local)
                        pos_tip_world_m = np.array(pos_tip_world_tuple) # <<< 单位是米 (m)
                        # 更新末端可视化
                        if len(my_robot_bodies) >= num_tip_bodies:
                           p.resetBasePositionAndOrientation(my_robot_bodies[-3], pos_tip_world_m, ori_tip_world)
                           gripper_offset1 = [0, 0.01, 0]; pos1, _ = p.multiplyTransforms(pos_tip_world_m, ori_tip_world, gripper_offset1, [0,0,0,1])
                           p.resetBasePositionAndOrientation(my_robot_bodies[-2], pos1, ori_tip_world)
                           gripper_offset2 = [0,-0.01, 0]; pos2, _ = p.multiplyTransforms(pos_tip_world_m, ori_tip_world, gripper_offset2, [0,0,0,1])
                           p.resetBasePositionAndOrientation(my_robot_bodies[-1], pos2, ori_tip_world)
                    except Exception as e:
                         if frame_count % 120 == 0: print(f"警告: 更新末端或记录位置时出错 - {e}")
                         pos_tip_world_m = np.array([np.nan, np.nan, np.nan]) # 出错时记录 NaN
            elif sol is None or sol.shape[1] < 3:
                 if frame_count % 120 == 0: print(f"警告: 第 {current_row_index} 帧无法获取有效形状({sol.shape if sol is not None else 'None'})。")
                 pos_tip_world_m = np.array([np.nan, np.nan, np.nan]) # 记录 NaN

            # --- <<< 修改：记录数据 (按要求的列和单位) >>> ---
            # 获取当前行的输入 cblen (mm)
            current_cblen_mm = absolute_lengths_mm[current_row_index]
            results_data['cblen1_mm'].append(current_cblen_mm[0])
            results_data['cblen2_mm'].append(current_cblen_mm[1])
            results_data['cblen3_mm'].append(current_cblen_mm[2])
            # 获取当前行的真实 XYZ (mm)
            current_real_xyz_mm = real_xyz_mm[current_row_index]
            results_data['X_real_mm'].append(current_real_xyz_mm[0])
            results_data['Y_real_mm'].append(current_real_xyz_mm[1])
            results_data['Z_real_mm'].append(current_real_xyz_mm[2])
            # 转换并记录仿真的 XYZ (mm)
            sim_x_mm = pos_tip_world_m[0] * 1000.0 if not np.isnan(pos_tip_world_m[0]) else np.nan
            sim_y_mm = pos_tip_world_m[1] * -1000.0 if not np.isnan(pos_tip_world_m[1]) else np.nan
            sim_z_mm = pos_tip_world_m[2] * 1000.0 -480 if not np.isnan(pos_tip_world_m[2]) else np.nan
            results_data['sim_X_mm'].append(sim_x_mm)
            results_data['sim_Y_mm'].append(sim_y_mm)
            results_data['sim_Z_mm'].append(sim_z_mm)
            # --- <<< 记录数据结束 >>> ---

            # --- 仿真步进和延时 ---
            p.stepSimulation()
            time.sleep(simulationStepTime)
            frame_count += 1
            current_row_index += 1

    # ... (异常处理和 PyBullet 断开连接保持不变) ...
    except KeyboardInterrupt: print("\n[信息] 检测到键盘中断 (Ctrl+C)...")
    except p.error as e: print(f"\n[错误] PyBullet 发生错误: {e}")
    except Exception as e: print(f"\n[错误] 发生意外错误: {e}"); import traceback; traceback.print_exc()
    finally:
        print("[信息] 断开 PyBullet 连接。")
        if 'physicsClientId' in locals() and p.isConnected(physicsClientId):
            try: p.disconnect(physicsClientId)
            except p.error: pass

    # --- <<< 修改：保存指定顺序和列名的结果到 Excel 文件 >>> ---
    print("\n--- 保存仿真结果 ---")
    try:
        if len(results_data['cblen1_mm']) > 0:
            # 定义期望的列顺序
            output_column_order = [
                'cblen1_mm', 'cblen2_mm', 'cblen3_mm',
                'X_real_mm', 'Y_real_mm', 'Z_real_mm',
                'sim_X_mm', 'sim_Y_mm', 'sim_Z_mm'
            ]
            # 创建 DataFrame 并按指定顺序排列列
            output_results_df = pd.DataFrame(results_data)
            output_results_df = output_results_df[output_column_order] # 重新排序列

            print(f"[信息] 正在将 {len(output_results_df)} 条结果 ({len(output_column_order)} 列) 保存到: {OUTPUT_RESULTS_PATH}")
            output_results_df.to_excel(OUTPUT_RESULTS_PATH, index=False, engine='openpyxl')
            print(f"[成功] 结果已保存至: {os.path.abspath(OUTPUT_RESULTS_PATH)}")
        else:
            print("[警告] 没有记录到任何仿真结果，未生成输出文件。")
    except ImportError: print("[错误] 保存 Excel 文件需要 'openpyxl' 库。请运行 'pip install openpyxl' 安装。")
    except KeyError as e: print(f"[错误] 尝试保存结果时发生列名错误: {e}。请检查 'results_data' 字典和 'output_column_order' 列表。")
    except Exception as e: print(f"[错误] 保存结果到 Excel 文件时出错: {e}")
    # --- <<< 保存结果结束 >>> ---

    print("--- 仿真结束 ---")