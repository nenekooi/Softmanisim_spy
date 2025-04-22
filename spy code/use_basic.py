# -*- coding: utf-8 -*-
# ==============================================================================
# 脚本功能: (最终版本 - 结合环境类简化，修正调试代码语法)
# 读取绳长数据，使用 SoftRobotBasicEnvironment，并确保其内部的 ODE 实例
# 使用了正确的参数 (包括 k_strain)，驱动仿真并记录结果。
# ==============================================================================

# 导入所需的库
import sys
import os
import numpy as np
# import pybullet as p # 由环境类处理
# import pybullet_data # 由环境类处理
import time
import math
from pprint import pprint
import pandas as pd
from scipy.spatial.transform import Rotation as R # 保留，可能需要

# --- 动态添加项目路径 ---
try:
    script_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    if project_root not in sys.path:
        sys.path.append(project_root)
except NameError:
    print("[警告] 无法自动确定项目根目录，请确保 BasicEnvironment 和 visualizer 模块在 Python 路径中。")

# --- 导入环境类 ---
try:
    from pybullet_env.BasicEnvironment import SoftRobotBasicEnvironment
    print("[调试] 成功从 'BasicEnvironment' 导入 SoftRobotBasicEnvironment")
except ImportError as e:
    print(f"[错误] 无法导入 SoftRobotBasicEnvironment: {e}")
    raise

# ==============================================================================
# 数据处理函数 (省略，与之前相同)
# ==============================================================================
def load_and_preprocess_data(file_path, sheet_name):
    """加载 Excel 文件，提取所需数据，并进行预处理。"""
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
    """初始化用于存储仿真结果的字典。"""
    return {
        'cblen1_mm': [], 'cblen2_mm': [], 'cblen3_mm': [],
        'X_real_mm': [], 'Y_real_mm': [], 'Z_real_mm': [],
        'sim_X_mm': [], 'sim_Y_mm': [], 'sim_Z_mm': []
    }

def append_result(results_dict, cblen_mm, real_xyz_mm, sim_xyz_m):
    """将单步的输入和仿真结果追加到结果字典中。"""
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
    """将结果字典保存到 Excel 文件。"""
    print("\n--- 保存仿真结果 ---")
    output_column_order = [
        'cblen1_mm', 'cblen2_mm', 'cblen3_mm',
        'X_real_mm', 'Y_real_mm', 'Z_real_mm',
        'sim_X_mm', 'sim_Y_mm', 'sim_Z_mm'
    ]
    if not results_dict or not results_dict.get('cblen1_mm'):
         print("[警告] 没有结果数据被记录，无法保存文件。")
         return
    try:
        output_results_df = pd.DataFrame(results_dict)
        for col in output_column_order:
            if col not in output_results_df.columns:
                print(f"[警告] 结果中缺少列 '{col}'，将填充 NaN。")
                output_results_df[col] = np.nan
        output_results_df = output_results_df[output_column_order]
        print(f"[信息] 正在将 {len(output_results_df)} 条结果 ({len(output_column_order)} 列) 保存到: {output_path}")
        output_results_df.to_excel(output_path, index=False, engine='openpyxl')
        print(f"[成功] 结果已保存至: {os.path.abspath(output_path)}")
    except Exception as e:
        print(f"[错误] 保存 Excel 文件时出错: {e}")

# ==============================================================================
# 辅助计算函数 (省略，与之前相同)
# ==============================================================================
def calculate_curvatures_from_dl(dl_segment, d, L0_seg, num_cables=3):
    """根据绳长变化量计算曲率 ux, uy (假设输入有效且为3绳索)。"""
    ux = 0.0; uy = 0.0
    if abs(d) < 1e-9: print("警告: 绳索半径 d 接近于零。"); return ux, uy
    dl1, dl2, dl3 = dl_segment[0], dl_segment[1], dl_segment[2]
    uy = -dl1 / d
    denominator_ux = d * math.sqrt(3.0)
    if abs(denominator_ux) > 1e-9: ux = (dl3 - dl2) / denominator_ux
    else: ux = 0.0; print("警告: 计算 ux 时分母接近零。")
    return ux, uy

def convert_curvatures_to_ode_action(ux, uy, length_change, d, L0_seg):
    """将曲率转换为 ODE action。"""
    l = L0_seg; action_ode = np.zeros(3)
    action_ode[0] = length_change
    action_ode[1] = uy * l * d
    action_ode[2] = -ux * l * d
    return action_ode


# ==============================================================================
# 主程序 (使用环境类，配置 ODE，修正连接检查语法)
# ==============================================================================
if __name__ == "__main__":

    print("--- 设置参数 ---")
    DATA_FILE_PATH = 'c:/Users/11647/Desktop/data/circle2_without_0.xlsx'
    SHEET_NAME = 'Sheet1'
    OUTPUT_RESULTS_PATH = os.path.join(os.path.dirname(DATA_FILE_PATH), 'env_debug_connection_v3.xlsx') # 新输出文件名

    # --- 机器人和仿真参数 ---
    num_cables = 3
    cable_distance = 0.004
    initial_length = 0.12
    number_of_segment = 1
    L0_seg = initial_length / number_of_segment
    print(f"机器人参数: L0={initial_length:.4f}m, d={cable_distance:.4f}m")
    axial_strain_coefficient = -20 # 传递给 ODE
    AXIAL_ACTION_SCALE = 0.02      # 用于计算 action

    # --- 可视化参数 (传递给环境) ---
    body_color = [1, 0.0, 0.0, 1]
    head_color = [0.0, 0.0, 0.75, 1]
    body_sphere_radius = 0.02
    number_of_sphere = 30
    gui_enabled = True

    # --- 基座参数 (传递给 env.move_robot_ori) ---
    absolute_base_pos = np.array([0, 0, 0.6])
    absolute_base_ori_euler = np.array([-math.pi / 2.0, 0, 0])

    # --- 加载数据 ---
    absolute_lengths_mm, real_xyz_mm, dl_sequence_m, L0_cables_mm = load_and_preprocess_data(DATA_FILE_PATH, SHEET_NAME)

    # --- 初始化结果存储 ---
    results_data = initialize_results_storage()

    # --- 初始化环境 ---
    env = None # 先声明变量
    connection_ok = False # 标记连接是否成功
    try:
        print("\n--- 初始化环境 ---")
        env = SoftRobotBasicEnvironment(
            gui=gui_enabled,
            body_color=body_color,
            head_color=head_color,
            body_sphere_radius=body_sphere_radius,
            number_of_sphere=number_of_sphere,
            number_of_segment=number_of_segment
        )
        print("仿真环境实例已创建。")

        # *** 再次修正调试代码：检查连接状态 ***
        if hasattr(env, 'bullet') and env.bullet:
            try:
                # *** 正确的调用方式 ***
                # env.bullet 存储的是 pybullet 模块 p
                # 所以直接用 env.bullet 调用模块函数
                gravity_check = env.bullet.getGravity()
                print(f"[调试] 环境初始化后 PyBullet 连接正常，重力: {gravity_check}")
                connection_ok = True # 标记连接成功
            except AttributeError as attr_err:
                 # 如果报 AttributeError，说明 env.bullet 不是预期的 pybullet 模块
                 print(f"[调试][错误] 无法调用 env.bullet.getGravity()，env.bullet 的类型是: {type(env.bullet)} - {attr_err}")
                 print("[提示] 请检查 SoftRobotBasicEnvironment 中 self.bullet 的赋值！")
            except Exception as conn_check_e:
                 # 其他错误，可能是连接问题
                 print(f"[调试][错误] 调用 env.bullet.getGravity() 时发生错误: {conn_check_e}")
                 print("[提示] PyBullet 连接可能在初始化过程中失败或丢失！检查 __init__。")
        else:
            print("[调试][错误] 环境对象 env 没有 'bullet' 属性或为 None。检查 __init__。")

        # 如果连接失败，则不继续执行
        if not connection_ok:
             # 尝试清理可能的半开连接
             if env and hasattr(env, 'bullet') and hasattr(env.bullet, 'disconnect'):
                 try: env.bullet.disconnect()
                 except: pass
             sys.exit("环境初始化或 PyBullet 连接失败。")

        # --- 配置环境内部的 ODE 实例参数 ---
        # (省略，与上一版本相同)
        print("\n--- 配置环境内部 ODE 参数 ---")
        if hasattr(env, '_ode') and env._ode:
            env._ode.l0 = initial_length
            env._ode.d = cable_distance
            if hasattr(env._ode, 'k_strain'):
                env._ode.k_strain = axial_strain_coefficient
                print(f"环境ODE参数已设置: l0={env._ode.l0:.4f}, d={env._ode.d:.4f}, k_strain={env._ode.k_strain}")
            else:
                print(f"[警告] 环境 ODE 对象无 k_strain 属性。轴向耦合将无效。")
                print(f"环境ODE参数已设置: l0={env._ode.l0:.4f}, d={env._ode.d:.4f}")
        else:
             print("[警告] 无法找到或访问环境内部的 _ode 对象来设置参数。")


    except Exception as env_init_e:
        print(f"[致命错误] 初始化 SoftRobotBasicEnvironment 时失败: {env_init_e}")
        # 尝试清理
        if env and hasattr(env, 'bullet') and hasattr(env.bullet, 'disconnect'):
             try: env.bullet.disconnect()
             except: pass
        sys.exit("环境初始化失败。")

    # --- 主循环 (仅在环境和连接都 OK 时执行) ---
    if env and connection_ok:
        num_data_rows = len(dl_sequence_m)
        print(f"\n--- 开始主仿真循环 ({num_data_rows} 组数据) ---")

        for current_row_index in range(num_data_rows):
            # ... (获取 dl_segment, cblen_mm, real_xyz_mm 的代码不变) ...
            dl_segment = dl_sequence_m[current_row_index]
            current_cblen_mm = absolute_lengths_mm[current_row_index]
            current_real_xyz_mm = real_xyz_mm[current_row_index]

            # --- 计算 ODE Action ---
            avg_dl = np.mean(dl_segment)
            commanded_length_change = avg_dl * AXIAL_ACTION_SCALE
            ux, uy = calculate_curvatures_from_dl(dl_segment, cable_distance, L0_seg, num_cables)
            action_ode_segment = convert_curvatures_to_ode_action(ux, uy, commanded_length_change, cable_distance, L0_seg)

            # --- 更新环境状态和可视化 ---
            pos_tip_world_m = np.array([np.nan, np.nan, np.nan]) # 初始化
            try:
                # 调用环境方法
                _, _ = env.move_robot_ori(action=action_ode_segment,
                                          base_pos=absolute_base_pos,
                                          base_orin=absolute_base_ori_euler)

                # 获取仿真末端位置
                if hasattr(env, '_head_pose') and env._head_pose is not None:
                    if isinstance(env._head_pose, (list, tuple)) and len(env._head_pose) > 0:
                        pos_tip_world_m = np.array(env._head_pose[0])
                    else:
                        print(f"[警告] env._head_pose 格式不符合预期: {env._head_pose}")
                else:
                    print("[警告] 无法从 env._head_pose 获取末端位置。")

            except Exception as e:
                 # 捕获运行时错误, 包括 "Not connected"
                 print(f"\n[错误] 执行 env.move_robot_ori 时发生错误 (行号 {current_row_index}): {e}")
                 print("[提示] 请仔细检查 SoftRobotBasicEnvironment 类中 move_robot_ori 方法的实现！")
                 break # 中断循环

            # --- 存储当前步结果 ---
            append_result(results_data, current_cblen_mm, current_real_xyz_mm, pos_tip_world_m)

            # --- 打印进度 (可选) ---
            # if (current_row_index + 1) % 100 == 0 or current_row_index == num_data_rows - 1:
            #      print(f"已处理 {current_row_index + 1}/{num_data_rows}...")


        print("[信息] 仿真循环处理完毕（可能因错误提前结束）。")

        # --- 保存结果 ---
        # save_results_to_excel(results_data, OUTPUT_RESULTS_PATH)

    else:
        print("[错误] 环境对象 'env' 未成功初始化或 PyBullet 连接失败，无法运行仿真循环。")


    # --- 仿真结束，无需手动断开连接 ---
    print("\n--- 程序结束 ---")