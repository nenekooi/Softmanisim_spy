# -*- coding: utf-8 -*-
# ==============================================================================
# 脚本功能:
# 读取包含仿真结果(mm)和真实数据(mm)的 CSV 文件 (gap.csv)，
# 绘制三张折线图，分别对比仿真和真实的 X, Y, Z 坐标 (单位: mm) 随样本序号的变化。
# 增加中文显示支持，并为仿真值和真实值使用不同颜色。
# ==============================================================================

import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# --- <<< 新增：配置 Matplotlib 支持中文显示 >>> ---
try:
    # 尝试使用 'SimHei' 字体，适用于 Windows/macOS (需安装) 或 Linux (需安装)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 解决负号显示为方块的问题
    plt.rcParams['axes.unicode_minus'] = False
    print("[信息] 尝试设置字体为 'SimHei' 以支持中文。")
except Exception as e:
    print(f"[警告] 设置 'SimHei' 字体失败: {e}")
    print("[警告] 中文可能无法正常显示。请确保您的系统安装了支持中文的字体（如 SimHei, Microsoft YaHei 等）并被 Matplotlib 识别。")
# ------------------------------------------------

# --- 参数设置 ---

# 输入数据文件的相对路径 (假设此脚本在 scripts 目录下)
# !!! 如果您的脚本位置不同，或者文件名/路径不同，请修改这里 !!!
INPUT_CSV_PATH = 'c:/Users/11647/SoftManiSim/spy code/Sim2Real_Results_lengtn0.14.csv'

# --- 核心绘图逻辑 ---

if __name__ == "__main__":
    print(f"--- 开始绘图 ---")
    print(f"[信息] 尝试加载数据文件: {INPUT_CSV_PATH}")

    # 检查文件是否存在
    if not os.path.exists(INPUT_CSV_PATH):
        print(f"[错误] 文件未找到: {INPUT_CSV_PATH}")
        print("[提示] 请确保文件路径相对于当前运行脚本的位置是正确的。")
        sys.exit(1)

    try:
        # --- 加载数据 ---
        df = pd.read_csv(INPUT_CSV_PATH)
        print(f"[成功] 已加载数据 {len(df)} 行。")

        sim_cols = ['sim_X_mm', 'sim_Y_mm', 'sim_Z_mm']
        real_cols = ['X_real_mm', 'Y_real_mm', 'Z_real_mm'] # 假设真实数据的列标题是 X, Y, Z
        required_cols = sim_cols + real_cols

        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"错误: 文件 '{os.path.basename(INPUT_CSV_PATH)}' 缺少必需的列: {missing_cols}")
        print("[信息] 所需数据列均存在。")

        # --- 获取数据 (单位均为 mm) ---
        print("[信息] 假设仿真和真实 XYZ 数据单位均为毫米 (mm)。")
        sim_x = df['sim_X_mm']
        sim_y = df['sim_Y_mm']
        sim_z = df['sim_Z_mm']

        real_x = df['X_real_mm']
        real_y = df['Y_real_mm']
        real_z = df['Z_real_mm']

        # 创建样本索引作为 X 轴 (例如时间步或数据点序号)
        sample_index = range(len(df))

        # --- 开始绘图 ---
        print("[信息] 正在生成对比图 (单位: mm)...")

        # 创建一个图形窗口，包含 3 个子图 (垂直排列)
        fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        fig.suptitle('仿真结果 vs 真实结果对比 (单位: mm)', fontsize=16) # 总标题

        # --- <<< 修改：为仿真和真实值设置不同颜色 >>> ---
        # 1. 绘制 X 坐标对比图
        axs[0].plot(sample_index, sim_x, label='仿真 X (mm)', color='blue', linestyle='--') # 仿真用蓝色虚线
        axs[0].plot(sample_index, real_x, label='真实 X (mm)', color='darkorange', linestyle='-') # 真实用橙色实线
        axs[0].set_ylabel('X 坐标 (mm)')
        axs[0].set_title('X 坐标对比')
        axs[0].legend()
        axs[0].grid(True)

        # 2. 绘制 Y 坐标对比图
        axs[1].plot(sample_index, sim_y, label='仿真 Y (mm)', color='blue', linestyle='--') 
        axs[1].plot(sample_index, real_y, label='真实 Y (mm)', color='darkorange', linestyle='-') 
        axs[1].set_ylabel('Y 坐标 (mm)')
        axs[1].set_title('Y 坐标对比')
        axs[1].legend()
        axs[1].grid(True)

        # 3. 绘制 Z 坐标对比图
        axs[2].plot(sample_index, sim_z, label='仿真 Z (mm)', color='blue', linestyle='--') 
        axs[2].plot(sample_index, real_z, label='真实 Z (mm)', color='darkorange', linestyle='-') 
        axs[2].set_ylabel('Z 坐标 (mm)')
        axs[2].set_title('Z 坐标对比')
        axs[2].set_xlabel('样本序号 (Sample Index)')
        axs[2].legend()
        axs[2].grid(True)


        # 调整子图布局，防止标签重叠
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])

        # 显示图形窗口
        print("[信息] 图形已生成，请查看弹出的窗口。")
        plt.show()

        print("--- 绘图结束 ---")

    # --- 错误处理 ---
    except FileNotFoundError: print(f"[错误] 无法找到文件: {INPUT_CSV_PATH}")
    except ValueError as e: print(f"[错误] 数据读取或处理时出错: {e}")
    except ImportError: print("[错误] 缺少必要的库。请确保已安装 pandas 和 matplotlib (pip install pandas matplotlib)"); sys.exit(1)
    except Exception as e: print(f"[错误] 发生未知错误: {e}"); import traceback; traceback.print_exc()