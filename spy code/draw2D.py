# -*- coding: utf-8 -*-
# ==============================================================================
# 脚本功能:
# 读取包含仿真结果(mm)和真实数据(mm)的 CSV 文件，
# 计算仿真与真实值之间的误差，
# 1. 绘制包含三张折线图的图表，分别对比 X, Y, Z 坐标随样本的变化，
#    并在各子图上标注 MAE。
# 2. 绘制一张独立的 2D 轨迹图，对比仿真和真实的 X-Y 坐标，
#    并在图上标注 X-Y 平面的平均距离误差。
# 增加中文显示支持，并为仿真值和真实值使用不同颜色。
# ==============================================================================

import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import numpy as np # 导入 numpy 用于计算

# --- 中文显示配置 ---
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    print("[信息] 尝试设置字体为 'SimHei' 以支持中文。")
except Exception as e:
    print(f"[警告] 设置 'SimHei' 字体失败: {e}")
    print("[警告] 中文可能无法正常显示。请确保系统安装了支持中文的字体。")
# ------------------------------------------------

# --- 参数设置 ---
# !!! 请确保这个路径指向你最新的包含 sim 和 real 数据的 CSV 文件 !!!
INPUT_CSV_PATH = 'c:/Users/11647/Desktop/data/changzcircle2test.csv' # <<< 检查这个路径

# --- 核心逻辑 ---

if __name__ == "__main__":
    print(f"--- 开始处理与绘图 ---")
    print(f"[信息] 尝试加载数据文件: {INPUT_CSV_PATH}")

    if not os.path.exists(INPUT_CSV_PATH):
        print(f"[错误] 文件未找到: {INPUT_CSV_PATH}")
        sys.exit(1)

    try:
        # --- 加载数据 ---
        df = pd.read_csv(INPUT_CSV_PATH)
        print(f"[成功] 已加载数据 {len(df)} 行。")

        sim_cols = ['sim_X_mm', 'sim_Y_mm', 'sim_Z_mm']
        real_cols = ['X_real_mm', 'Y_real_mm', 'Z_real_mm']
        required_cols = sim_cols + real_cols

        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"错误: 文件缺少必需列: {missing_cols}")
        print("[信息] 所需数据列均存在。")

        # --- 获取数据 ---
        # 完整数据用于绘图
        sim_x = df['sim_X_mm']
        sim_y = df['sim_Y_mm']
        sim_z = df['sim_Z_mm']
        real_x = df['X_real_mm']
        real_y = df['Y_real_mm']
        real_z = df['Z_real_mm']

        # 创建样本索引作为第一个图的 X 轴
        sample_index = range(len(df))

        # --- 计算误差 (处理潜在的 NaN 值) ---
        print("[信息] 正在计算误差...")

        # 创建一个包含所有需要计算误差的列的 DataFrame副本，以便处理 NaN
        error_df = df[required_cols].copy()

        # 删除任何包含 NaN 的行，以确保误差计算的有效性
        rows_before_dropna = len(error_df)
        error_df.dropna(inplace=True)
        rows_after_dropna = len(error_df)

        mae_x, mae_y, mae_z = np.nan, np.nan, np.nan
        mae_xy = np.nan # 初始化为 NaN
        mae_3d = np.nan # 初始化为 NaN

        if rows_before_dropna > rows_after_dropna:
            print(f"[警告] 数据中包含 NaN 值，已删除 {rows_before_dropna - rows_after_dropna} 行数据后进行误差计算。")

        if rows_after_dropna == 0:
             print("[错误] 清理 NaN 后无有效数据可用于计算误差。")
             # 可以选择退出或继续绘图但不显示误差
        else:
            # 提取清理后的数据用于误差计算
            sim_x_clean = error_df['sim_X_mm']
            sim_y_clean = error_df['sim_Y_mm']
            sim_z_clean = error_df['sim_Z_mm']
            real_x_clean = error_df['X_real_mm']
            real_y_clean = error_df['Y_real_mm']
            real_z_clean = error_df['Z_real_mm']

            # 1. 计算各轴绝对误差
            abs_err_x = np.abs(real_x_clean - sim_x_clean)
            abs_err_y = np.abs(real_y_clean - sim_y_clean)
            abs_err_z = np.abs(real_z_clean - sim_z_clean)

            # 2. 计算各轴平均绝对误差 (MAE - 平均损失)
            mae_x = np.mean(abs_err_x)
            mae_y = np.mean(abs_err_y)
            mae_z = np.mean(abs_err_z)

            # 3. 计算每个点的 2D 欧氏距离误差 (X-Y平面)
            dist_error_xy = np.sqrt((real_x_clean - sim_x_clean)**2 +
                                    (real_y_clean - sim_y_clean)**2)

            # 4. 计算 X-Y 平面平均距离误差 (MAE_XY)
            mae_xy = np.mean(dist_error_xy)

            # 5. 计算每个点的 3D 欧氏距离误差
            dist_error_3d = np.sqrt((real_x_clean - sim_x_clean)**2 +
                                    (real_y_clean - sim_y_clean)**2 +
                                    (real_z_clean - sim_z_clean)**2)

            # 6. 计算 3D 平均欧氏距离误差 (MAE_3D)
            mae_3d = np.mean(dist_error_3d)

            print(f"计算完成:")
            print(f"  MAE (各轴平均损失) - X: {mae_x:.3f} mm, Y: {mae_y:.3f} mm, Z: {mae_z:.3f} mm")
            print(f"  X-Y 平面平均距离误差 (MAE_XY): {mae_xy:.3f} mm")
            print(f"  3D 平均欧氏距离误差 (MAE_3D): {mae_3d:.3f} mm")
        # --- 误差计算结束 ---


        # ==============================================================
        # --- 开始绘制第一个图：X, Y, Z 坐标 vs 样本序号 ---
        # ==============================================================
        print("[信息] 正在生成第一个对比图 (各轴坐标 vs 样本序号)...")
        fig1, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True) # 使用 fig1

        fig1.suptitle(f'仿真 vs 真实结果对比 (坐标 vs 样本序号)\nOverall 3D MAE: {mae_3d:.3f} mm', fontsize=16)

        # --- 绘制 X 坐标对比图 ---
        axs[0].plot(sample_index, sim_x, label='仿真 X (mm)', color='blue', linestyle='--')
        axs[0].plot(sample_index, real_x, label='真实 X (mm)', color='darkorange', linestyle='-')
        axs[0].set_ylabel('X 坐标 (mm)')
        axs[0].set_title('X 坐标对比')
        axs[0].legend()
        axs[0].grid(True)
        axs[0].text(0.02, 0.95, f'MAE_X: {mae_x:.3f} mm',
                    transform=axs[0].transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

        # --- 绘制 Y 坐标对比图 ---
        axs[1].plot(sample_index, sim_y, label='仿真 Y (mm)', color='blue', linestyle='--')
        axs[1].plot(sample_index, real_y, label='真实 Y (mm)', color='darkorange', linestyle='-')
        axs[1].set_ylabel('Y 坐标 (mm)')
        axs[1].set_title('Y 坐标对比')
        axs[1].legend()
        axs[1].grid(True)
        axs[1].text(0.02, 0.95, f'MAE_Y: {mae_y:.3f} mm',
                    transform=axs[1].transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

        # --- 绘制 Z 坐标对比图 ---
        axs[2].plot(sample_index, sim_z, label='仿真 Z (mm)', color='blue', linestyle='--')
        axs[2].plot(sample_index, real_z, label='真实 Z (mm)', color='darkorange', linestyle='-')
        axs[2].set_ylabel('Z 坐标 (mm)')
        axs[2].set_title('Z 坐标对比')
        axs[2].set_xlabel('样本序号 (Sample Index)')
        axs[2].legend()
        axs[2].grid(True)
        axs[2].text(0.02, 0.95, f'MAE_Z: {mae_z:.3f} mm',
                    transform=axs[2].transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

        # 调整子图布局
        fig1.tight_layout(rect=[0, 0.03, 1, 0.94]) # 稍微调整 rect 避免标题和总标题重叠
        print("[信息] 第一个图表已准备好。")
        # 注意：这里的 plt.show() 会阻塞，直到第一个图的窗口关闭才继续执行后续代码
        # 如果希望同时显示，可能需要不同的方法或只在最后调用一次 plt.show()
        # plt.show() # 可以在这里显示第一个图


        # ==============================================================
        # --- 开始绘制第二个图：X-Y 二维轨迹 ---
        # ==============================================================
        print("[信息] 正在生成第二个对比图 (X-Y 轨迹)...")
        # --- <<< 新增：创建第二个 Figure 和 Axes >>> ---
        fig2, ax2 = plt.subplots(1, 1, figsize=(8, 8)) # 使用 fig2, ax2

        # --- <<< 绘制 X-Y 轨迹 >>> ---
        ax2.plot(sim_x, sim_y, label=f'仿真轨迹 (XY)', color='blue', linestyle='--')
        ax2.plot(real_x, real_y, label=f'真实轨迹 (XY)', color='darkorange', linestyle='-')

        # --- <<< 设置标签和标题 >>> ---
        ax2.set_xlabel('X 坐标 (mm)')
        ax2.set_ylabel('Y 坐标 (mm)')
        ax2.set_title(f'仿真 vs 真实 X-Y 轨迹对比\nXY平面平均距离误差 (MAE_XY): {mae_xy:.3f} mm', fontsize=14)

        ax2.legend()
        ax2.grid(True)

        # --- <<< 设置等轴比例 >>> ---
        ax2.set_aspect('equal', adjustable='box')

        # 调整布局
        fig2.tight_layout() # 使用 fig2
        print("[信息] 第二个图表已准备好。")


        # ==============================================================
        # --- 显示所有图表 ---
        # ==============================================================
        # 调用一次 plt.show() 会显示所有当前创建的 Figure
        print("[信息] 显示所有图表窗口...")
        plt.show()

        print("--- 所有绘图结束 ---")

    except FileNotFoundError: print(f"[错误] 无法找到文件: {INPUT_CSV_PATH}")
    except ValueError as e: print(f"[错误] 数据读取或处理时出错: {e}")
    except ImportError: print("[错误] 缺少必要的库 (pandas, matplotlib, numpy)。请运行 'pip install pandas matplotlib numpy'"); sys.exit(1)
    except Exception as e: print(f"[错误] 发生未知错误: {e}"); import traceback; traceback.print_exc()