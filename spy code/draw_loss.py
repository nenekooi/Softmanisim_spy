# -*- coding: utf-8 -*-
# ==============================================================================
# 脚本功能:
# 读取包含仿真结果(mm)和真实数据(mm)的 XLSX 文件， # <<< 修改：从 XLSX 文件读取
# 计算仿真与真实值之间的平均绝对误差 (MAE) 和 3D 欧氏距离误差，
# 绘制三张折线图，对比 X, Y, Z 坐标，并在图上标注各轴的 MAE 和总的 3D MAE。
# 增加中文显示支持，并为仿真值和真实值使用不同颜色。
# ==============================================================================

import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import numpy as np

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
# !!! 请确保这个路径指向你最新的包含 sim 和 real 数据的 XLSX 文件 !!!
# <<< 修改：变量名和文件扩展名 >>>
INPUT_XLSX_PATH = 'c:/Users/11647/Desktop/data/changznewcircle2_with_EMA_filter.xlsx' # <<< 请将这里替换为你的 XLSX 文件路径
SHEET_NAME = 'Sheet1' # <<< 如果你的数据不在第一个工作表，修改这里的工作表名称或索引 (e.g., 0)

# --- 核心绘图逻辑 ---

if __name__ == "__main__":
    print(f"--- 开始处理与绘图 ---")
    print(f"[信息] 尝试加载数据文件: {INPUT_XLSX_PATH}")

    if not os.path.exists(INPUT_XLSX_PATH):
        print(f"[错误] 文件未找到: {INPUT_XLSX_PATH}")
        sys.exit(1)

    try:
        # --- <<< 修改：使用 pd.read_excel() 加载数据 >>> ---
        # 需要安装 openpyxl: pip install openpyxl
        df = pd.read_excel(INPUT_XLSX_PATH, sheet_name=SHEET_NAME, engine='openpyxl')
        print(f"[成功] 已加载数据 {len(df)} 行。")

        # --- 后续的数据处理、误差计算和绘图逻辑保持不变 ---

        sim_cols = ['sim_X_mm', 'sim_Y_mm', 'sim_Z_mm']
        real_cols = ['X_real_mm', 'Y_real_mm', 'Z_real_mm']
        required_cols = sim_cols + real_cols

        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"错误: 文件缺少必需列: {missing_cols}")
        print("[信息] 所需数据列均存在。")

        # --- 获取数据 ---
        sim_x = df['sim_X_mm']
        sim_y = -df['sim_Y_mm']
        sim_z = df['sim_Z_mm']
        real_x = df['X_real_mm']
        real_y = df['Y_real_mm']
        real_z = df['Z_real_mm']

        # --- 计算误差 (处理 NaN 值) ---
        print("[信息] 正在计算误差...")
        error_df = df[required_cols].copy()
        rows_before_dropna = len(error_df)
        error_df.dropna(inplace=True)
        rows_after_dropna = len(error_df)

        if rows_before_dropna > rows_after_dropna:
            print(f"[警告] 数据中包含 NaN 值，已删除 {rows_before_dropna - rows_after_dropna} 行数据后进行误差计算。")

        if rows_after_dropna == 0:
            print("[错误] 清理 NaN 后无有效数据可用于计算误差。")
            mae_x, mae_y, mae_z = np.nan, np.nan, np.nan
            # sae_x, sae_y, sae_z = np.nan, np.nan, np.nan # SAE 可能不太常用，可以注释掉
            mae_3d = np.nan
        else:
            sim_x_clean = error_df['sim_X_mm']
            sim_y_clean = error_df['sim_Y_mm']
            sim_z_clean = error_df['sim_Z_mm']
            real_x_clean = error_df['X_real_mm']
            real_y_clean = error_df['Y_real_mm']
            real_z_clean = error_df['Z_real_mm']

            abs_err_x = np.abs(real_x_clean - sim_x_clean)
            abs_err_y = np.abs(real_y_clean - sim_y_clean)
            abs_err_z = np.abs(real_z_clean - sim_z_clean)

            mae_x = np.mean(abs_err_x)
            mae_y = np.mean(abs_err_y)
            mae_z = np.mean(abs_err_z)

            # sae_x = np.sum(abs_err_x) # SAE 可能不太常用，可以注释掉
            # sae_y = np.sum(abs_err_y)
            # sae_z = np.sum(abs_err_z)

            dist_error = np.sqrt((real_x_clean - sim_x_clean)**2 +
                                 (real_y_clean - sim_y_clean)**2 +
                                 (real_z_clean - sim_z_clean)**2)
            mae_3d = np.mean(dist_error)

            print(f"计算完成:")
            print(f"  MAE (平均绝对误差) - X: {mae_x:.3f} mm, Y: {mae_y:.3f} mm, Z: {mae_z:.3f} mm")
            # print(f"  SAE (总绝对误差)   - X: {sae_x:.3f} mm, Y: {sae_y:.3f} mm, Z: {sae_z:.3f} mm")
            print(f"  3D 平均欧氏距离误差 (MAE_3D): {mae_3d:.3f} mm")

        # --- 绘图 ---
        sample_index = range(len(df)) # 绘图仍用原始索引长度
        print("[信息] 正在生成对比图 (单位: mm)...")
        fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        fig.suptitle(f'仿真 vs 真实结果对比 (单位: mm)\nOverall 3D MAE: {mae_3d:.3f} mm', fontsize=16)

        # 绘制 X 对比
        axs[0].plot(sample_index, sim_x, label='仿真 X (mm)', color='blue', linestyle='--')
        axs[0].plot(sample_index, real_x, label='真实 X (mm)', color='darkorange', linestyle='-')
        axs[0].set_ylabel('X 坐标 (mm)')
        axs[0].set_title('X 坐标对比')
        axs[0].legend()
        axs[0].grid(True)
        axs[0].text(0.02, 0.95, f'MAE_X: {mae_x:.3f} mm',
                    transform=axs[0].transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

        # 绘制 Y 对比
        axs[1].plot(sample_index, sim_y, label='仿真 Y (mm)', color='blue', linestyle='--')
        axs[1].plot(sample_index, real_y, label='真实 Y (mm)', color='darkorange', linestyle='-')
        axs[1].set_ylabel('Y 坐标 (mm)')
        axs[1].set_title('Y 坐标对比')
        axs[1].legend()
        axs[1].grid(True)
        axs[1].text(0.02, 0.95, f'MAE_Y: {mae_y:.3f} mm',
                    transform=axs[1].transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

        # 绘制 Z 对比
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

        plt.tight_layout(rect=[0, 0.03, 1, 0.94])
        print("[信息] 图形已生成，请查看弹出的窗口。")
        plt.show()
        print("--- 绘图结束 ---")

    except FileNotFoundError: print(f"[错误] 无法找到文件: {INPUT_XLSX_PATH}") # <<< 使用更新后的变量名
    except ValueError as e: print(f"[错误] 数据读取或处理时出错: {e}")
    except ImportError as e:
        # <<< 修改：提示需要 openpyxl >>>
        if 'openpyxl' in str(e).lower():
             print("[错误] 读取 Excel 文件需要 'openpyxl' 库。请运行 'pip install openpyxl' 安装。")
        else:
             print(f"[错误] 缺少必要的库 (pandas, matplotlib, numpy): {e}。请运行 'pip install pandas matplotlib numpy openpyxl'");
        sys.exit(1)
    except Exception as e: print(f"[错误] 发生未知错误: {e}"); import traceback; traceback.print_exc()