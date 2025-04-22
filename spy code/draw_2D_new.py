# -*- coding: utf-8 -*-
# ==============================================================================
# 脚本功能:
# 读取包含仿真结果(mm)和真实数据(mm)的 XLSX 文件，
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
# !!! 请确保这个路径指向你的包含 sim 和 real 数据的 XLSX 文件 !!!
# !!! 注意：默认读取 Excel 文件的第一个工作表 (sheet) !!!
INPUT_FILE_PATH = 'c:/Users/11647/Desktop/data/circle2_without_0(k=-20,act=0.02)_env_reverted_action——1.xlsx' # <<< 更改为你的 XLSX 文件路径
# 如果数据不在第一个工作表，请取消下面一行的注释并修改工作表名称或索引
# SHEET_NAME = 'Sheet1' # 或者 SHEET_NAME = 0

# --- 核心逻辑 ---

if __name__ == "__main__":
    print(f"--- 开始处理与绘图 ---")
    print(f"[信息] 尝试加载数据文件: {INPUT_FILE_PATH}")

    if not os.path.exists(INPUT_FILE_PATH):
        print(f"[错误] 文件未找到: {INPUT_FILE_PATH}")
        sys.exit(1)

    try:
        # --- 加载数据 ---
        # 使用 pd.read_excel 读取 XLSX 文件
        # 如果指定了 SHEET_NAME，则使用它
        if 'SHEET_NAME' in locals():
            df = pd.read_excel(INPUT_FILE_PATH, sheet_name=SHEET_NAME)
            print(f"[信息] 正在从工作表 '{SHEET_NAME}' 加载数据...")
        else:
            df = pd.read_excel(INPUT_FILE_PATH) # 默认读取第一个工作表
            print("[信息] 正在从第一个工作表加载数据...")

        print(f"[成功] 已加载数据 {len(df)} 行。")
        print(f"[信息] 数据列: {df.columns.tolist()}") # 打印列名以便检查

        sim_cols = ['sim_X_mm', 'sim_Y_mm', 'sim_Z_mm']
        real_cols = ['X_real_mm', 'Y_real_mm', 'Z_real_mm']
        required_cols = sim_cols + real_cols

        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"错误: 文件缺少必需列: {missing_cols}。请检查 Excel 文件中的列名是否完全匹配。")
        print("[信息] 所需数据列均存在。")

        # --- 获取数据 ---
        # 确保数据类型是数值型，以防Excel中存在非数值单元格
        try:
            sim_x = pd.to_numeric(df['sim_X_mm'], errors='coerce')
            sim_y = -pd.to_numeric(df['sim_Y_mm'], errors='coerce') # 注意这里的负号
            sim_z = pd.to_numeric(df['sim_Z_mm'], errors='coerce')
            real_x = pd.to_numeric(df['X_real_mm'], errors='coerce')
            real_y = pd.to_numeric(df['Y_real_mm'], errors='coerce')
            real_z = pd.to_numeric(df['Z_real_mm'], errors='coerce')
        except KeyError as e:
             raise ValueError(f"错误: 访问列时出错 - {e}。请再次确认列名。")

        # 将包含转换错误的行标记出来，并可能在后续处理中排除它们
        if sim_x.isnull().any() or sim_y.isnull().any() or sim_z.isnull().any() or \
           real_x.isnull().any() or real_y.isnull().any() or real_z.isnull().any():
            print("[警告] 部分单元格未能成功转换为数值型 (可能包含文本或为空)，这些行将在误差计算中被忽略。")
            # 在绘图时，含有NaN的点不会被绘制出来，这是 matplotlib 的默认行为

        # 创建样本索引作为第一个图的 X 轴
        sample_index = range(len(df))

        # --- 计算误差 (处理潜在的 NaN 值) ---
        print("[信息] 正在计算误差...")

        # 创建一个包含所有需要计算误差的列的 DataFrame副本
        # 直接使用上面转换过的 Series 来创建，这样 NaN 已经存在
        error_df = pd.DataFrame({
            'sim_X_mm': sim_x, 'sim_Y_mm': df['sim_Y_mm'], 'sim_Z_mm': sim_z, # 注意 sim_y 用原始值计算误差
            'X_real_mm': real_x, 'Y_real_mm': real_y, 'Z_real_mm': real_z
        })

        # 删除任何包含 NaN 的行，以确保误差计算的有效性
        rows_before_dropna = len(error_df)
        error_df.dropna(inplace=True)
        rows_after_dropna = len(error_df)

        mae_x, mae_y, mae_z = np.nan, np.nan, np.nan
        mae_xy = np.nan # 初始化为 NaN
        mae_3d = np.nan # 初始化为 NaN

        if rows_before_dropna > rows_after_dropna:
            print(f"[警告] 数据中包含 NaN 值或非数值数据，已忽略 {rows_before_dropna - rows_after_dropna} 行数据后进行误差计算。")

        if rows_after_dropna == 0:
              print("[错误] 清理 NaN 后无有效数据可用于计算误差。")
              # 可以选择退出或继续绘图但不显示误差
        else:
            # 提取清理后的数据用于误差计算
            sim_x_clean = error_df['sim_X_mm']
            sim_y_clean = error_df['sim_Y_mm'] # 使用原始 sim_Y 进行误差计算
            sim_z_clean = error_df['sim_Z_mm']
            real_x_clean = error_df['X_real_mm']
            real_y_clean = error_df['Y_real_mm']
            real_z_clean = error_df['Z_real_mm']

            # 1. 计算各轴绝对误差
            abs_err_x = np.abs(real_x_clean - sim_x_clean)
            abs_err_y = np.abs(real_y_clean - sim_y_clean) # 直接比较
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
        # 使用原始的 sim_x, real_x (包含可能的NaN，绘图时会自动忽略)
        axs[0].plot(sample_index, sim_x, label='仿真 X (mm)', color='blue', linestyle='--')
        axs[0].plot(sample_index, real_x, label='真实 X (mm)', color='darkorange', linestyle='-')
        axs[0].set_ylabel('X 坐标 (mm)')
        axs[0].set_title('X 坐标对比')
        axs[0].legend()
        axs[0].grid(True)
        # 仅当 mae_x 不是 NaN 时显示
        if not np.isnan(mae_x):
            axs[0].text(0.02, 0.95, f'MAE_X: {mae_x:.3f} mm',
                        transform=axs[0].transAxes, fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

        # --- 绘制 Y 坐标对比图 ---
        # 注意：绘图时使用反转后的 sim_y
        axs[1].plot(sample_index, sim_y, label='仿真 Y (mm)', color='blue', linestyle='--') # sim_y 已被反转
        axs[1].plot(sample_index, real_y, label='真实 Y (mm)', color='darkorange', linestyle='-')
        axs[1].set_ylabel('Y 坐标 (mm)')
        axs[1].set_title('Y 坐标对比')
        axs[1].legend()
        axs[1].grid(True)
        # 仅当 mae_y 不是 NaN 时显示
        if not np.isnan(mae_y):
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
        # 仅当 mae_z 不是 NaN 时显示
        if not np.isnan(mae_z):
            axs[2].text(0.02, 0.95, f'MAE_Z: {mae_z:.3f} mm',
                        transform=axs[2].transAxes, fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

        # 调整子图布局
        fig1.tight_layout(rect=[0, 0.03, 1, 0.94]) # 稍微调整 rect 避免标题和总标题重叠
        print("[信息] 第一个图表已准备好。")


        # ==============================================================
        # --- 开始绘制第二个图：X-Y 二维轨迹 ---
        # ==============================================================
        print("[信息] 正在生成第二个对比图 (X-Y 轨迹)...")
        fig2, ax2 = plt.subplots(1, 1, figsize=(8, 8)) # 使用 fig2, ax2

        # --- 绘制 X-Y 轨迹 ---
        # 使用原始的 sim_x, real_x 和反转后的 sim_y, 原始 real_y
        ax2.plot(sim_x, sim_y, label=f'仿真轨迹 (XY)', color='blue', linestyle='--') # sim_y 已反转
        ax2.plot(real_x, real_y, label=f'真实轨迹 (XY)', color='darkorange', linestyle='-')

        # --- 设置标签和标题 ---
        ax2.set_xlabel('X 坐标 (mm)')
        ax2.set_ylabel('Y 坐标 (mm)')
        # 仅当 mae_xy 不是 NaN 时显示
        if not np.isnan(mae_xy):
            ax2.set_title(f'仿真 vs 真实 X-Y 轨迹对比\nXY平面平均距离误差 (MAE_XY): {mae_xy:.3f} mm', fontsize=14)
        else:
            ax2.set_title(f'仿真 vs 真实 X-Y 轨迹对比\n(无法计算 XY 平面误差)', fontsize=14)

        ax2.legend()
        ax2.grid(True)

        # --- 设置等轴比例 ---
        ax2.set_aspect('equal', adjustable='box')

        # 调整布局
        fig2.tight_layout() # 使用 fig2
        print("[信息] 第二个图表已准备好。")


        # ==============================================================
        # --- 显示所有图表 ---
        # ==============================================================
        print("[信息] 显示所有图表窗口...")
        plt.show()

        print("--- 所有绘图结束 ---")

    except FileNotFoundError: print(f"[错误] 无法找到文件: {INPUT_FILE_PATH}")
    except ValueError as e: print(f"[错误] 数据读取或处理时出错: {e}")
    except ImportError:
        print("[错误] 缺少必要的库。请确保已安装 pandas, matplotlib, numpy 和 openpyxl。")
        print("请尝试运行: pip install pandas matplotlib numpy openpyxl")
        sys.exit(1)
    except Exception as e:
        print(f"[错误] 发生未知错误: {e}")
        import traceback
        traceback.print_exc()