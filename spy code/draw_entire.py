# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import joblib # 用于加载 scaler
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_absolute_error # 用于计算 MAE

# --- 中文显示设置 (与训练脚本一致) ---
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    print("[信息] 尝试设置字体为 'SimHei' 以支持中文。")
except Exception as e:
    print(f"[警告] 设置 'SimHei' 字体失败: {e}")
    print("[警告] 中文可能无法正常显示。")

print("TensorFlow 版本:", tf.__version__)

# --- 配置: 修改为你自己的路径 ---
# 1. 指向包含 cblen, real_xyz, sim_xyz 的数据文件
#    (通常是 main.py/change_uxuy_new.py 的输出文件)
DATA_FILE_PATH = 'D:/data/save_data/circle(u_new_4,cab=0.04,k=-20,a=1).xlsx' # <<< 修改这里

# 2. 指向保存了模型 (.keras) 和 scalers (.joblib) 的目录
#    (需要与 train_residual_model_25.py 中的 MODEL_SAVE_DIR 一致)
MODEL_SAVE_DIR = 'D:/data/dl_model_0_random_data(u_new_4,cab=0.04,k=-20,a=1)' # <<< 修改这里

# 3. 定义保存绘图的文件名 (可选)
PLOT_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'trajectory_comparison_with_residual.png') # <<< 修改这里 (可选)

# --- 1. 加载数据 ---
print(f"正在从以下路径加载数据: {DATA_FILE_PATH}")
if not os.path.exists(DATA_FILE_PATH):
    print(f"[错误] 数据文件未找到: {DATA_FILE_PATH}")
    exit()
try:
    df = pd.read_excel(DATA_FILE_PATH, engine='openpyxl')
    print(f"数据加载成功。形状: {df.shape}")
    # 检查必需的列
    required_cols = ['cblen1_mm', 'cblen2_mm', 'cblen3_mm',
                     'X_real_mm', 'Y_real_mm', 'Z_real_mm',
                     'sim_X_mm', 'sim_Y_mm', 'sim_Z_mm']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        print(f"[错误] 数据文件缺少必需的列: {missing}")
        exit()
except Exception as e:
    print(f"加载数据时出错: {e}")
    exit()

# --- 2. 加载模型和 Scalers ---
print(f"正在从以下目录加载模型和 Scalers: {MODEL_SAVE_DIR}")
model_path = os.path.join(MODEL_SAVE_DIR, 'residual_predictor_model.keras')
x_scaler_path = os.path.join(MODEL_SAVE_DIR, 'x_scaler.joblib')
y_scaler_path = os.path.join(MODEL_SAVE_DIR, 'y_scaler.joblib')

if not all(os.path.exists(p) for p in [model_path, x_scaler_path, y_scaler_path]):
    print("[错误] 找不到模型或 scaler 文件。请确保路径正确且训练已完成并保存。")
    print(f"  检查模型路径: {model_path}")
    print(f"  检查 X scaler 路径: {x_scaler_path}")
    print(f"  检查 Y scaler 路径: {y_scaler_path}")
    exit()

try:
    model = keras.models.load_model(model_path)
    x_scaler = joblib.load(x_scaler_path)
    y_scaler = joblib.load(y_scaler_path)
    print("模型和 Scalers 加载成功。")
except Exception as e:
    print(f"加载模型或 Scaler 时出错: {e}")
    exit()

# --- 3. 准备输入数据并进行预测 ---
print("准备输入数据并进行预测...")
# 提取缆绳长度作为模型输入特征
X_features = df[['cblen1_mm', 'cblen2_mm', 'cblen3_mm']].values

# 检查输入特征中是否有 NaN
if np.isnan(X_features).any():
    print("[警告] 输入特征包含 NaN 值。预测结果可能不准确。考虑处理 NaN 值。")
    # 可以选择填充或删除 NaN 行，但这里我们继续，模型可能会输出 NaN
    # X_features = df[['cblen1_mm', 'cblen2_mm', 'cblen3_mm']].dropna().values # 或者删除

# 使用加载的 x_scaler 标准化输入特征
X_features_scaled = x_scaler.transform(X_features)

# 使用加载的模型预测缩放后的残差
predicted_residuals_scaled = model.predict(X_features_scaled)

# 使用加载的 y_scaler 将预测的残差反向转换回原始尺度 (毫米)
predicted_residuals_mm = y_scaler.inverse_transform(predicted_residuals_scaled)
print(f"残差预测完成。形状: {predicted_residuals_mm.shape}")

# 将预测的残差添加到 DataFrame 中，方便后续计算和处理 NaN
df['pred_residual_X'] = predicted_residuals_mm[:, 0]
df['pred_residual_Y'] = predicted_residuals_mm[:, 1]
df['pred_residual_Z'] = predicted_residuals_mm[:, 2]

# --- 4. 计算修正后的仿真轨迹 ---
print("计算修正后的仿真轨迹...")
# 修正轨迹 = 原始仿真轨迹 + 预测的残差
# 注意：如果原始仿真值 (sim_X_mm 等) 是 NaN，结果也应该是 NaN
df['corrected_sim_X_mm'] = df['sim_X_mm'] + df['pred_residual_X']
df['corrected_sim_Y_mm'] = df['sim_Y_mm'] + df['pred_residual_Y']
df['corrected_sim_Z_mm'] = df['sim_Z_mm'] + df['pred_residual_Z']

# --- 5. 计算修正后的 MAE (可选，用于绘图标题) ---
# 需要处理 NaN 值，只在 real 和 corrected 都有效的地方计算 MAE
valid_indices = df[['X_real_mm', 'Y_real_mm', 'Z_real_mm',
                   'corrected_sim_X_mm', 'corrected_sim_Y_mm', 'corrected_sim_Z_mm']].dropna().index

if len(valid_indices) > 0:
    real_valid = df.loc[valid_indices, ['X_real_mm', 'Y_real_mm', 'Z_real_mm']].values
    corrected_valid = df.loc[valid_indices, ['corrected_sim_X_mm', 'corrected_sim_Y_mm', 'corrected_sim_Z_mm']].values

    mae_x_corrected = mean_absolute_error(real_valid[:, 0], corrected_valid[:, 0])
    mae_y_corrected = mean_absolute_error(real_valid[:, 1], corrected_valid[:, 1])
    mae_z_corrected = mean_absolute_error(real_valid[:, 2], corrected_valid[:, 2])
    # 计算 Overall 3D Euclidean Distance MAE (可选)
    mae_3d_corrected = np.mean(np.linalg.norm(real_valid - corrected_valid, axis=1))
    print(f"\n修正后 MAE (有效点): X={mae_x_corrected:.3f}mm, Y={mae_y_corrected:.3f}mm, Z={mae_z_corrected:.3f}mm, 3D={mae_3d_corrected:.3f}mm")
    plot_title = f'真实轨迹 vs 修正后仿真轨迹 (Overall 3D MAE: {mae_3d_corrected:.3f} mm)'
else:
    print("\n[警告] 没有足够的有效数据点来计算修正后的 MAE。")
    mae_x_corrected, mae_y_corrected, mae_z_corrected = np.nan, np.nan, np.nan
    plot_title = '真实轨迹 vs 修正后仿真轨迹 (无法计算 MAE)'


# --- 6. 绘制对比图 ---
print("正在绘制轨迹对比图...")
num_samples = len(df)
sample_index = np.arange(num_samples)

fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True) # 3 行 1 列

# X 轴对比
axs[0].plot(sample_index, df['X_real_mm'], label='真实 X (mm)', color='darkorange', linewidth=1.5)
axs[0].plot(sample_index, df['corrected_sim_X_mm'], label=f'修正后仿真 X (MAE={mae_x_corrected:.2f}mm)', color='royalblue', linestyle='--', linewidth=1)
axs[0].set_ylabel('X 坐标 (mm)')
axs[0].set_title('X 轴坐标对比')
axs[0].legend()
axs[0].grid(True, linestyle=':')

# Y 轴对比
axs[1].plot(sample_index, df['Y_real_mm'], label='真实 Y (mm)', color='darkorange', linewidth=1.5)
axs[1].plot(sample_index, df['corrected_sim_Y_mm'], label=f'修正后仿真 Y (MAE={mae_y_corrected:.2f}mm)', color='royalblue', linestyle='--', linewidth=1)
axs[1].set_ylabel('Y 坐标 (mm)')
axs[1].set_title('Y 轴坐标对比')
axs[1].legend()
axs[1].grid(True, linestyle=':')

# Z 轴对比
axs[2].plot(sample_index, df['Z_real_mm'], label='真实 Z (mm)', color='darkorange', linewidth=1.5)
axs[2].plot(sample_index, df['corrected_sim_Z_mm'], label=f'修正后仿真 Z (MAE={mae_z_corrected:.2f}mm)', color='royalblue', linestyle='--', linewidth=1)
axs[2].set_ylabel('Z 坐标 (mm)')
axs[2].set_title('Z 轴坐标对比')
axs[2].legend()
axs[2].grid(True, linestyle=':')

# 设置 X 轴标签和总标题
axs[2].set_xlabel('样本序号 (Sample Index)')
fig.suptitle(plot_title, fontsize=14)
fig.tight_layout(rect=[0, 0.03, 1, 0.96]) # 调整布局，防止标题重叠

# 保存或显示图像
if PLOT_SAVE_PATH:
    try:
        plt.savefig(PLOT_SAVE_PATH, dpi=300)
        print(f"轨迹对比图已保存至: {PLOT_SAVE_PATH}")
    except Exception as e:
        print(f"保存图像时出错: {e}")

print("显示图像...")
plt.show()

print("\n脚本执行完毕。")