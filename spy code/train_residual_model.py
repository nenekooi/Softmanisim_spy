import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib # 用于保存/加载 scaler
import matplotlib.pyplot as plt
import os

print("TensorFlow 版本:", tf.__version__)

# --- 配置 ---
# !!! 将这里替换成你的数据文件的实际路径 !!!
DATA_FILE_PATH = 'D:/data/save_data/random_data(k=-20,a=0.02)/random_data(k=-20,a=0.02).xlsx'
MODEL_SAVE_DIR = 'D:/data/dl_model' # 保存模型和 scaler 的目录
os.makedirs(MODEL_SAVE_DIR, exist_ok=True) # 如果目录不存在则创建

# --- 加载数据 ---
print(f"正在从以下路径加载数据: {DATA_FILE_PATH}")
try:
    df = pd.read_excel(DATA_FILE_PATH, engine='openpyxl')
    print(f"数据加载成功。形状: {df.shape}")
    print("数据列名:", df.columns.tolist())
except FileNotFoundError:
    print(f"错误: 在 {DATA_FILE_PATH} 未找到数据文件")
    exit()
except Exception as e:
    print(f"加载数据时出错: {e}")
    exit()

# --- 计算残差 (目标变量) ---
print("正在计算残差...")
df['residual_X'] = df['X_real_mm'] - df['sim_X_mm']
df['residual_Y'] = df['Y_real_mm'] - df['sim_Y_mm']
df['residual_Z'] = df['Z_real_mm'] - df['sim_Z_mm']

# 检查由于仿真失败可能引入的 NaN 值
nan_rows = df[df[['residual_X', 'residual_Y', 'residual_Z']].isna().any(axis=1)]
if not nan_rows.empty:
    print(f"\n警告: 发现 {len(nan_rows)} 行包含 NaN 残差 (可能是仿真问题导致的)。")
    print("原始数据形状:", df.shape)
    df = df.dropna(subset=['residual_X', 'residual_Y', 'residual_Z']) # 删除包含 NaN 残差的行
    print("删除 NaN 残差后的形状:", df.shape)
    if df.empty:
        print("错误: 删除 NaN 后没有有效数据剩余。请检查仿真输出。")
        exit()

# --- 定义特征 (输入) 和标签 (目标) ---
feature_columns = ['cblen1_mm', 'cblen2_mm', 'cblen3_mm'] # 输入特征是三个绳长
label_columns = ['residual_X', 'residual_Y', 'residual_Z']   # 输出目标是三个残差

X = df[feature_columns].values # 提取特征数据，转换为 NumPy 数组
y = df[label_columns].values   # 提取标签数据，转换为 NumPy 数组

print(f"\n特征 (X) 形状: {X.shape}") # 应该是 (样本数, 3)
print(f"标签 (y) 形状: {y.shape}")     # 应该是 (样本数, 3)

# --- 划分数据 (70% 训练集, 15% 验证集, 15% 测试集) ---
print("\n正在划分数据...")
# 首先划分为训练集 (70%) 和临时集 (30%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42 # 使用 random_state 保证每次划分结果一致
)
# 再将临时集 (30%) 划分为验证集 (15%) 和测试集 (15%)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42 # 0.5 * 0.3 = 0.15 (占总数据的 15%)
)

print(f"训练集形状: X={X_train.shape}, y={y_train.shape}")
print(f"验证集形状: X={X_val.shape}, y={y_val.shape}")
print(f"测试集形状: X={X_test.shape}, y={y_test.shape}")

# --- 使用 StandardScaler 进行数据缩放 ---
print("\n正在缩放数据...")
# 初始化 scaler
x_scaler = StandardScaler() # 输入特征的 scaler
y_scaler = StandardScaler() # 输出标签的 scaler (使用独立的 scaler)

# *仅仅* 在训练数据上拟合 (fit) scaler
X_train_scaled = x_scaler.fit_transform(X_train) # 拟合并转换训练特征
y_train_scaled = y_scaler.fit_transform(y_train) # 拟合并转换训练标签

# 使用*已拟合*的 scaler 转换 (transform) 验证集和测试集
X_val_scaled = x_scaler.transform(X_val)    # 只转换验证特征
X_test_scaled = x_scaler.transform(X_test)   # 只转换测试特征
y_val_scaled = y_scaler.transform(y_val)    # 只转换验证标签
y_test_scaled = y_scaler.transform(y_test)   # 只转换测试标签

print("数据缩放完成。")
print("示例缩放后的 X_train[0]:", X_train_scaled[0])
print("示例缩放后的 y_train[0]:", y_train_scaled[0])

# --- 保存 scaler ---
# 保存 scaler 非常重要，因为之后在新数据上做预测时需要用它们
x_scaler_path = os.path.join(MODEL_SAVE_DIR, 'x_scaler.joblib')
y_scaler_path = os.path.join(MODEL_SAVE_DIR, 'y_scaler.joblib')
joblib.dump(x_scaler, x_scaler_path)
joblib.dump(y_scaler, y_scaler_path)
print(f"输入特征 scaler 已保存至: {x_scaler_path}")
print(f"输出标签 scaler 已保存至: {y_scaler_path}")

# --- 构建神经网络模型 ---
print("\n正在构建模型...")

def build_mlp_model(input_shape):
    """构建一个简单的 MLP 模型"""
    model = keras.Sequential([
        # 使用英文名称替换之前的中文名称
        layers.Input(shape=input_shape, name='input_layer'), # 输入层
        layers.Dense(64, activation='relu', name='hidden_layer_1'), # 隐藏层 1
        layers.Dense(128, activation='relu', name='hidden_layer_2'),# 隐藏层 2
        # layers.Dense(128, activation='relu', name='hidden_layer_3'), # 可选: 如果需要可以增加更多层
        layers.Dense(64, activation='relu', name='hidden_layer_4'), # 隐藏层 4
        layers.Dense(3, name='output_layer') # 输出层 (保持不变，因为已经是英文)
    ])

    # 编译模型
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001) # 使用 Adam 优化器
    model.compile(
        optimizer=optimizer,
        loss='mean_squared_error', # 均方误差损失
        metrics=['mean_absolute_error'] # 平均绝对误差
    )
    return model

# Get input shape from scaled training data (这部分代码不变)
input_shape = (X_train_scaled.shape[1],) # 形状应该是 (3,)

model = build_mlp_model(input_shape)

# Print model summary (这部分代码不变)
model.summary()

# --- 训练模型 ---
print("\n正在训练模型...")

# 定义 EarlyStopping 回调函数
# monitor='val_loss': 监控验证集上的损失 (MSE)
# patience=20: 如果验证损失在 20 个 epoch 内没有改善，则停止训练
# restore_best_weights=True: 训练结束后，恢复到验证损失最佳的那次迭代的模型权重
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=20, # 可以根据需要调整这个值 (例如 10-50)
    restore_best_weights=True,
    verbose=1 # 打印早停信息
)

# 设置训练参数
EPOCHS = 200 # 最大训练轮数 (epoch)
BATCH_SIZE = 32 # 每批处理的样本数，可根据数据集大小和内存调整

# 开始训练
# history 对象会记录训练过程中的损失和评估指标
history = model.fit(
    X_train_scaled, y_train_scaled,           # 训练数据 (已缩放)
    epochs=EPOCHS,                            # 最大轮数
    batch_size=BATCH_SIZE,                    # 批大小
    validation_data=(X_val_scaled, y_val_scaled), # 验证数据 (已缩放)
    callbacks=[early_stopping],               # 使用早停法
    verbose=1                                 # verbose=1 显示进度条, 2 每轮一行, 0 静默
)

print("\n训练完成。")

# --- 绘制训练历史曲线 (损失和 MAE) ---
def plot_history(history):
    """绘制训练和验证的损失及 MAE 曲线"""
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.xlabel('Epoch (训练轮数)')
    plt.ylabel('Mean Squared Error (均方误差损失)')
    plt.plot(hist['epoch'], hist['loss'], label='训练集损失')
    plt.plot(hist['epoch'], hist['val_loss'], label='验证集损失')
    plt.title('损失 (MSE) vs. 训练轮数')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.xlabel('Epoch (训练轮数)')
    plt.ylabel('Mean Absolute Error (平均绝对误差)')
    plt.plot(hist['epoch'], hist['mean_absolute_error'], label='训练集 MAE')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'], label='验证集 MAE')
    plt.title('平均绝对误差 (MAE) vs. 训练轮数')
    plt.legend()
    plt.grid(True)

    plt.tight_layout() # 调整子图布局
    # 保存图像
    plot_path = os.path.join(MODEL_SAVE_DIR, 'training_history_chinese.png')
    plt.savefig(plot_path)
    print(f"\n训练历史曲线图已保存至: {plot_path}")
    plt.show() # 显示图像

# 调用函数绘制曲线
plot_history(history)

# --- 在测试集上评估模型 ---
print("\n正在测试集上评估模型...")

# 使用缩放后的数据进行评估 (loss 是 MSE, metrics 是 MAE)
test_loss, test_mae_scaled = model.evaluate(X_test_scaled, y_test_scaled, verbose=0) # verbose=0 不打印评估过程
print(f"测试集损失 (MSE, 缩放后): {test_loss:.4f}")
print(f"测试集平均绝对误差 (MAE, 缩放后): {test_mae_scaled:.4f}")

# 在缩放后的测试集上进行预测
y_pred_scaled = model.predict(X_test_scaled)

# 将预测结果和真实标签反向转换回原始尺度 (毫米)
y_pred_mm = y_scaler.inverse_transform(y_pred_scaled)  # 反向转换预测值
y_test_mm = y_scaler.inverse_transform(y_test_scaled) # 反向转换真实测试标签 (结果与 y_test 相同)

# 计算原始尺度 (毫米) 上的 MAE
mae_mm = np.mean(np.abs(y_pred_mm - y_test_mm), axis=0) # 分别计算 X, Y, Z 轴的 MAE
mean_mae_mm = np.mean(mae_mm) # 计算三个轴的平均 MAE

print("\n--- 以毫米为单位的评估结果 ---")
print(f"测试集 MAE (X 残差, 毫米): {mae_mm[0]:.4f}")
print(f"测试集 MAE (Y 残差, 毫米): {mae_mm[1]:.4f}")
print(f"测试集 MAE (Z 残差, 毫米): {mae_mm[2]:.4f}")
print(f"测试集平均 MAE (X, Y, Z 残差的平均值, 毫米): {mean_mae_mm:.4f}")
print("\n解释: 这个结果表示，对于模型从未见过的数据，")
print(f"它预测的 X, Y, 或 Z 轴残差平均会偏离真实残差约 {mean_mae_mm:.4f} 毫米。")

# 可选：绘制预测值 vs 真实值的散点图 (例如 Z 轴残差)
plt.figure(figsize=(6, 6))
plt.scatter(y_test_mm[:, 2], y_pred_mm[:, 2], alpha=0.5) # Z 轴是第 3 列 (索引为 2)
plt.xlabel('真实 Z 轴残差 (毫米)')
plt.ylabel('预测 Z 轴残差 (毫米)')
plt.title('预测值 vs. 真实值 (Z 轴残差, 测试集)')
# 添加 y=x 参考线
lims = [np.min([plt.xlim(), plt.ylim()]), np.max([plt.xlim(), plt.ylim()])] # 获取当前坐标轴范围
plt.plot(lims, lims, 'r--', alpha=0.75, zorder=0) # 绘制红色虚线
plt.xlim(lims) # 设置 x 轴范围
plt.ylim(lims) # 设置 y 轴范围
plt.grid(True)
scatter_path = os.path.join(MODEL_SAVE_DIR, 'prediction_scatter_Z_chinese.png')
plt.savefig(scatter_path) # 保存散点图
print(f"预测散点图已保存至: {scatter_path}")
plt.show()

# --- 保存训练好的模型 ---
model_path = os.path.join(MODEL_SAVE_DIR, 'residual_predictor_model_chinese.keras')
model.save(model_path) # 保存整个模型（结构、权重、优化器状态）
print(f"\n训练好的模型已成功保存至: {model_path}")