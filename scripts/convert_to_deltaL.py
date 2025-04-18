# -*- coding: utf-8 -*-
# ==============================================================================
# 脚本功能:
# 读取包含伺服电机编码器数据的文件 (CSV 或 Excel)，
# 根据指定的物理参数 (卷线轮半径、电机分辨率) 和初始位置假设，
# 计算每个电机对应的绳长变化量 (Delta L)，
# 并将原始编码器数据和计算出的 Delta L 保存到一个新的 Excel 文件中。
# ==============================================================================

# 导入所需的库
import pandas as pd
import math
import os

# --- 参数设置 (!!! 必须根据你的实际硬件进行验证和设置 !!!) ---

# 卷线轮半径 (单位: 毫米 mm) - 参照论文中的 r = 6 mm
# !!! 请确认这个值是否与你的机器人硬件完全一致 !!!
SPOOL_RADIUS_MM = 6.0

# 电机编码器分辨率和对应的角度范围
# 假设使用 Dynamixel XL-320 (参照论文)
# 分辨率: 1024 steps (值为 0 到 1023)
# 对应角度范围: 300 度
# !!! 如果你使用的电机型号或工作模式不同，必须修改下面两行 !!!
ENCODER_RESOLUTION = 1024.0  # 电机在一个完整角度范围内的总步数 (例如 1024 或 4096)
DEGREES_PER_RANGE = 300.0    # 该分辨率对应的总角度范围 (单位: 度, 例如 300 或 360)

# --- 文件路径 (!!! 如果你的文件不在这个位置或名称不同，请修改 !!!) ---

# 你的输入数据文件路径 (可以是 .csv 或 .xlsx 格式)
# 脚本会根据文件的实际扩展名自动选择读取方式
# !!! 请确保这个路径正确，并且文件确实包含 'actuator_1', 'actuator_2', 'actuator_3' 列 !!!
INPUT_FILE_PATH = 'c:/Users/11647/SoftManiSim/code/Modeling/Processed_Data3w_20250318.xlsx'
# 示例：如果你要处理的是 Excel 文件，路径可能像这样:
# INPUT_FILE_PATH = 'SoftManiSim/code/Modeling/processed data 3w 20250318.xlsx'
# 示例：如果文件在脚本同目录下:
# INPUT_FILE_PATH = 'my_data.csv'

# 输出 Excel 文件的路径 (脚本默认在输入文件相同目录下生成此文件)
# 你也可以指定一个全新的绝对路径或相对路径
output_dir = os.path.dirname(INPUT_FILE_PATH) if os.path.dirname(INPUT_FILE_PATH) else '.' # 获取输入文件所在目录，如果是当前目录则为'.'
OUTPUT_FILE_PATH = os.path.join(output_dir, 'Processed_Data_with_DeltaL.xlsx') # 组合输出文件的完整路径

# --- 核心转换函数 (Encoder Value to Delta L) ---

def encoder_to_delta_L(current_encoder_value, initial_encoder_value,
                       spool_radius, encoder_resolution, degrees_per_range):
    """
    将伺服电机编码值的变化转换为绳长的变化量 (Delta L)。

    参数:
        current_encoder_value (int/float): 当前的电机编码器读数。
        initial_encoder_value (int/float): 电机在参考状态 (Delta L = 0) 时的编码器读数。
        spool_radius (float): 卷线轮的半径 (单位与输出长度单位一致, 如 mm)。
        encoder_resolution (float): 电机编码器的总步数 (例如 1024, 4096)。
        degrees_per_range (float): 编码器分辨率对应的总角度范围 (单位: 度)。

    返回:
        float: 绳长的变化量 (Delta L)，单位与 spool_radius 一致 (例如 mm)。
    """
    try:
        # 步骤 1: 计算编码器值的变化量 (确保转为浮点数进行计算)
        # 注意：这里假设了编码器值是连续增加或减少的，没有显式处理环绕(wrap-around)的情况。
        # 对于从一个固定初始点计算相对变化量，这通常是足够的。
        delta_encoder = float(current_encoder_value) - float(initial_encoder_value)

        # 步骤 2: 将编码器变化量转换为角度变化量 (单位: 度)
        delta_theta_deg = delta_encoder * (degrees_per_range / encoder_resolution)

        # 步骤 3: 将角度变化量从 度 转换为 弧度
        delta_theta_rad = math.radians(delta_theta_deg)
        # 备用计算方式: delta_theta_rad = delta_theta_deg * (math.pi / 180.0)

        # 步骤 4: 应用公式 Delta L = r * Delta Theta 计算绳长变化量
        delta_L = spool_radius * delta_theta_rad

        return delta_L
    except Exception as e:
        # 如果计算过程中出现错误 (例如输入值非数字)，返回 NaN 并打印错误
        print(f"警告: 在计算 Delta_L 时发生错误: {e}. "
              f"当前值: {current_encoder_value}, 初始值: {initial_encoder_value}. 返回 NaN.")
        return float('nan')


# --- 主脚本执行逻辑 ---
# 使用 if __name__ == "__main__": 确保代码块只在直接运行脚本时执行
if __name__ == "__main__":
    try:
        # 步骤 1: 检查输入文件是否存在并根据扩展名加载数据
        print(f"--- 开始处理 ---")
        print(f"[信息] 正在检查输入文件: {INPUT_FILE_PATH}")
        if not os.path.exists(INPUT_FILE_PATH):
            # 如果文件不存在，则抛出 FileNotFoundError 异常
            raise FileNotFoundError(f"错误: 输入文件未找到，请确认路径是否正确: {INPUT_FILE_PATH}")

        # 获取文件扩展名 (转换为小写以兼容 .CSV, .XLSX 等)
        file_extension = os.path.splitext(INPUT_FILE_PATH)[1].lower()
        print(f"[信息] 检测到文件扩展名为: {file_extension}")

        # 根据扩展名选择合适的 pandas 读取函数
        if file_extension == '.csv':
            # 读取 CSV 文件
            df = pd.read_csv(INPUT_FILE_PATH)
            print(f"[成功] 已从 CSV 文件加载数据，共 {len(df)} 行。")
        elif file_extension == '.xlsx':
            # 读取 Excel 文件，需要安装 openpyxl (`pip install openpyxl`)
            # 假设数据在第一个工作表 (sheet_name=0 或 'Sheet1')
            try:
                df = pd.read_excel(INPUT_FILE_PATH, engine='openpyxl')
                print(f"[成功] 已从 Excel 文件加载数据，共 {len(df)} 行。")
            except ImportError:
                 # 如果缺少 openpyxl 库，则给出提示
                raise ImportError("错误: 读取 Excel 文件需要 'openpyxl' 库。请使用 'pip install openpyxl' 命令安装。")
        else:
            # 如果文件扩展名不支持，则抛出 ValueError 异常
            raise ValueError(f"错误: 不支持的文件扩展名: '{file_extension}'。脚本目前仅支持 '.csv' 和 '.xlsx' 文件。")

        # 步骤 2: 检查必需的 'actuator' 列是否存在
        required_columns = ['actuator_1', 'actuator_2', 'actuator_3']
        print(f"[信息] 正在检查文件是否包含必需的列: {required_columns}")
        if not all(col in df.columns for col in required_columns):
            # 如果缺少列，找出具体缺少的列名并报错
            missing_cols = [col for col in required_columns if col not in df.columns]
            raise ValueError(f"错误: 输入文件 {os.path.basename(INPUT_FILE_PATH)} 中缺少以下必需的列: {missing_cols}")
        print("[成功] 所有必需的 'actuator' 列均存在。")

        # 步骤 3: 确定初始参考编码器位置
        # !!! 关键假设: 数据文件的第一行对应 Delta_L = 0 的状态 !!!
        # !!! 如果你的机器人零点不是第一行记录的状态，或者你有特定的初始值，请务必修改这里的逻辑 !!!
        if len(df) > 0:
            # 提取第一行数据作为初始参考值
            initial_servo_pos1 = df['actuator_1'].iloc[0]
            initial_servo_pos2 = df['actuator_2'].iloc[0]
            initial_servo_pos3 = df['actuator_3'].iloc[0]
            print(f"[假设] 使用数据文件的第一行作为初始参考位置 (假定此时 Delta_L=0):")
            print(f"  初始电机位置 1 (actuator_1): {initial_servo_pos1}")
            print(f"  初始电机位置 2 (actuator_2): {initial_servo_pos2}")
            print(f"  初始电机位置 3 (actuator_3): {initial_servo_pos3}")
        else:
            # 如果数据文件为空，则无法继续
            raise ValueError("错误: 数据文件为空，无法确定初始参考位置或进行后续计算。")

        # 步骤 4: 对每一行数据应用转换函数，计算 Delta_L
        print("[信息] 正在为 actuator_1, actuator_2, actuator_3 计算对应的 Delta_L (绳长变化量，单位: mm)...")
        # 为 'actuator_1' 列的每个值计算 'delta_L1_mm'
        df['delta_L1_mm'] = df['actuator_1'].apply(
            lambda current_pos: encoder_to_delta_L(
                current_pos, initial_servo_pos1, SPOOL_RADIUS_MM, ENCODER_RESOLUTION, DEGREES_PER_RANGE
            )
        )
        # 为 'actuator_2' 列的每个值计算 'delta_L2_mm'
        df['delta_L2_mm'] = df['actuator_2'].apply(
            lambda current_pos: encoder_to_delta_L(
                current_pos, initial_servo_pos2, SPOOL_RADIUS_MM, ENCODER_RESOLUTION, DEGREES_PER_RANGE
            )
        )
        # 为 'actuator_3' 列的每个值计算 'delta_L3_mm'
        df['delta_L3_mm'] = df['actuator_3'].apply(
            lambda current_pos: encoder_to_delta_L(
                current_pos, initial_servo_pos3, SPOOL_RADIUS_MM, ENCODER_RESOLUTION, DEGREES_PER_RANGE
            )
        )
        print("[成功] Delta_L 计算完成。")

        # 步骤 5: 选择要保存到新 Excel 文件中的列
        # 默认包含原始 actuator 列和新计算的 delta_L 列
        output_columns = ['actuator_1', 'actuator_2', 'actuator_3', 'delta_L1_mm', 'delta_L2_mm', 'delta_L3_mm']

        # 可选: 检查并添加其他你想在输出文件中保留的原始列 (例如时间戳、位姿数据等)
        # 这里我们尝试保留所有原始列，并将 delta_L 列添加到末尾
        original_columns = df.columns.tolist()
        # 移除可能已存在的 delta_L 列名（以防万一），然后添加新的 delta_L 列名
        final_output_columns = [col for col in original_columns if not col.startswith('delta_L')]
        final_output_columns.extend(['delta_L1_mm', 'delta_L2_mm', 'delta_L3_mm'])

        # 创建只包含选定列的新 DataFrame
        # 使用 .reindex() 可以保证列的顺序，即使某些可选列不存在也不会报错
        output_df = df.reindex(columns=final_output_columns).copy() # 使用 .copy() 避免潜在的 SettingWithCopyWarning

        # 步骤 6: 将结果保存到新的 Excel 文件
        print(f"[信息] 正在将包含 Delta_L 的结果保存到 Excel 文件: {OUTPUT_FILE_PATH}")
        # 使用 'openpyxl' 引擎来写入 .xlsx 文件 (需要先安装: pip install openpyxl)
        output_df.to_excel(OUTPUT_FILE_PATH, index=False, engine='openpyxl')
        print("-" * 50)
        print("[完成] 处理成功！")
        print(f"结果已保存至: {os.path.abspath(OUTPUT_FILE_PATH)}") # 打印绝对路径以便查找
        print("-" * 50)

    # --- 统一错误处理 ---
    except FileNotFoundError as e:
        # 文件未找到错误
        print("\n" + "="*15 + " 文件错误 " + "="*15)
        print(e)
        print("请执行以下检查:")
        print("1. 确认脚本中的 'INPUT_FILE_PATH' 变量设置是否正确。")
        print("2. 确认该路径下的文件确实存在。")
        print("3. 确认脚本运行的当前工作目录是否正确（如果使用的是相对路径）。")
        print("="*40)
    except ValueError as e:
        # 数据或逻辑错误 (例如缺少列、文件为空、文件格式不对等)
        print("\n" + "="*15 + " 数据或逻辑错误 " + "="*15)
        print(e)
        print("请执行以下检查:")
        print("1. 确认输入文件是否包含必需的 'actuator_1', 'actuator_2', 'actuator_3' 列。")
        print("2. 确认输入文件不是空的。")
        print("3. 如果错误信息与初始位置相关，请检查关于初始位置的假设是否合理，或修改代码指定正确的初始值。")
        print("4. 确认文件扩展名是否为脚本支持的 .csv 或 .xlsx。")
        print("="*40)
    except ImportError as e:
        # 缺少必要的 Python 库 (例如 pandas, openpyxl)
        print("\n" + "="*15 + " 库导入错误 " + "="*15)
        print(f"错误: {e}")
        if 'openpyxl' in str(e).lower():
            print("看起来运行此脚本需要 'openpyxl' 库来读/写 Excel 文件 (.xlsx)。")
            print("请在你的 Python 环境中运行以下命令来安装它:")
            print("  pip install openpyxl")
        elif 'pandas' in str(e).lower():
             print("看起来运行此脚本需要 'pandas' 库。")
             print("请在你的 Python 环境中运行以下命令来安装它:")
             print("  pip install pandas")
        else:
             print("请确保已安装所有必需的 Python 库。")
        print("="*40)
    except Exception as e:
        # 捕获所有其他未能预料的异常
        print("\n" + "="*15 + " 未知错误 " + "="*15)
        print(f"脚本执行过程中发生了一个预期之外的错误: {type(e).__name__} - {e}")
        print("请仔细检查错误信息、代码逻辑以及输入数据是否有异常。")
        # 可以考虑在这里添加更详细的错误追踪信息 (import traceback; traceback.print_exc())
        print("="*40)