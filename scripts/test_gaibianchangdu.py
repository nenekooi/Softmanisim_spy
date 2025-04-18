# -*- coding: utf-8 -*-
# ==============================================================================
# 脚本功能:
# 创建并静态可视化一个具有指定初始长度 L0 的单段连续体机器人。
# 通过修改脚本顶部的 initial_length 参数来改变显示的机器人长度。
# !!! 重要: 此脚本现在假设您已经按要求修改了 visualizer.py 中的 ODE 类 !!!
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
# 不再需要 pandas

# 动态添加 SoftManiSim 文件夹到 sys.path
try:
    softmanisim_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if softmanisim_path not in sys.path: sys.path.append(softmanisim_path); print(f"[调试] 已添加路径: {softmanisim_path}")
except NameError: print("[警告] 无法自动确定项目根目录，请确保 visualizer 模块在 Python 路径中。")

# --- 检查 visualizer 模块导入 ---
try:
    from visualizer.visualizer import ODE # 导入的是修改后的 ODE 类
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

# --- 主程序 ---
if __name__ == "__main__":

    # --- 参数设置 ---
    print("--- 设置参数 ---")
    # 1. 机器人物理参数 (!!! 在这里修改机器人长度 !!!)
    initial_length = 0.2   # <<< !!! 在这里设置您想测试的机器人总长度 (单位: 米) !!!
    cable_distance = 7.5e-3 # 绳索到中心线的距离 d (单位: 米)
    number_of_segment = 1 # 固定为单段
    if number_of_segment <= 0: print("错误: 段数必须为正。"); sys.exit(1)
    L0_seg = initial_length / number_of_segment # 每段长度 = 总长度
    print(f"机器人参数: 期望总长度L0={initial_length:.4f}m, 绳索半径d={cable_distance:.4f}m")

    # 2. 可视化参数
    body_color = [0.8, 0.1, 0.1, 1]; head_color = [0.1, 0.1, 0.8, 1]
    body_sphere_radius = 0.02; number_of_sphere = 30
    my_sphere_radius = body_sphere_radius; my_number_of_sphere = number_of_sphere
    my_head_color = head_color

    # --- PyBullet 初始化 ---
    print("--- 初始化 PyBullet ---")
    simulationStepTime = 0.01; physicsClientId = -1
    try:
        physicsClientId = p.connect(p.GUI);
        if physicsClientId < 0: raise ConnectionError("未能连接。")
        print(f"连接 PyBullet, Client ID: {physicsClientId}")
    except Exception as e: print(f"连接 PyBullet 出错: {e}"); sys.exit(1)
    p.setAdditionalSearchPath(pybullet_data.getDataPath()); p.setGravity(0, 0, -9.81); p.setTimeStep(simulationStepTime)
    try: planeId = p.loadURDF("plane.urdf"); print(f"加载 plane.urdf, ID: {planeId}")
    except p.error as e: print(f"加载 plane.urdf 出错: {e}"); p.disconnect(physicsClientId); sys.exit(1)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
    p.resetDebugVisualizerCamera(cameraDistance=max(1.0, initial_length * 1.5), cameraYaw=45, cameraPitch=-20, cameraTargetPosition=[0, 0, initial_length / 2.0 * 0.8])
    print("[信息] 已设置相机视角。")

    # --- <<< 修改：初始化 ODE 对象时传入长度 >>> ---
    print("--- 初始化 ODE 对象 ---")
    try:
        # 直接将 initial_length 传递给 ODE 类的构造函数
        my_ode = ODE(initial_length_m=initial_length)
        # 如果需要，仍然设置 d (如果 ODE 的 __init__ 没有处理 d)
        my_ode.d = cable_distance
        # 验证 l0 是否正确设置
        print(f"ODE 已初始化 L0={my_ode.l0:.4f}m, d={my_ode.d:.4f}m") # 这里的 l0 应该是你传入的值
    except Exception as e:
         print(f"[错误] 初始化 ODE 对象时出错: {e}")
         print("[提示] 请确保您已正确修改 visualizer.py 中的 ODE.__init__ 方法。")
         p.disconnect(physicsClientId); sys.exit(1)
    # --- <<< 修改结束 >>> ---


    # --- 计算初始形态 (对应 dl=0) ---
    print("--- 计算初始形状 (dl=0) ---")
    # 初始 action 设为 [0, 0, 0]，因为长度已经在初始化时设定
    act0_segment = np.zeros(3)
    # 调用 reset (现在不会覆盖 l0 了)
    my_ode._reset_y0()
    # 更新 action (虽然是零，但遵循流程)
    my_ode.updateAction(act0_segment)
    # 求解
    sol0 = my_ode.odeStepFull() # 求解器现在应该使用正确的 l0 (例如 0.8) 进行积分
    if sol0 is None or sol0.shape[1] < 3:
        print("错误: 初始 ODE 求解失败或点数不足(<3)。"); p.disconnect(physicsClientId); sys.exit(1)
    print(f"初始形状计算完成。Sol0 shape: {sol0.shape}") # 确认初始解维度

    # --- 设置基座位置和姿态 (使用用户确认的竖直向下姿态) ---
    base_pos = np.array([0, 0, initial_length * 1.1]) # 根据最终长度调整基座高度
    base_ori_euler = np.array([-math.pi / 2.0, 0, 0]) # [Roll, Pitch, Yaw] (绕X轴旋转-90度)
    base_ori = p.getQuaternionFromEuler(base_ori_euler)
    print(f"[设置] 机器人基座世界坐标: {base_pos}")
    print(f"[设置] 机器人基座世界姿态 (Euler): {base_ori_euler}")

    radius = my_sphere_radius

    # --- 创建 PyBullet 形状 ---
    print("--- 创建 PyBullet 形状 ---")
    # ... (与上一版本相同，代码省略) ...
    try:
        shape = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
        visualShapeId = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=body_color)
        visualShapeId_tip_body = p.createVisualShape(p.GEOM_SPHERE, radius=radius+0.0025, rgbaColor=my_head_color)
        visualShapeId_tip_gripper = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.002, 0.01, 0.002], rgbaColor=head_color)
    except p.error as e: print(f"创建形状出错: {e}"); p.disconnect(physicsClientId); sys.exit(1)


    # --- 创建 PyBullet 物体 (基于计算出的初始形态 sol0 和基座位姿) ---
    print("--- 创建 PyBullet 物体 (显示初始形态) ---")
    # ... (与上一版本相同，代码省略) ...
    if sol0.shape[1] < 3: print("错误: 初始解点数不足3。"); p.disconnect(physicsClientId); sys.exit(1)
    idx0 = np.linspace(0, sol0.shape[1] - 1, my_number_of_sphere, dtype=int)
    positions0_local = [(sol0[0, i], sol0[2, i], sol0[1, i]) for i in idx0] # YZ交换
    my_robot_bodies = []
    try: # 创建身体
        for i, pos_local in enumerate(positions0_local):
            pos_world, ori_world = p.multiplyTransforms(base_pos, base_ori, pos_local, [0,0,0,1])
            my_robot_bodies.append(p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=visualShapeId, basePosition=pos_world, baseOrientation=ori_world))
    except p.error as e: print(f"创建身体时出错: {e}"); p.disconnect(physicsClientId); sys.exit(1)
    num_initial_bodies = len(my_robot_bodies); num_tip_bodies_expected = 3 # 创建末端
    if len(positions0_local) >= 3 and num_initial_bodies == my_number_of_sphere:
        try:
            ori_tip_local, _ = calculate_orientation(positions0_local[-3], positions0_local[-1])
            pos_tip_world, ori_tip_world = p.multiplyTransforms(base_pos, base_ori, positions0_local[-1], ori_tip_local)
            my_robot_bodies.append(p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=visualShapeId_tip_body, basePosition=pos_tip_world, baseOrientation=ori_tip_world))
            gripper_offset1_local = [0, 0.01, 0]; pos1, _ = p.multiplyTransforms(pos_tip_world, ori_tip_world, gripper_offset1_local, [0,0,0,1])
            my_robot_bodies.append(p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=visualShapeId_tip_gripper, basePosition=pos1, baseOrientation=ori_tip_world))
            gripper_offset2_local = [0, -0.01, 0]; pos2, _ = p.multiplyTransforms(pos_tip_world, ori_tip_world, gripper_offset2_local, [0,0,0,1])
            my_robot_bodies.append(p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=visualShapeId_tip_gripper, basePosition=pos2, baseOrientation=ori_tip_world))
            print(f"总共创建了 {len(my_robot_bodies)} 个物体。")
        except Exception as e: print(f"创建末端物体出错: {e}"); p.disconnect(physicsClientId); sys.exit(1)
    else: print("警告: 无法创建末端物体。")


    print("--- 初始化和创建完成 ---")
    print(f"已生成长度为 {initial_length:.3f} 米的静态机器人可视化。")
    print("您可以手动关闭 PyBullet 窗口来结束程序。")

    # --- 保持 PyBullet 窗口打开 ---
    try:
        while p.isConnected(physicsClientId):
            # 静态显示，不需要执行仿真步骤
            time.sleep(0.1) # 短暂休眠，避免 CPU 占用过高
    except KeyboardInterrupt: print("\n[信息] 检测到键盘中断 (Ctrl+C)...")
    except p.error: print("[信息] PyBullet 窗口可能已关闭。")
    finally:
        # 清理工作
        print("[信息] 断开 PyBullet 连接。")
        if p.isConnected(physicsClientId):
            try: p.disconnect(physicsClientId)
            except p.error: pass

    print("--- 程序结束 ---")