# visualizer.py (修正版 - 确保 odeStepFull 返回对象)
import numpy as np
import time
from matplotlib import pyplot as plt

import matplotlib.animation as animation
from scipy.integrate import solve_ivp # 确认导入
from mpl_toolkits.mplot3d import Axes3D
# import torch # 如果没用到 torch 可以注释掉

class ODE():
    # 使用你文件中的 __init__ 版本
    def __init__(self, initial_length_m=0.12, cable_distance_m=0.0004, ode_step_ds=0.0005,
                 axial_coupling_coefficient=0.0):
        """
        初始化 ODE 求解器。
        """
        self.l = 0.0 # 当前积分长度
        self.uy = 0.0 # 当前 Y 方向曲率
        self.ux = 0.0 # 当前 X 方向曲率
        # --- 其他属性 ---
        self.l0 = initial_length_m
        self.d = cable_distance_m
        self.ds = ode_step_ds # 这个 ds 主要用于 t_eval 点数估计
        self.k_strain = axial_coupling_coefficient
        self.epsilon = 0.0 # 初始化轴向应变
        self.states = None
        self.y0 = None # 先设为 None
        self._reset_y0(initialize=True) # 初始化时设置默认的 y0

    def _reset_y0(self, initialize=False):
         """重置初始状态向量 y0 为基座状态 [0,0,0, 1,0,0, 0,1,0, 0,0,1]。"""
         r0 = np.array([0,0,0])
         R0 = np.eye(3)
         # R 按列展平存储: [R00, R10, R20, R01, R11, R21, R02, R12, R22]
         y0_flat = np.concatenate((r0, R0.flatten(order='F')), axis=0) # 'F' for column-major flatten
         if len(y0_flat) != 12:
             print("[错误] _reset_y0: 状态向量长度不为 12！")
             y0_flat = np.array([0,0,0, 1,0,0, 0,1,0, 0,0,1])

         self.y0 = np.copy(y0_flat) # 设置默认的 y0
         self.states = np.copy(self.y0)

         if initialize:
             self.ux = 0.0
             self.uy = 0.0
             self.epsilon = 0.0
             self.l = self.l0

    def set_initial_state(self, y0_new):
        """显式设置下一次 ODE 求解的初始状态向量。"""
        if y0_new is not None and len(y0_new) == 12:
            self.y0 = np.copy(np.squeeze(y0_new))
        else:
            print("[警告] ODE.set_initial_state: 提供的初始状态无效，将使用之前的 y0。")

    def set_kinematic_state(self, length_change_signal, ux_input, uy_input):
        """直接设置ODE对象当前的运动学状态（长度变化、曲率）。"""
        self.l = self.l0 + length_change_signal
        self.ux = ux_input
        self.uy = uy_input
        self.epsilon = self._calculate_axial_strain()

    def odeFunction(self,s,y):
        """定义 Cosserat rod 的 ODE 方程 dy/ds = f(s,y)。"""
        dydt  = np.zeros_like(y)
        e3    = np.array([0,0,1]) # 作为 1D array
        u_hat = np.array([[0, 0, self.uy], [0, 0, -self.ux],[-self.uy, self.ux, 0]])

        try:
            # 从状态向量 y (1D array) 中提取 R (按列优先)
            R = y[3:].reshape((3, 3), order='F')
        except ValueError:
             print(f"[错误] odeFunction: 无法将 y[3:] (shape: {y[3:].shape}) 重塑为 3x3 矩阵。")
             return np.zeros_like(y)

        # 计算导数
        dR = R @ u_hat # R' = R*[u]x
        dr = (1 + self.epsilon) * (R @ e3) # r' = (1+eps)*R*e3

        # 存储导数 (dr 和 dR 按列展平)
        dydt[0:3]  = dr
        dydt[3:12] = dR.flatten(order='F')

        return dydt

    # ==============================================================
    # === 关键：确保 odeStepFull 返回 solve_ivp 的完整对象 ===
    # ==============================================================
    def odeStepFull(self):
        """使用 solve_ivp 求解 ODE 从 s=0 到 s=self.l。"""
        cableLength = (0, self.l)

        if self.l <= 1e-9:
            print(f"[警告] odeStepFull: 目标长度 l ({self.l:.4f}) 无效。")
            sol = type('obj', (object,), {'status': -10, 'message': 'Invalid length l', 'y': None})()
            return sol # 返回模拟失败的对象

        num_points = max(20, int(self.l / self.ds)) if self.ds > 1e-9 else 50 # 避免 ds=0
        t_eval = np.linspace(0, self.l, num_points)

        if self.y0 is None or len(self.y0) != 12 or np.isnan(self.y0).any() or np.isinf(self.y0).any():
            print(f"[错误] odeStepFull: 无效的初始状态 y0: {self.y0}。")
            sol = type('obj', (object,), {'status': -11, 'message': 'Invalid y0', 'y': None})()
            return sol # 返回模拟失败的对象

        try:
            sol = solve_ivp(
                fun=self.odeFunction,
                t_span=cableLength,
                y0=self.y0,
                method='RK45',
                t_eval=t_eval,
                rtol=1e-4,
                atol=1e-7
            )
            # 检查求解器状态
            if sol.status != 0:
                print(f"[警告] ODE solver failed. Status: {sol.status}, Message: {sol.message}")
            elif sol.y is None or sol.y.shape[1] == 0:
                 print(f"[警告] ODE solver status 0 but returned empty solution.")
                 sol.status = -12 # 自定义状态码

            # 检查解中是否有 NaN/Inf
            if sol.y is not None and (np.isnan(sol.y).any() or np.isinf(sol.y).any()):
                 print(f"[警告] ODE solution contains NaN or Inf (status={sol.status}).")
                 sol.status = -13 # 自定义状态码

            # *** 返回完整的 solve_ivp 结果对象 sol ***
            return sol

        except Exception as e:
            print(f"[错误] ODE solver encountered an exception: {e}")
            print(f"--> Current y0[:3]: {self.y0[:3] if self.y0 is not None else 'None'}")
            print(f"--> Current l={self.l:.4f}, ux={self.ux:.4f}, uy={self.uy:.4f}, eps={self.epsilon:.4f}")
            sol = type('obj', (object,), {'status': -99, 'message': str(e), 'y': None})()
            return sol # 返回模拟失败的对象

    def _calculate_axial_strain(self):
        """根据曲率计算轴向应变 epsilon。"""
        curvature_squared = self.ux**2 + self.uy**2
        calculated_epsilon = self.k_strain * curvature_squared * self.l0
        # calculated_epsilon = np.clip(calculated_epsilon, -0.8, 0.5) # 可选：限制应变范围
        return calculated_epsilon

# --- softRobotVisualizer 类 (保持不变) ---
# ... (如果不需要可以删除这个类的代码) ...
class softRobotVisualizer():
     def __init__(self,obsEn = False,title=None,ax_lim=None) -> None:
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        if title == None:
            self.title = self.ax.set_title('Visualizer-1.02')
        else:
            self.title = self.ax.set_title(title)

        self.xlabel = self.ax.set_xlabel("x (m)")
        self.ylabel = self.ax.set_ylabel("y (m)")
        self.zlabel = self.ax.set_zlabel("z (m)")
        if ax_lim is None:
            self.ax.set_xlim([-0.08,0.08])
            self.ax.set_ylim([-0.08,0.08])
            self.ax.set_zlim([-0.0,0.15])
        else:
            self.ax.set_xlim(ax_lim[0])
            self.ax.set_ylim(ax_lim[1])
            self.ax.set_zlim(ax_lim[2])
        self.speed = 1

        self.actions = None
        self.endtips = None
        self.obsEn = obsEn
        self._ax = None


        self.ode = ODE() # Visualizer 内部有自己的 ODE 实例，与 main 不同
        self.robot = self.ax.scatter([], [], [],marker='o',lw=6)
        self.robotBackbone, = self.ax.plot([], [], [],'r',lw=4)
        self.endTipLine, = self.ax.plot([], [], [],'r',lw=2)

        if self.obsEn:
            self.obsPos1  = None
            self.obsPos2  = None
            self.obsPos3  = None
            self.obsPos4  = None

            self.obs1 =  self.ax.scatter([], [], [],marker='o',lw=7)
            self.obs2 =  self.ax.scatter([], [], [],marker='o',lw=7)
            self.obs3 =  self.ax.scatter([], [], [],marker='o',lw=7)
            self.obs4 =  self.ax.scatter([], [], [],marker='o',lw=7)


     def visualize_3d_plot(self,data,color='b'):
        if self._ax is None:
            # Creating figure
            fig = plt.figure()
            # Adding 3D subplot
            self._ax = fig.add_subplot(111, projection='3d')


         # Creating plot
        self._ax.scatter(data[0,:], data[1,:], data[2,:],color)


     def update_graph(self,num):

        if self.actions is None:
            # This part seems for standalone animation, maybe not relevant for main sim
            self.ode.set_kinematic_state(num/100, num/1000, num/1000) # Use new method if applicable
            # self.ode.updateAction(np.array((0+num/100,num/1000,num/1000))) # Old method
        else:
            # Assuming actions map directly to kinematic state for the visualizer's ODE
            # This needs adjustment based on how 'actions' are defined
            # Example: If actions are [length_change, ux, uy]
            # self.ode.set_kinematic_state(self.actions[int(num*self.speed),0],
            #                              self.actions[int(num*self.speed),1],
            #                              self.actions[int(num*self.speed),2])
            # If actions are [dl1, dl2, dl3]
            dl_vis = self.actions[int(num*self.speed),:]
            avg_dl_vis = np.mean(dl_vis)
            ux_vis, uy_vis = calculate_curvatures_from_dl_v3(dl_vis, self.ode.d, self.ode.l0, 1.0) # Assuming v3
            self.ode.set_kinematic_state(avg_dl_vis, ux_vis, uy_vis)


        sol_obj_vis = self.ode.odeStepFull() # Returns object

        if sol_obj_vis is not None and sol_obj_vis.status == 0 and sol_obj_vis.y.shape[1] > 0:
            sol_vis = sol_obj_vis.y
            self.robot._offsets3d = (sol_vis[0,:], sol_vis[1,:], sol_vis[2,:])
            self.robotBackbone.set_data(sol_vis[0,:], sol_vis[1,:])
            self.robotBackbone.set_3d_properties(sol_vis[2,:])

            if self.endtips is not None: # Check if endtips data exists
                self.endTipLine.set_data(self.endtips[0:int(num*self.speed),0], self.endtips[0:int(num*self.speed),1])
                self.endTipLine.set_3d_properties(self.endtips[0:int(num*self.speed),2])
        else:
             # Handle solver failure in visualization if needed
             pass

        # Obstacle update logic (keep as is)
        if self.obsEn:
            if self.obsPos1 is not None:
                self.obs1._offsets3d = (self.obsPos1[num*self.speed-1:num*self.speed,0], self.obsPos1[num*self.speed-1:num*self.speed,1],self.obsPos1[num*self.speed-1:num*self.speed,2])
            # ... (rest of obs updates) ...


if __name__ == "__main__":
    # --- This part is for testing visualizer.py standalone ---
    # --- It does not affect how main.py uses the ODE class ---
    print("Running visualizer.py standalone test...")
    try:
        ode_test = ODE()
        sfVis_test = softRobotVisualizer()

        # Test case 1
        print("Test Case 1: Setting state...")
        ode_test.set_kinematic_state(length_change_signal=-0.01, ux_input=0, uy_input=5) # Example state
        sol1_obj = ode_test.odeStepFull()
        if sol1_obj is not None and sol1_obj.status == 0:
            print("Test Case 1: Plotting...")
            sfVis_test.visualize_3d_plot(data=sol1_obj.y, color='b')
            # Use the last state as the initial state for the next step
            ode_test.set_initial_state(sol1_obj.y[:, -1])
        else:
            print("Test Case 1: Failed.")

        # Test case 2 (incremental)
        print("Test Case 2: Setting state...")
        ode_test.set_kinematic_state(length_change_signal=0.01, ux_input=5, uy_input=0) # Example state
        sol2_obj = ode_test.odeStepFull()
        if sol2_obj is not None and sol2_obj.status == 0:
            print("Test Case 2: Plotting...")
            sfVis_test.visualize_3d_plot(data=sol2_obj.y, color='r')
        else:
             print("Test Case 2: Failed.")

        # Showing plot if any data was plotted
        if sfVis_test._ax is not None:
            print("Displaying plot...")
            plt.show()
        else:
            print("No data plotted.")

    except Exception as e_test:
        print(f"Error during visualizer standalone test: {e_test}")

    print("Visualizer standalone test finished.")
    # --- End of standalone test ---