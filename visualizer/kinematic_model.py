import numpy as np
from scipy.integrate import solve_ivp

# 为了清晰，重命名类名
class ODE_PhysicsBased:
    # 移除旧的 axial_coupling_coefficient 参数
    def __init__(self, initial_length_m=0.1, cable_distance_m=7.5e-3, ode_step_ds=0.005):
        """
        初始化基于物理输入的PCC运动学ODE求解器。

        参数:
            initial_length_m (float): 机器人的参考长度 (米)。
            cable_distance_m (float): 拉线径向距离 (米)。
            ode_step_ds (float): ODE 积分步长 (用于评估点)。
        """
        # --- 存储基本参数 ---
        self.l0 = initial_length_m     # 参考长度
        self.d = cable_distance_m      # 拉线距离 (可能不再需要在类内部直接使用)
        self.ds = ode_step_ds          # 积分步长 (用于 t_eval)

        # --- 初始化当前段的状态变量 ---
        self.l = self.l0               # 当前长度, 默认为 l0
        self.ux = 0.0                  # 当前 X 曲率 (将由外部更新)
        self.uy = 0.0                  # 当前 Y 曲率 (将由外部更新)
        self.epsilon = 0.0             # 当前轴向应变 (将由外部更新)

        # --- 初始化 ODE 状态向量 y = [r; R] ---
        self.y0 = None                 # s=0 处的初始状态
        self.states = None             # 段末端 (s=l) 的状态
        self._reset_y0()               # 设置初始 y0

    def _reset_y0(self):
        """重置 s=0 处的初始状态向量 y0。"""
        r0 = np.array([0, 0, 0]).reshape(3, 1)
        R0 = np.eye(3, 3)
        R0_flat = np.reshape(R0, (9, 1))
        y0 = np.concatenate((r0, R0_flat), axis=0)
        self.y0 = np.squeeze(np.asarray(y0))
        # 重置 y0 时也重置最终状态
        self.states = np.copy(self.y0)

    def update_inputs(self, ux, uy, epsilon, delta_l=0.0):
        """
        更新由外部基于物理计算得到的运动学输入(曲率、应变、长度变化)。

        参数:
            ux (float): 绕段局部 x 轴的曲率。
            uy (float): 绕段局部 y 轴的曲率。
            epsilon (float): 沿段骨架的轴向应变。
            delta_l (float): 指令性的段长度变化量 (米), 加到 l0 上。
        """
        self.ux = ux
        self.uy = uy
        self.epsilon = epsilon
        self.l = self.l0 + delta_l # 更新当前段的实际长度

    # 移除旧的 updateAction 方法和 _calculate_axial_strain 方法

    def odeFunction(self, s, y):
        """
        定义 PCC ODE 系统: dr/ds 和 dR/ds。
        使用当前存储的 self.ux, self.uy, self.epsilon。
        """
        dydt = np.zeros(12)
        # 状态向量 y: [rx, ry, rz, R11, R12, R13, R21, R22, R23, R31, R32, R33]

        # 从状态 y 中提取旋转矩阵 R
        # r = y[0:3].reshape(3, 1) # 计算导数时实际上不需要位置 r
        R = np.array([y[3:6], y[6:9], y[9:12]]).reshape(3, 3)

        # 定义 e3 向量 (局部 z 轴)
        e3 = np.array([0, 0, 1]).reshape(3, 1)

        # 定义曲率向量 u = [ux, uy, 0] 的反对称矩阵 (假设无扭转)
        # 使用存储的 self.ux, self.uy
        u_hat = np.array([[0, 0, self.uy],
                          [0, 0, -self.ux],
                          [-self.uy, self.ux, 0]])

        # --- 计算导数 ---
        # dr/ds = (1 + epsilon) * R @ e3
        dr = (1.0 + self.epsilon) * (R @ e3) # 使用存储的 self.epsilon

        # dR/ds = R @ u_hat
        dR = R @ u_hat

        # 将导数存入 dydt 数组 (R 按行存储，dR.T 便于按列提取)
        dRR = dR.T # 转置以便按列访问
        dydt[0:3] = dr.flatten() # dr 是 (3,1), 展平为 (3,)
        dydt[3:6] = dRR[:, 0]
        dydt[6:9] = dRR[:, 1]
        dydt[9:12] = dRR[:, 2]

        # 返回一个扁平数组 (solve_ivp 需要)
        return dydt

    def odeStepFull(self):
        """
        求解整个段长度 'l' 上的 ODE。
        """
        # 定义积分区间 [0, l]
        integration_span = (0, self.l)

        # 定义需要求解器输出结果的点 (沿 s 轴)
        num_eval_points = max(2, int(self.l / self.ds)) # 保证至少有起点和终点
        t_eval = np.linspace(0, self.l, num_eval_points)

        # 求解初值问题
        try:
            sol = solve_ivp(
                fun=self.odeFunction,       # ODE 函数
                t_span=integration_span,    # 积分区间 [0, l]
                y0=self.y0,                 # s=0 处的初始状态
                method='RK45',              # 积分方法 (默认的通常不错)
                t_eval=t_eval               # 需要输出解的点
            )

            if sol.success:
                # 存储段末端 (s=l) 的状态
                self.states = np.squeeze(np.asarray(sol.y[:, -1]))
                # 返回完整的解矩阵 [状态维度 x 输出点数]
                return sol.y
            else:
                print(f"警告: ODE 求解失败 - {sol.message}")
                self.states = np.copy(self.y0) # 失败时重置状态
                return None # 表示失败
        except Exception as e:
            print(f"错误: ODE 求解过程中发生异常 - {e}")
            self.states = np.copy(self.y0) # 异常时重置状态
            return None


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
        

        self.ode = ODE()       
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
            self.ode.updateAction(np.array((0+num/100,num/1000,num/1000)))
        else:
            self.ode.updateAction(self.actions[int(num*self.speed),:])

        self.sol = self.ode.odeStepFull()
      
        self.robot._offsets3d = (self.sol[0,:], self.sol[1,:], self.sol[2,:])
        self.robotBackbone.set_data(self.sol[0,:], self.sol[1,:])
        self.robotBackbone.set_3d_properties(self.sol[2,:])
        
        self.endTipLine.set_data(self.endtips[0:int(num*self.speed),0], self.endtips[0:int(num*self.speed),1])
        self.endTipLine.set_3d_properties(self.endtips[0:int(num*self.speed),2])

        if self.obsEn:
            if self.obsPos1 is not None:
                self.obs1._offsets3d = (self.obsPos1[num*self.speed-1:num*self.speed,0], self.obsPos1[num*self.speed-1:num*self.speed,1],self.obsPos1[num*self.speed-1:num*self.speed,2])
            if self.obsPos2 is not None:
                self.obs2._offsets3d = (self.obsPos2[num*self.speed-1:num*self.speed,0], self.obsPos2[num*self.speed-1:num*self.speed,1],self.obsPos2[num*self.speed-1:num*self.speed,2])
            if self.obsPos3 is not None:
                self.obs3._offsets3d = (self.obsPos3[num*self.speed-1:num*self.speed,0], self.obsPos3[num*self.speed-1:num*self.speed,1],self.obsPos3[num*self.speed-1:num*self.speed,2])
            if self.obsPos4 is not None:
                self.obs4._offsets3d = (self.obsPos4[num*self.speed-1:num*self.speed,0], self.obsPos4[num*self.speed-1:num*self.speed,1],self.obsPos4[num*self.speed-1:num*self.speed,2])
            
        
        
        
