import numpy as np
import time
from matplotlib import pyplot as plt

import matplotlib.animation as animation
from   scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
import torch



class ODE():
    # def __init__(self) -> None:
    #     self.l  = 0
    #     self.uy = 0
    #     self.ux = 0
    #     self.dp = 0
    #     self.err = np.array((0,0,0))
    #     self.errp = np.array((0,0,0))

    #     self.simCableLength  = 0
    #     # initial length of robot
    #     self.l0 = 100e-3
    #     # cables offset
    #     self.d  = 7.5e-3
    #     # ode step time
    #     self.ds     = 0.005 #0.0005  
    #     # r0 = np.array([0,0,0]).reshape(3,1)  
    #     # R0 = np.eye(3,3)
    #     # R0 = np.reshape(R0,(9,1))
    #     # y0 = np.concatenate((r0, R0), axis=0)
    #     self._reset_y0()






    # def __init__(self, initial_length_m=0.1, cable_distance_m=7.5e-3, ode_step_ds=0.005): # <<< 可以将更多参数移到这里
    #     """
    #     初始化 ODE 求解器。
    #     Args:
    #         initial_length_m (float): 机器人的参考长度 (单位: 米)。
    #     """
    #     self.l = 0.0 # 当前积分长度？(通常由 l0 + action[0] 决定，但可以初始化为0)
    #     self.uy = 0.0 # 当前 Y 方向曲率
    #     self.ux = 0.0 # 当前 X 方向曲率
    #     self.dp = 0   # 未知用途，按原样初始化
    #     self.err = np.array([0.0, 0.0, 0.0]) # 未知用途，按原样初始化
    #     self.errp = np.array([0.0, 0.0, 0.0])# 未知用途，按原样初始化
    #     self.simCableLength = 0 # 未知用途，按原样初始化

    #     # --- 设置核心物理参数 ---
    #     self.l0 = initial_length_m    # <<< 使用传入的参数设置初始长度
    #     self.d = cable_distance_m     # <<< 使用传入的参数设置绳索距离
    #     self.ds = ode_step_ds         # <<< 使用传入的参数设置步长

    #     # --- 初始化 ODE 状态向量 ---
    #     # self.y0 在 _reset_y0 中设置，这里不需要显式设置
    #     self.states = None # 可以先设为 None

    #     # --- 初始化 action ---
    #     self.action = np.zeros(3)

    #     # --- 调用重置方法来设置 y0 (确保 _reset_y0 已修改！) ---
    #     self._reset_y0() # 注意：调用这个会设置 self.y0 和 self.states
    
    def __init__(self, initial_length_m=0.1, cable_distance_m=7.5e-3, ode_step_ds=0.005, 
                 axial_coupling_coefficient=0.0): # <<< 添加新参数 axial_coupling_coefficient
        """
        初始化 ODE 求解器。
        Args:
            initial_length_m (float): 机器人的参考长度 (单位: 米)。
            cable_distance_m (float): 缆绳到中心线的径向距离 (单位: 米)。
            ode_step_ds (float): ODE 积分步长。
            axial_coupling_coefficient (float): 弯曲导致轴向应变的耦合系数 (经验值, 通常为负数或零)。
        """
       
        self.l = 0.0 # 当前积分长度？(通常由 l0 + action[0] 决定，但可以初始化为0)
        self.uy = 0.0 # 当前 Y 方向曲率
        self.ux = 0.0 # 当前 X 方向曲率
        self.dp = 0   
        self.err = np.array([0.0, 0.0, 0.0]) 
        self.errp = np.array([0.0, 0.0, 0.0])
        self.simCableLength = 0 #

        # --- 设置核心物理参数 ---
        self.l0 = initial_length_m    # <<< 使用传入的参数设置初始长度
        self.d = cable_distance_m     # <<< 使用传入的参数设置绳索距离
        self.ds = ode_step_ds         # <<< 使用传入的参数设置步长
        self.k_strain = axial_coupling_coefficient # <<< 存储传入的耦合系数
       # --- 初始化应变 ---
        self.epsilon = 0.0 # <<< 初始化轴向应变
 
        # --- 初始化 ODE 状态向量 ---
        # self.y0 在 _reset_y0 中设置，这里不需要显式设置
        self.states = None # 可以先设为 None

        # --- 初始化 action ---
        self.action = np.zeros(3)

        # --- 调用重置方法来设置 y0 (确保 _reset_y0 已修改！) ---
        self._reset_y0() # 注意：调用这个会设置 self.y0 和 self.states    










    # def _reset_y0(self):
    #     r0 = np.array([0,0,0]).reshape(3,1)  
    #     R0 = np.eye(3,3)
    #     R0 = np.reshape(R0,(9,1))
    #     y0 = np.concatenate((r0, R0), axis=0)
    #     self.l0 = 100e-3       
    #     self.states = np.squeeze(np.asarray(y0))
    #     self.y0 = np.copy(self.states)
    

    # def _reset_y0(self, initialize=False): # <<< 添加可选参数
    #      """重置初始状态向量 y0。"""
    #      r0 = np.array([0,0,0]).reshape(3,1)
    #      R0 = np.eye(3,3)
    #      R0 = np.reshape(R0,(9,1))
    #      y0 = np.concatenate((r0, R0), axis=0)
    #      # --- !!! 删除或注释掉下面这行，防止覆盖 l0 !!! ---
    #      # self.l0 = 100e-3
    #      # ---------------------------------------------
    #      self.states = np.squeeze(np.asarray(y0))
    #      self.y0 = np.copy(self.states)
    #      # 可选：在初始化调用时，可以额外重置 action
    #      if initialize:
    #          self.action = np.zeros(3)
    def _reset_y0(self, initialize=False): # <<< 添加可选参数
         """重置初始状态向量 y0。"""
         r0 = np.array([0,0,0]).reshape(3,1)
         R0 = np.eye(3,3)
         R0 = np.reshape(R0,(9,1))
         y0 = np.concatenate((r0, R0), axis=0)
         # --- !!! 删除或注释掉下面这行，防止覆盖 l0 !!! ---
         # self.l0 = 100e-3
         # ---------------------------------------------
         self.states = np.squeeze(np.asarray(y0))
         self.y0 = np.copy(self.states)
         # 可选：在初始化调用时，可以额外重置 action
         if initialize:
             self.action = np.zeros(3)
                






    def _update_l0(self,l0):
        self.l0 = l0
        
    # def updateAction(self,action):
    #     self.l  = self.l0 + action[0]
    #     # self.l  = action0
        
    #     self.uy = (action[1]) /  (self.l * self.d)
    #     self.ux = (action[2]) / -(self.l * self.d)

    def updateAction(self,action):
        self.l  = self.l0 + action[0]
        # self.l  = action0
        
        # 计算曲率 ux, uy (添加除零保护)
        denominator = self.l * self.d
        if abs(denominator) > 1e-9:
            self.uy = (action[1]) / denominator
            self.ux = (action[2]) / -denominator # 保持负号
        else:
            self.uy = 0.0
            self.ux = 0.0
            # 可以加一个警告 print("Warning: Denominator is close to zero in updateAction.")

        # <<< 新增：计算并存储轴向应变 >>>
        self.epsilon = self._calculate_axial_strain() 
        # 现在 self.ux, self.uy, self.epsilon 都已更新完毕，可供 odeFunction 使用

    # def odeFunction(self,s,y):
    #     dydt  = np.zeros(12)
    #     # % 12 elements are r (3) and R (9), respectively
    #     e3    = np.array([0,0,1]).reshape(3,1)              
    #     u_hat = np.array([[0,0,self.uy], [0, 0, -self.ux],[-self.uy, self.ux, 0]])
    #     r     = y[0:3].reshape(3,1)
    #     R     = np.array( [y[3:6],y[6:9],y[9:12]]).reshape(3,3)
    #     # % odes
    #     dR  = R @ u_hat
    #     dr  = R @ e3
    #     dRR = dR.T
    #     dydt[0:3]  = dr.T
    #     dydt[3:6]  = dRR[:,0]
    #     dydt[6:9]  = dRR[:,1]
    #     dydt[9:12] = dRR[:,2]
    #     return dydt.T


    def odeFunction(self,s,y):
        dydt  = np.zeros(12)
        # % 12 elements are r (3) and R (9), respectively
        e3    = np.array([0,0,1]).reshape(3,1)              
        u_hat = np.array([[0,0,self.uy], [0, 0, -self.ux],[-self.uy, self.ux, 0]])
        r     = y[0:3].reshape(3,1)
        R     = np.array( [y[3:6],y[6:9],y[9:12]]).reshape(3,3)
        # % odes
        dR  = R @ u_hat
        dr  = (1 + self.epsilon) * (R @ e3)
        dRR = dR.T
        # 存储导数
        dydt[0:3]  = dr.T
        dydt[3:6]  = dRR[:,0]
        dydt[6:9]  = dRR[:,1]
        dydt[9:12] = dRR[:,2]
        return dydt.T

    def odeStepFull(self):        
        cableLength          = (0,self.l)
        
        t_eval               = np.linspace(0, self.l, int(self.l/self.ds))
        sol                  = solve_ivp(self.odeFunction,cableLength,self.y0,t_eval=t_eval)
        self.states          = np.squeeze(np.asarray(sol.y[:,-1]))
        return sol.y




    def _calculate_axial_strain(self):
        """
        根据弯曲曲率计算近似的轴向应变 (epsilon)。
        这是一个简化的经验模型，假设弯曲会导致压缩。
        """
        # self.k_strain: 负值表示弯曲引起压缩，0 表示无耦合
        # ux, uy: 当前的中心线曲率
        # l0: 参考长度，用于可能的缩放 (也可以不用)

        # 模型: 应变与总曲率的平方成正比
        curvature_squared = self.ux**2 + self.uy**2

        # 计算应变。乘以 self.l0 是为了让 k_strain 更接近无量纲？
        # 或者直接 k_strain * curvature_squared。需要根据实际情况调整。
        # 这里的尺度因子非常重要，需要根据实验数据仔细调整 k_strain！
        self.epsilon = self.k_strain * curvature_squared 

        # (可选) 限制应变范围，防止极端值
        # 例如，限制最大压缩为 30%，最大拉伸为 5%
        # max_compression = -0.3 
        # max_extension = 0.05 
        # self.epsilon = np.clip(self.epsilon, max_compression, max_extension)

        # 在这个简化模型中，我们直接返回计算出的应变
        # 注意：我们没有将计算出的 epsilon 存储回 self.epsilon，
        # 因为在这个模型里，ux, uy 不变，epsilon 也不变。
        # 在 odeFunction 中直接使用这里返回的值。
        # 如果 epsilon 会变化，则需要存储 self.epsilon。
        # 为了代码清晰，我们在 updateAction 中计算并存储 self.epsilon。
        # 所以这里直接返回计算值。

        # --- 更新：改为在 updateAction 中调用并存储 self.epsilon ---
        # --- 所以这个方法只负责计算逻辑 ---
        calculated_epsilon = self.k_strain * curvature_squared * self.l0 
        # 添加可选的 clip
        # calculated_epsilon = np.clip(calculated_epsilon, -0.3, 0.05) 
        return calculated_epsilon

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
            
        
        
        

if __name__ == "__main__":
  
    ode = ODE()
    sfVis = softRobotVisualizer()
    ode.updateAction(np.array([0,-0.01,0]))
    y = ode.odeStepFull()
    sfVis.visualize_3d_plot(data=y,color='b')
    ode.y0 = y[:,-1]
    ode.updateAction(np.array([0,0.01,0]))    
    y = ode.odeStepFull()
    sfVis.visualize_3d_plot(data=y,color='r')
    
    # Showing plot
    plt.show()
    
    
    sfVis = softRobotVisualizer()
    data = np.loadtxt("logData/data_corl22_20220606-144227",dtype=np.float32,delimiter=',',comments='#')
    len = data.shape[0]
    sfVis.actions = data[:,:3]

    ani = animation.FuncAnimation(sfVis.fig, sfVis.update_graph, len, interval=100, blit=False)
    timestr   = time.strftime("%Y%m%d-%H%M%S")
    gifName = "visualizer/saveGIFs/gif_visualizer_"+ timestr+".gif"
    print (f"saving gif: {gifName}")
    
    plt.show()
    writergif = animation.PillowWriter(fps=15) 
    ani.save(gifName, writer=writergif)
    print (f"gif file has been saved: {gifName}")
    # ani.save(gifName, writer='imagemagick', fps=30)

