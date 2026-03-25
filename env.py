import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt


class Config:
    """环境与训练参数配置类 (纯二维版本)"""
    # 环境基础参数
    env_size = 50.0  # 环境边长 50m x 50m
    grid_res = 0.5  # 栅格分辨率 0.5m
    grid_size = int(env_size / grid_res)  # 100 x 100 栅格
    max_steps = 200  # 最大步数

    # 无人机参数
    num_agents = 3
    uav_radius = 0.5  # 无人机碰撞半径
    det_radius = 5.0  # 探测半径
    safe_radius = 3.0  # 起终点安全区半径

    # 运动学边界
    v_max = 2.0  # 最大线速度 m/s
    w_max = np.pi / 4  # 最大角速度 rad/s

    # 障碍物参数
    num_obstacles = 5  # 障碍物数量
    obs_radius_range = [4.0, 4.0]  # 障碍物半径范围
    obs_min_gap = 5.0  # 障碍物边缘之间的最小通行距离

    # 感知参数
    num_lidar_rays = 16  # 激光雷达射线数
    lidar_range = 10.0  # 雷达最大量程
    local_map_size = 10  # 局部覆盖地图维度

    # 奖励函数系数
    c_step = 0.005  # 步数惩罚
    c_app = 0.1  # 靠近终点奖励
    c_dir = 0.02  # 方向对齐奖励
    c_smooth = 0.02  # 动作平滑惩罚
    c_cov = 0.04  # 有效覆盖奖励 (每覆盖一个新栅格)
    c_overlap = 0.0002  # 覆盖重叠惩罚

    # 终止态绝对奖励
    r_col = -2.0  # 碰撞极大惩罚
    r_arr = 2.0  # 到达极大奖励


class MultiUAVCoverageEnv(gym.Env):
    """多无人机协同覆盖与路径规划环境 (2D)"""

    def __init__(self, cfg=Config()):
        super(MultiUAVCoverageEnv, self).__init__()
        self.cfg = cfg
        self.n = cfg.num_agents

        # 动作空间: [v, w] 连续动作
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n, 2), dtype=np.float32)

        # 状态空间维度计算: 自身(5) + 目标(2) + 雷达(16) + 队友(2*4) + 局部地图(100) = 131
        obs_dim = 5 + 2 + cfg.num_lidar_rays + (self.n - 1) * 4 + cfg.local_map_size ** 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.n, obs_dim), dtype=np.float32)

        # 环境内部状态变量初始化 (纯二维)
        self.agents_pos = np.zeros((self.n, 2))  # x, y
        self.agents_theta = np.zeros(self.n)
        self.agents_vel = np.zeros((self.n, 2))  # v, w
        self.target_pos = np.zeros(2)
        self.start_pos = np.zeros(2)

        # 障碍物列表与全局覆盖地图
        self.obstacles = []
        self.global_map = np.zeros((self.cfg.grid_size, self.cfg.grid_size), dtype=np.int8)
        self.steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.global_map.fill(0)

        # 1. 生成对角线起终点
        pad = self.cfg.safe_radius + 2.0
        if np.random.rand() > 0.5:
            self.start_pos = np.array([pad, pad])
            self.target_pos = np.array([self.cfg.env_size - pad, self.cfg.env_size - pad])
        else:
            self.start_pos = np.array([pad, self.cfg.env_size - pad])
            self.target_pos = np.array([self.cfg.env_size - pad, pad])

        # 2. 初始化无人机位置 (二维平面散开)
        for i in range(self.n):
            angle = i * (2 * np.pi / self.n)
            r = 1.0
            self.agents_pos[i, 0] = self.start_pos[0] + r * np.cos(angle)
            self.agents_pos[i, 1] = self.start_pos[1] + r * np.sin(angle)
            self.agents_theta[i] = np.arctan2(self.target_pos[1] - self.agents_pos[i, 1],
                                              self.target_pos[0] - self.agents_pos[i, 0])
        self.agents_vel.fill(0.0)

        # 3. 静态圆形障碍物生成逻辑 (均匀分布 + 最小间距 + 动态半径)
        self.obstacles = []
        max_retries = 100

        for _ in range(self.cfg.num_obstacles):
            for _ in range(max_retries):
                orad = np.random.uniform(self.cfg.obs_radius_range[0], self.cfg.obs_radius_range[1])
                ox = np.random.uniform(orad, self.cfg.env_size - orad)
                oy = np.random.uniform(orad, self.cfg.env_size - orad)

                valid_position = True

                dist_to_start = np.hypot(ox - self.start_pos[0], oy - self.start_pos[1])
                dist_to_target = np.hypot(ox - self.target_pos[0], oy - self.target_pos[1])

                min_req_dist = self.cfg.safe_radius + orad + self.cfg.obs_min_gap
                if dist_to_start < min_req_dist or dist_to_target < min_req_dist:
                    valid_position = False
                    continue

                for ex, ey, erad in self.obstacles:
                    dist_to_obs = np.hypot(ox - ex, oy - ey)
                    if dist_to_obs < (orad + erad + self.cfg.obs_min_gap):
                        valid_position = False
                        break

                if valid_position:
                    self.obstacles.append((ox, oy, orad))
                    grid_x = int(ox / self.cfg.grid_res)
                    grid_y = int(oy / self.cfg.grid_res)
                    grid_rad = int(orad / self.cfg.grid_res)
                    for gx in range(max(0, grid_x - grid_rad), min(self.cfg.grid_size, grid_x + grid_rad)):
                        for gy in range(max(0, grid_y - grid_rad), min(self.cfg.grid_size, grid_y + grid_rad)):
                            if (gx - grid_x) ** 2 + (gy - grid_y) ** 2 <= grid_rad ** 2:
                                self.global_map[gx, gy] = -1
                    break

        self._update_coverage()
        return self._get_obs(), {}

    def step(self, action):
        self.steps += 1
        rewards = np.zeros(self.n)
        dones = np.zeros(self.n, dtype=bool)
        info = {f'agent_{i}': {} for i in range(self.n)}

        prev_pos = self.agents_pos.copy()
        prev_vel = self.agents_vel.copy()

        # 1. 运动学更新 (二维)
        for i in range(self.n):
            a_v, a_w = action[i]

            v = ((a_v + 1.0) / 2.0) * self.cfg.v_max
            w = a_w * self.cfg.w_max

            self.agents_vel[i, 0] = v
            self.agents_vel[i, 1] = w
            self.agents_theta[i] += w
            self.agents_pos[i, 0] += v * np.cos(self.agents_theta[i])
            self.agents_pos[i, 1] += v * np.sin(self.agents_theta[i])

        # 2. 更新覆盖率并获取 N_new, N_old
        cov_stats = self._update_coverage()

        # 3. 计算奖励与终止条件
        for i in range(self.n):
            r_components = {}

            # (1) 步数惩罚
            r_step = -self.cfg.c_step
            r_components['step'] = r_step

            # (2) 靠近终点奖励
            d_prev = np.hypot(self.target_pos[0] - prev_pos[i, 0], self.target_pos[1] - prev_pos[i, 1])
            d_curr = np.hypot(self.target_pos[0] - self.agents_pos[i, 0], self.target_pos[1] - self.agents_pos[i, 1])
            r_app = self.cfg.c_app * (d_prev - d_curr)
            r_components['approach'] = r_app

            # (3) 方向奖励
            angle_to_target = np.arctan2(self.target_pos[1] - self.agents_pos[i, 1],
                                         self.target_pos[0] - self.agents_pos[i, 0])
            angle_diff = abs(angle_to_target - self.agents_theta[i])
            angle_diff = min(angle_diff, 2 * np.pi - angle_diff)
            r_dir = self.cfg.c_dir * (1.0 - angle_diff / np.pi)
            r_components['dir'] = r_dir

            # (4) 动作平滑惩罚
            r_smooth = -self.cfg.c_smooth * (
                        (self.agents_vel[i, 0] - prev_vel[i, 0]) ** 2 + (self.agents_vel[i, 1] - prev_vel[i, 1]) ** 2)
            r_components['smooth'] = r_smooth

            # (5) 协同覆盖奖励
            n_new, n_old = cov_stats[i]
            r_cov = self.cfg.c_cov * n_new
            r_overlap = -self.cfg.c_overlap * n_old
            r_components['cov'] = r_cov
            r_components['overlap'] = r_overlap

            # (6) 碰撞与到达终点检测
            r_terminal = 0.0
            col_flag = False

            x, y = self.agents_pos[i, 0], self.agents_pos[i, 1]
            if x < 0 or x > self.cfg.env_size or y < 0 or y > self.cfg.env_size:
                col_flag = True

            for ox, oy, orad in self.obstacles:
                if np.hypot(x - ox, y - oy) < (self.cfg.uav_radius + orad):
                    col_flag = True
                    break

            if col_flag:
                r_terminal = self.cfg.r_col
                dones[i] = True
            elif d_curr <= self.cfg.safe_radius:
                # 计算当前总覆盖率 (非障碍物区域)
                total_valid_grids = np.sum(self.global_map >= 0)
                covered_grids = np.sum(self.global_map == 1)
                coverage_rate = covered_grids / total_valid_grids
                # 动态终点奖励：覆盖率越高，到达奖励越高 (2.0 是基础，最高可达 4.0)
                r_terminal = self.cfg.r_arr + (coverage_rate * 2.0)
                dones[i] = True

            r_components['terminal'] = r_terminal

            # 单步总奖励加总与截断
            total_r = sum(r_components.values())
            rewards[i] = np.clip(total_r, -2.0, 2.0)

            info[f'agent_{i}'] = r_components
            info[f'agent_{i}']['clipped_total'] = rewards[i]

        truncated = self.steps >= self.cfg.max_steps
        done_all = np.all(dones)

        return self._get_obs(), rewards, done_all, truncated, info

    def _update_coverage(self):
        """更新全局栅格地图并返回每架无人机的新/旧覆盖网格数"""
        stats = []
        for i in range(self.n):
            n_new, n_old = 0, 0
            x, y = self.agents_pos[i, 0], self.agents_pos[i, 1]
            gx = int(x / self.cfg.grid_res)
            gy = int(y / self.cfg.grid_res)
            grad = int(self.cfg.det_radius / self.cfg.grid_res)

            for dx in range(-grad, grad + 1):
                for dy in range(-grad, grad + 1):
                    if dx ** 2 + dy ** 2 <= grad ** 2:
                        nx, ny = gx + dx, gy + dy
                        if 0 <= nx < self.cfg.grid_size and 0 <= ny < self.cfg.grid_size:
                            cell_val = self.global_map[nx, ny]
                            if cell_val == 0:
                                n_new += 1
                                self.global_map[nx, ny] = 1  # 标记为已覆盖
                            elif cell_val == 1:
                                n_old += 1
            stats.append((n_new, n_old))
        return stats

    def _get_obs(self):
        """构造归一化的局部观测状态 (纯二维)"""
        obs = []
        for i in range(self.n):
            # 1. 自身状态归一化
            x, y = self.agents_pos[i, 0], self.agents_pos[i, 1]
            v, w = self.agents_vel[i, 0], self.agents_vel[i, 1]
            theta = self.agents_theta[i]
            s_self = [
                x / self.cfg.env_size, y / self.cfg.env_size,
                theta / np.pi, v / self.cfg.v_max, w / self.cfg.w_max
            ]

            # 2. 目标信息归一化
            d_target = np.hypot(self.target_pos[0] - x, self.target_pos[1] - y)
            angle_to_target = np.arctan2(self.target_pos[1] - y, self.target_pos[0] - x)
            angle_diff = angle_to_target - theta
            angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
            s_target = [d_target / (self.cfg.env_size * 1.414), angle_diff / np.pi]

            # 3. 极简版局部地图 (以自身为中心截取 local_map_size)
            half_s = self.cfg.local_map_size // 2
            gx, gy = int(x / self.cfg.grid_res), int(y / self.cfg.grid_res)
            local_map = np.zeros((self.cfg.local_map_size, self.cfg.local_map_size))
            for dx in range(-half_s, half_s):
                for dy in range(-half_s, half_s):
                    nx, ny = gx + dx, gy + dy
                    if 0 <= nx < self.cfg.grid_size and 0 <= ny < self.cfg.grid_size:
                        local_map[dx + half_s, dy + half_s] = self.global_map[nx, ny]
                    else:
                        local_map[dx + half_s, dy + half_s] = -1  # 越界视为障碍物
            s_local = local_map.flatten().tolist()

            # 4. 雷达射线
            s_lidar = np.ones(self.cfg.num_lidar_rays).tolist()

            # 5. 队友共享信息
            s_shared = []
            for j in range(self.n):
                if i != j:
                    dx_r = (self.agents_pos[j, 0] - x) / self.cfg.env_size
                    dy_r = (self.agents_pos[j, 1] - y) / self.cfg.env_size
                    s_shared.extend([
                        dx_r, dy_r,
                        self.agents_vel[j, 0] / self.cfg.v_max,
                        self.agents_vel[j, 1] / self.cfg.w_max
                    ])

            # 拼接
            agent_obs = np.concatenate([s_self, s_target, s_lidar, s_shared, s_local])
            obs.append(agent_obs)

        return np.array(obs, dtype=np.float32)


class Plot:
    """通用绘图类接口"""

    @staticmethod
    def plot_learning_curve(episode_rewards, title="Multi-UAV Coverage Learning Curve"):
        plt.figure(figsize=(10, 6))

        window_size = 50
        if len(episode_rewards) >= window_size:
            moving_avg = np.convolve(episode_rewards, np.ones(window_size) / window_size, mode='valid')
            plt.plot(np.arange(len(moving_avg)) + window_size - 1, moving_avg, color='red', linewidth=2,
                     label='Moving Average (50 eps)')

        plt.plot(episode_rewards, alpha=0.3, color='blue', label='Episode Reward')
        plt.axhline(y=400, color='g', linestyle='--', label='Max Possible Return')
        plt.axhline(y=-400, color='r', linestyle='--', label='Min Possible Return')

        plt.xlabel('Episodes')
        plt.ylabel('Total Reward')
        plt.title(title)
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.show()