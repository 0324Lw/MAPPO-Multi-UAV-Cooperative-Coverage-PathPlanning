import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Circle, Wedge

# 导入我们写好的环境和配置
from env import MultiUAVCoverageEnv, Config


# ==========================================
# 1. 提取 Actor 网络 (为了独立运行，在此重定义)
# ==========================================
def orthogonal_init(layer, gain=1.0):
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain=gain)
        nn.init.constant_(layer.bias, 0)


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))

        self.net.apply(orthogonal_init)
        orthogonal_init(self.mean_layer, gain=0.01)

    def forward(self, obs):
        x = self.net(obs)
        mean = torch.tanh(self.mean_layer(x))
        return mean  # 测试时只取确定的均值动作


# ==========================================
# 2. 核心渲染与生成 GIF 逻辑
# ==========================================
def run_and_save_gif(env, actor, episode_idx, gif_dir="gifs"):
    print(f"\n--- 开始录制第 {episode_idx} 个回合 ---")
    os.makedirs(gif_dir, exist_ok=True)

    # 1. 运行环境，收集完整轨迹
    obs, _ = env.reset()
    history = []
    done = False
    truncated = False

    # 记录初始状态
    history.append({
        'pos': env.agents_pos.copy(),
        'theta': env.agents_theta.copy(),
        'map': env.global_map.copy()
    })

    step_count = 0
    total_reward = 0

    while not (done or truncated):
        # 将观测转为 Tensor
        obs_tensor = torch.tensor(obs, dtype=torch.float32)

        # 使用 Actor 网络进行确定性推理 (Deterministic)
        with torch.no_grad():
            action = actor(obs_tensor).numpy()

        obs, rewards, done, truncated, info = env.step(action)

        total_reward += np.sum(rewards) / env.n
        step_count += 1

        # 记录每步的状态
        history.append({
            'pos': env.agents_pos.copy(),
            'theta': env.agents_theta.copy(),
            'map': env.global_map.copy()
        })

    print(f"回合结束。总步数: {step_count}, 平均奖励: {total_reward:.2f}")
    print("正在渲染并生成 GIF，请稍候...")

    # 2. 设置 Matplotlib 画布
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, env.cfg.env_size)
    ax.set_ylim(0, env.cfg.env_size)
    ax.set_aspect('equal')
    ax.set_title(f"Multi-UAV Coverage - Episode {episode_idx}", fontsize=14)

    # 定义覆盖地图的颜色映射 (-1: 障碍物(灰), 0: 未覆盖(白), 1: 已覆盖(浅绿))
    cmap = ListedColormap(['#808080', '#FFFFFF', '#C8E6C9'])
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)

    # 绘制初始底层覆盖地图
    # 注意：我们的 global_map 是 x 对应行，y 对应列，imshow 默认 origin='upper'，需要转置并设为 lower
    img_map = ax.imshow(history[0]['map'].T, origin='lower', extent=[0, env.cfg.env_size, 0, env.cfg.env_size],
                        cmap=cmap, norm=norm, alpha=0.6)

    # 绘制起终点安全区
    ax.add_patch(Circle(env.start_pos, env.cfg.safe_radius, color='blue', alpha=0.15, label='Start Zone'))
    ax.add_patch(Circle(env.target_pos, env.cfg.safe_radius, color='red', alpha=0.15, label='Target Zone'))
    ax.plot(env.start_pos[0], env.start_pos[1], 'bo')
    ax.plot(env.target_pos[0], env.target_pos[1], 'r*', markersize=12)

    # 初始化无人机实体和探测范围
    uav_bodies = []
    uav_detects = []
    uav_headings = []
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 为3架无人机分配不同颜色

    for i in range(env.n):
        # 探测范围 (半透明大圆)
        det_circle = Circle(history[0]['pos'][i], env.cfg.det_radius, color=colors[i], alpha=0.2)
        ax.add_patch(det_circle)
        uav_detects.append(det_circle)

        # 无人机实体 (小圆)
        body = Circle(history[0]['pos'][i], env.cfg.uav_radius, color=colors[i])
        ax.add_patch(body)
        uav_bodies.append(body)

        # 航向指示器 (扇形/三角形)
        heading = Wedge(history[0]['pos'][i], env.cfg.uav_radius * 2,
                        np.degrees(history[0]['theta'][i]) - 15,
                        np.degrees(history[0]['theta'][i]) + 15,
                        color='black', alpha=0.8)
        ax.add_patch(heading)
        uav_headings.append(heading)

    # 3. 动画更新函数
    def update(frame):
        state = history[frame]

        # 更新底层覆盖地图
        img_map.set_data(state['map'].T)

        # 更新每架无人机的位置、探测圈和航向
        for i in range(env.n):
            pos = state['pos'][i]
            theta = state['theta'][i]

            uav_bodies[i].center = pos
            uav_detects[i].center = pos

            uav_headings[i].set_center(pos)
            uav_headings[i].set_theta1(np.degrees(theta) - 15)
            uav_headings[i].set_theta2(np.degrees(theta) + 15)

        return [img_map] + uav_bodies + uav_detects + uav_headings

    # 创建动画
    ani = animation.FuncAnimation(fig, update, frames=len(history), interval=200, blit=False)

    # 保存为 GIF (fps=8 保证演示速度足够慢，让你能看清覆盖过程)
    gif_path = os.path.join(gif_dir, f"coverage_test_{episode_idx}.gif")
    ani.save(gif_path, writer='pillow', fps=8)
    print(f"✅ GIF 已保存至: {gif_path}")

    plt.close(fig)  # 关闭画布防止内存泄漏


# ==========================================
# 3. 主函数
# ==========================================
if __name__ == "__main__":
    env = MultiUAVCoverageEnv(Config())
    obs_dim = env.observation_space.shape[1]
    action_dim = env.action_space.shape[1]

    # 实例化 Actor 并加载权重
    actor = Actor(obs_dim, action_dim, hidden_dim=256)

    # 【请在这里修改为你实际保存的模型路径！】
    model_path = "./models/actor_step_307200.pth"

    if os.path.exists(model_path):
        print(f"成功加载模型权重: {model_path}")
        actor.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    else:
        print(f"⚠️ 警告: 未找到模型文件 {model_path}。")
        print("将使用【随机初始化的权重】进行演示，无人机可能会像无头苍蝇一样乱飞。")

    # 运行并生成 5 张 GIF
    for i in range(1, 6):
        run_and_save_gif(env, actor, episode_idx=i, gif_dir="test_gifs")

    print("\n🎉 全部 5 张演示 GIF 均已生成完毕！快去 test_gifs 文件夹看看吧！")