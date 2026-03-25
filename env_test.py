import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from env import MultiUAVCoverageEnv, Config  # 假设之前的代码保存为 env.py


def test_spaces_and_step(env):
    """测试状态空间、动作空间的维度和数值输出是否正常，以及 step() 是否无 Bug"""
    print("=" * 50)
    print("1. 空间维度与环境交互测试")
    print("=" * 50)

    obs, info = env.reset()
    print(f"观测空间维度 (Observation Space): {env.observation_space.shape}")
    print(f"动作空间维度 (Action Space): {env.action_space.shape}")

    # 检查状态是否包含 NaN 或 Inf
    assert not np.isnan(obs).any(), "观测状态中存在 NaN!"
    assert not np.isinf(obs).any(), "观测状态中存在 Inf!"

    # 执行一步随机动作测试 step 功能
    action = env.action_space.sample()
    next_obs, rewards, done, truncated, info = env.step(action)

    print("执行 1 步随机动作测试成功！")
    print(f"单步奖励向量: {rewards}")
    print(f"Done: {done}, Truncated: {truncated}\n")


def plot_random_environments(env, num_envs=10):
    """随机生成并绘制多张环境二维平面图"""
    print("=" * 50)
    print(f"2. 随机生成 {num_envs} 张环境平面图 (请查看弹出的绘图窗口)")
    print("=" * 50)

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle("Random Environment Generation Test", fontsize=16)
    axes = axes.flatten()

    for i in range(num_envs):
        env.reset()
        ax = axes[i]

        # 绘制边界
        ax.set_xlim(0, env.cfg.env_size)
        ax.set_ylim(0, env.cfg.env_size)
        ax.set_aspect('equal')

        # 绘制障碍物
        for ox, oy, orad in env.obstacles:
            obs_circle = Circle((ox, oy), orad, color='gray', alpha=0.7)
            ax.add_patch(obs_circle)

        # 绘制起终点及 3m 安全区
        start_safe = Circle(env.start_pos, env.cfg.safe_radius, color='blue', alpha=0.2)
        target_safe = Circle(env.target_pos, env.cfg.safe_radius, color='green', alpha=0.2)
        ax.add_patch(start_safe)
        ax.add_patch(target_safe)

        ax.plot(env.start_pos[0], env.start_pos[1], 'bo', label='Start')
        ax.plot(env.target_pos[0], env.target_pos[1], 'g*', markersize=10, label='Target')

        ax.set_title(f"Env {i + 1}")
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.show()


def run_random_policy_and_analyze(env, total_steps=3000):
    """运行随机策略，抓取 info 数据并使用 Pandas 分析"""
    print("=" * 50)
    print(f"3. 运行随机策略 {total_steps} 步并分析奖励分布")
    print("=" * 50)

    env.reset()
    reward_data = []

    for step in range(total_steps):
        # 使用随机动作极限测试环境边界
        action = env.action_space.sample()
        obs, rewards, done, truncated, info = env.step(action)

        # 将每个智能体的奖励组件提取出来放入列表
        for i in range(env.n):
            agent_info = info[f'agent_{i}'].copy()
            reward_data.append(agent_info)

        if done or truncated:
            env.reset()

    # 使用 Pandas 进行数据统计分析
    df = pd.DataFrame(reward_data)

    # 我们需要的统计指标：平均值、方差、最小值、25%分位数、中位数、75%分位数、最大值
    stats_df = pd.DataFrame({
        'Mean': df.mean(),
        'Variance': df.var(),
        'Min': df.min(),
        '25%': df.quantile(0.25),
        'Median': df.median(),
        '75%': df.quantile(0.75),
        'Max': df.max()
    })

    # 格式化输出，保留四位小数以便于观察
    pd.set_option('display.float_format', lambda x: '%.4f' % x)
    print(f"经过 {total_steps} 步 (共 {total_steps * env.n} 个智能体步数) 的奖励组件统计如下：\n")
    print(stats_df)

    # 检查单步总奖励截断是否生效
    clipped_min = stats_df.loc['clipped_total', 'Min']
    clipped_max = stats_df.loc['clipped_total', 'Max']
    print("\n--- 截断规则检查 ---")
    print(f"设定限制: [-2.0000, 2.0000]")
    print(f"实际极值: [{clipped_min:.4f}, {clipped_max:.4f}]")
    if clipped_min >= -2.0001 and clipped_max <= 2.0001:
        print("状态：正常。截断机制完美生效！")
    else:
        print("状态：异常！截断机制失效，请检查代码。")


if __name__ == "__main__":
    # 实例化环境
    test_env = MultiUAVCoverageEnv(Config())

    # 1. 基础空间测试
    test_spaces_and_step(test_env)

    # 2. 场景生成可视化测试
    plot_random_environments(test_env, num_envs=10)

    # 3. 数据流与奖励分布分析
    run_random_policy_and_analyze(test_env, total_steps=3000)