import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

# 导入我们昨天写好的环境和配置
from env import MultiUAVCoverageEnv, Config, Plot


# ==========================================
# 1. 超参数配置 (MAPPO Hyperparameters)
# ==========================================
class PPOConfig:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 训练循环参数
    max_train_steps = 1000000  # 总训练环境步数 (约 300 万步)
    rollout_steps = 2048  # 每次策略更新前的采样步数
    mini_batch_size = 512  # PPO 更新的 mini-batch 大小
    ppo_epochs = 10  # 每次采样的网络更新轮数

    # 网络与优化器参数
    hidden_dim = 256  # MLP 隐藏层维度 (轻量级，利于边缘端部署)
    lr_actor = 3e-4  # Actor 初始学习率
    lr_critic = 1e-3  # Critic 初始学习率

    # RL 核心参数
    gamma = 0.99  # 折扣因子
    gae_lambda = 0.95  # GAE 平滑参数
    clip_param = 0.2  # PPO 裁剪阈值
    entropy_coef = 0.01  # 熵正则化系数 (鼓励探索)
    vloss_coef = 0.5  # 价值损失系数
    max_grad_norm = 0.5  # 梯度裁剪防爆阈值

    # 日志与保存
    log_interval = 1  # 每隔多少次更新打印一次日志
    save_interval = 50  # 每隔多少次更新保存一次模型
    model_dir = "./models"


# ==========================================
# 2. 网络结构与初始化 (Networks)
# ==========================================
def orthogonal_init(layer, gain=1.0):
    """正交初始化，有效防止深度网络梯度消失/爆炸，PPO 标配"""
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain=gain)
        nn.init.constant_(layer.bias, 0)


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))  # 独立可学习的对数标准差

        # 初始化
        self.net.apply(orthogonal_init)
        orthogonal_init(self.mean_layer, gain=0.01)  # 动作输出层 gain 设小，保证初始动作在 0 附近

    def forward(self, obs):
        x = self.net(obs)
        mean = torch.tanh(self.mean_layer(x))  # 将均值限制在 [-1, 1] 之间
        std = self.log_std.exp().expand_as(mean)
        return mean, std


class Critic(nn.Module):
    def __init__(self, global_obs_dim, hidden_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(global_obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.net.apply(orthogonal_init)

    def forward(self, global_obs):
        return self.net(global_obs)


# ==========================================
# 3. 经验回放池 (Rollout Buffer)
# ==========================================
class RolloutBuffer:
    def __init__(self, steps, num_agents, obs_dim, global_obs_dim, action_dim, device):
        self.obs = torch.zeros((steps, num_agents, obs_dim), dtype=torch.float32).to(device)
        self.global_obs = torch.zeros((steps, num_agents, global_obs_dim), dtype=torch.float32).to(device)
        self.actions = torch.zeros((steps, num_agents, action_dim), dtype=torch.float32).to(device)
        self.logprobs = torch.zeros((steps, num_agents, 1), dtype=torch.float32).to(device)
        self.rewards = torch.zeros((steps, num_agents, 1), dtype=torch.float32).to(device)
        self.values = torch.zeros((steps, num_agents, 1), dtype=torch.float32).to(device)
        self.dones = torch.zeros((steps, num_agents, 1), dtype=torch.float32).to(device)
        self.step = 0
        self.max_steps = steps

    def add(self, obs, global_obs, action, logprob, reward, value, done):
        self.obs[self.step] = obs
        self.global_obs[self.step] = global_obs
        self.actions[self.step] = action
        self.logprobs[self.step] = logprob
        self.rewards[self.step] = reward
        self.values[self.step] = value
        self.dones[self.step] = done
        self.step += 1

    def clear(self):
        self.step = 0

    def compute_returns_and_advantages(self, next_value, next_done, gamma, gae_lambda):
        """计算 GAE 优势函数"""
        advantages = torch.zeros_like(self.rewards)
        last_gae_lam = 0
        for t in reversed(range(self.max_steps)):
            if t == self.max_steps - 1:
                next_non_terminal = 1.0 - next_done
                next_values = next_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_values = self.values[t + 1]

            delta = self.rewards[t] + gamma * next_values * next_non_terminal - self.values[t]
            advantages[t] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam

        returns = advantages + self.values
        return returns, advantages


# ==========================================
# 4. MAPPO 算法主体 (Agent)
# ==========================================
class MAPPO:
    def __init__(self, obs_dim, action_dim, num_agents, config):
        self.cfg = config
        self.num_agents = num_agents
        self.global_obs_dim = obs_dim * num_agents  # CTDE 拼接全观测

        self.actor = Actor(obs_dim, action_dim, self.cfg.hidden_dim).to(self.cfg.device)
        self.critic = Critic(self.global_obs_dim, self.cfg.hidden_dim).to(self.cfg.device)

        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=self.cfg.lr_actor, eps=1e-5)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=self.cfg.lr_critic, eps=1e-5)

        self.buffer = RolloutBuffer(self.cfg.rollout_steps, num_agents, obs_dim,
                                    self.global_obs_dim, action_dim, self.cfg.device)

    def get_action_and_value(self, obs, global_obs, deterministic=False):
        with torch.no_grad():
            mean, std = self.actor(obs)
            dist = Normal(mean, std)
            if deterministic:
                action = mean
            else:
                action = dist.sample()

            # 限制动作边界
            action = torch.clamp(action, -1.0, 1.0)
            logprob = dist.log_prob(action).sum(dim=-1, keepdim=True)
            value = self.critic(global_obs)
        return action, logprob, value

    def evaluate_actions(self, obs, global_obs, actions):
        mean, std = self.actor(obs)
        dist = Normal(mean, std)
        logprobs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        values = self.critic(global_obs)
        return logprobs, values, entropy

    def update(self):
        # 1. 获取并处理数据
        obs = self.buffer.obs.view(-1, self.buffer.obs.shape[-1])
        global_obs = self.buffer.global_obs.view(-1, self.buffer.global_obs.shape[-1])
        actions = self.buffer.actions.view(-1, self.buffer.actions.shape[-1])
        old_logprobs = self.buffer.logprobs.view(-1, 1)

        # 优势归一化 (Mini-batch 级别外的一层归一化)
        returns = self.buffer.returns.view(-1, 1)
        advantages = self.buffer.advantages.view(-1, 1)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_size = obs.shape[0]
        indices = np.arange(dataset_size)

        total_loss, actor_loss, critic_loss = 0, 0, 0

        # 2. PPO 多轮迭代更新
        for _ in range(self.cfg.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, self.cfg.mini_batch_size):
                end = start + self.cfg.mini_batch_size
                mb_inds = indices[start:end]

                mb_obs = obs[mb_inds]
                mb_global_obs = global_obs[mb_inds]
                mb_actions = actions[mb_inds]
                mb_advantages = advantages[mb_inds]
                mb_returns = returns[mb_inds]
                mb_old_logprobs = old_logprobs[mb_inds]

                # 前向传播计算新的概率和价值
                new_logprobs, values, entropy = self.evaluate_actions(mb_obs, mb_global_obs, mb_actions)

                # 策略损失 (Actor)
                ratio = torch.exp(new_logprobs - mb_old_logprobs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_param, 1.0 + self.cfg.clip_param) * mb_advantages
                loss_pi = -torch.min(surr1, surr2).mean()

                # 价值损失 (Critic)
                loss_v = nn.MSELoss()(values, mb_returns)

                # 熵正则化 (鼓励探索)
                loss_ent = entropy.mean()

                # 总损失
                loss = loss_pi + self.cfg.vloss_coef * loss_v - self.cfg.entropy_coef * loss_ent

                # 梯度反向传播
                self.optimizer_actor.zero_grad()
                self.optimizer_critic.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
                self.optimizer_actor.step()
                self.optimizer_critic.step()

                total_loss += loss.item()
                actor_loss += loss_pi.item()
                critic_loss += loss_v.item()

        num_updates = self.cfg.ppo_epochs * (dataset_size // self.cfg.mini_batch_size)
        return total_loss / num_updates, actor_loss / num_updates, critic_loss / num_updates

    def lr_decay(self, current_step, total_steps):
        """学习率线性衰减"""
        lr_a = self.cfg.lr_actor * (1 - current_step / total_steps)
        lr_c = self.cfg.lr_critic * (1 - current_step / total_steps)
        for param_group in self.optimizer_actor.param_groups:
            param_group['lr'] = lr_a
        for param_group in self.optimizer_critic.param_groups:
            param_group['lr'] = lr_c


# ==========================================
# 5. 训练主循环 (Main Runner)
# ==========================================
def train():
    env = MultiUAVCoverageEnv(Config())
    cfg = PPOConfig()
    os.makedirs(cfg.model_dir, exist_ok=True)

    obs_dim = env.observation_space.shape[1]
    action_dim = env.action_space.shape[1]
    num_agents = env.n

    agent = MAPPO(obs_dim, action_dim, num_agents, cfg)

    # 状态与日志初始化
    obs, _ = env.reset()
    obs_tensor = torch.tensor(obs, dtype=torch.float32).to(cfg.device)
    global_obs_tensor = obs_tensor.flatten().unsqueeze(0).repeat(num_agents, 1)

    global_step = 0
    num_updates = int(cfg.max_train_steps // cfg.rollout_steps)

    # 记录数据用于最终绘图
    all_ep_rewards = []

    # 当前轮次指标收集
    ep_reward = 0
    ep_len = 0
    ep_cov_grids = 0

    # 历史滑动平均统计
    window_size = 50
    history_rewards, history_lens, history_succ, history_cov = [], [], [], []

    print("\n" + "=" * 80)
    print("   Step |  Avg_Rew | Avg_Len | Avg_Loss | Succ_Rate | Avg_Cov_Grids |   FPS")
    print("=" * 80)

    start_time = time.time()

    for update in range(1, num_updates + 1):
        # --- 阶段 1：环境交互与数据收集 ---
        for step in range(cfg.rollout_steps):
            global_step += 1

            # 获取动作
            action, logprob, value = agent.get_action_and_value(obs_tensor, global_obs_tensor)
            action_np = action.cpu().numpy()

            # 环境步进
            next_obs, rewards, done, truncated, info = env.step(action_np)
            ep_reward += np.sum(rewards) / num_agents
            ep_len += 1

            # 统计步内新增覆盖面积 (通过 info 解析)
            for i in range(num_agents):
                # cov 奖励 = c_cov * n_new。反推 n_new
                n_new = info[f'agent_{i}']['cov'] / env.cfg.c_cov
                ep_cov_grids += n_new

            # 转换为 Tensor
            next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32).to(cfg.device)
            next_global_obs_tensor = next_obs_tensor.flatten().unsqueeze(0).repeat(num_agents, 1)
            reward_tensor = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(cfg.device)
            # 因为 CTDE，当环境 done 时，所有 agent 的 done 是一致的
            done_tensor = torch.tensor([[float(done or truncated)]] * num_agents, dtype=torch.float32).to(cfg.device)

            # 存入 Buffer
            agent.buffer.add(obs_tensor, global_obs_tensor, action, logprob, reward_tensor, value, done_tensor)

            obs_tensor = next_obs_tensor
            global_obs_tensor = next_global_obs_tensor

            # 回合结束处理
            if done or truncated:
                all_ep_rewards.append(ep_reward)
                history_rewards.append(ep_reward)
                history_lens.append(ep_len)
                history_cov.append(ep_cov_grids)

                # 判定是否成功到达终点 (依据 Info 中 terminal 的奖励成分判断是否撞击)
                is_success = 1.0 if (done and not truncated and info['agent_0']['terminal'] > 0) else 0.0
                history_succ.append(is_success)

                # 维护滑动窗口大小
                if len(history_rewards) > window_size:
                    history_rewards.pop(0);
                    history_lens.pop(0)
                    history_succ.pop(0);
                    history_cov.pop(0)

                # 重置环境
                obs, _ = env.reset()
                obs_tensor = torch.tensor(obs, dtype=torch.float32).to(cfg.device)
                global_obs_tensor = obs_tensor.flatten().unsqueeze(0).repeat(num_agents, 1)
                ep_reward, ep_len, ep_cov_grids = 0, 0, 0

        # --- 阶段 2：计算优势与网络更新 ---
        with torch.no_grad():
            next_value = agent.critic(global_obs_tensor)
            # 判断最后一步是否终止
            next_done = torch.tensor([[float(done or truncated)]] * num_agents, dtype=torch.float32).to(cfg.device)

        returns, advantages = agent.buffer.compute_returns_and_advantages(next_value, next_done, cfg.gamma,
                                                                          cfg.gae_lambda)
        agent.buffer.returns = returns
        agent.buffer.advantages = advantages

        # 核心更新
        avg_loss, a_loss, c_loss = agent.update()
        agent.buffer.clear()

        # 学习率调度
        agent.lr_decay(global_step, cfg.max_train_steps)

        # --- 阶段 3：日志输出与模型保存 ---
        if update % cfg.log_interval == 0 and len(history_rewards) > 0:
            fps = int(global_step / (time.time() - start_time))
            avg_rew = np.mean(history_rewards)
            avg_l = np.mean(history_lens)
            succ_r = np.mean(history_succ) * 100
            avg_c = np.mean(history_cov)

            # 极具强迫症福音的对齐输出
            print(
                f"{global_step:>7} | {avg_rew:>8.2f} | {avg_l:>7.0f} | {avg_loss:>8.4f} | {succ_r:>7.1f}% | {avg_c:>13.1f} | {fps:>5}")

        if update % cfg.save_interval == 0:
            torch.save(agent.actor.state_dict(), os.path.join(cfg.model_dir, f"actor_step_{global_step}.pth"))
            torch.save(agent.critic.state_dict(), os.path.join(cfg.model_dir, f"critic_step_{global_step}.pth"))

    print("=" * 80)
    print("Training Completed! Generating final learning curves...")
    Plot.plot_learning_curve(all_ep_rewards, title="MAPPO Multi-UAV Cooperative Coverage")


if __name__ == "__main__":
    train()