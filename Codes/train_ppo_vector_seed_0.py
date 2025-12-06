import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from gymnasium.vector import SyncVectorEnv
from torch.distributions import Categorical
from subway_env_latency_test import SubwayEnv
from stats_logger import StatsLogger
from tqdm import tqdm

# ---------- 시드 설정 ----------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
# ---------- Model Save Paths ----------
MODEL_FINAL_PATH = "ppo_vector_final.pth"
MODEL_BEST_PATH = "ppo_vector_best.pth"


# Actor-Critic Network (PPO)
class ActorCritic(nn.Module):
    """
    최적화된 Actor-Critic 네트워크
    - Actor (policy) : 행동 확률 분포 출력
    - Critic (value) : 상태 가치 추정
    개선 사항 
    - Dropout 제거: PPO는 on-policy 알고리즘으로 dropout이 학습을 불안정하게 만듦
    - LayerNorm 추가: 학습 안정성 향상 (배치마다 정규화)
    """
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )

        # Policy head : 행동 확률 분포 출력 
        # Value head : 상태 가치 V(s) 출력
        self.policy_head = nn.Linear(128, action_dim)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x): # 순전파
        h = self.shared(x)
        logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        return logits, value

    def get_action_and_value(self, state):
        # 행동 선택과 가치 추정을 동시 수행
        logits, value = self.forward(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value, dist

# PPO 전용 Rollout Buffer
class PPOBuffer:
    """
    - DQN의 Replay Buffer와 달리 한번의 roll out 데이터만 저장 가능
    - 업데이트 후 buffer 비우고 새 데이터 수집
    - 모든 환경의 데이터를 배치로 처리 
    """
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def add(self, state, action, log_prob, reward, done, value):
        # 한 스텝의 모든 환경 데이터 저장 
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def get(self):
        """
        저장된 데이터를 numpy array 반환
        return: (rollout_steps, num_envs, ...) 형태
        """
        return (
            np.array(self.states, dtype=np.float32),
            np.array(self.actions, dtype=np.int64),
            np.array(self.log_probs, dtype=np.float32),
            np.array(self.rewards, dtype=np.float32),
            np.array(self.dones, dtype=np.float32),
            np.array(self.values, dtype=np.float32),
        )

    def clear(self):
        # 업데이트 후 buffer 초기화
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()


def compute_gae(rewards, dones, values, next_value, gamma=0.99, lam=0.95):
    """
    Generalized Advantage Estimation (GAE) 계산
    - PPO : bias, variance trade-off 조절

    """
    T = len(rewards)
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = 0.0

    # 미래->현재 (역순으로 계산)
    for t in reversed(range(T)):
        if t == T - 1:
            # 마지막 스텝 : next_value 사용 
            next_non_terminal = 1.0 - dones[-1]
            next_value_est = next_value
        else:
            # 중간 스텝 : 다음 step의 value 사용
            next_non_terminal = 1.0 - dones[t + 1]
            next_value_est = values[t + 1]

        # TD error 계산 :  δ_t = r_t + γV(s_{t+1}) - V(s_t)
        delta = rewards[t] + gamma * next_value_est * next_non_terminal - values[t]
        # GAE 재귀식 : A_t = δ_t + γλA_{t+1}
        last_gae = delta + gamma * lam * next_non_terminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values # 실제 수익 근사
    return advantages, returns


# PPO Training main 함수
def train_ppo(config):
    """
    최적화된 PPO 학습 함수
    학습 흐름:
    1. Rollout: 현재 정책으로 데이터 수집 (256 스텝 × 64 환경)
    2. GAE 계산: 각 행동의 이점 계산
    3. PPO Update: 미니배치로 여러 번 정책 업데이트
    4. 반복
    """
    # 시드 설정
    seed = config.get("seed", 0)
    set_seed(seed)

    # 하이퍼파라미터
    num_episodes     = config.get("num_episodes", 100000)
    num_envs         = config.get("num_envs", 64)
    rollout_steps    = config.get("rollout_steps", 256)
    update_epochs    = config.get("update_epochs", 6)
    mini_batch_size  = config.get("mini_batch_size", 1024)
    gamma            = config.get("gamma", 0.99)
    lam              = config.get("lam", 0.95)
    clip_coef        = config.get("clip_coef", 0.2)
    lr               = config.get("lr", 2.5e-4)
    entropy_coef     = config.get("entropy_coef", 0.01)
    value_coef       = config.get("value_coef", 0.5)
    max_grad_norm    = config.get("max_grad_norm", 0.5)
    target_kl        = config.get("target_kl", 0.02)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device} | Envs: {num_envs} | Algorithm: PPO (Vectorized) | seed={seed}")

    # 환경 생성
    def make_env():
        return SubwayEnv(render_mode=None)

    env = SyncVectorEnv([make_env for _ in range(num_envs)])
    # 각 env에 다른 seed 부여
    env_seeds = [seed + i for i in range(num_envs)]
    state_dim = env.single_observation_space.shape[0]
    action_dim = env.single_action_space.n

    print(f"[INFO] State dim: {state_dim}, Action dim: {action_dim}")

    # 모델 및 옵티마이저
    model = ActorCritic(state_dim, action_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-5)
    
    # Learning rate scheduler : cosine decay 
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_episodes, eta_min=lr * 0.1
    )

    buffer = PPOBuffer()
    # seed별로 CSV 이름 분리
    logger = StatsLogger(csv_path=f"ppo_vector_stats_seed{seed}.csv")

    # 학습 상태 변수
    episode_rewards = np.zeros(num_envs, dtype=np.float32)
    finished_rewards = []
    completed_eps = 0
    best_avg_reward = -float("inf")
    global_step = 0

    states, _ = env.reset(seed=env_seeds)
    pbar = tqdm(total=num_episodes, desc="PPO Training")

    # seed별 모델 저장 경로
    model_final_path = f"ppo_vector_final_seed{seed}.pth"
    model_best_path  = f"ppo_vector_best_seed{seed}.pth"

    # 메인 학습 루프
    while completed_eps < num_episodes:
        buffer.clear()
        steps_in_rollout = 0

        # Rollout Phase (데이터 수집)
        # 현재 policy로 rollout_steps 동안 환경과 상호작용 
        while steps_in_rollout < rollout_steps and completed_eps < num_episodes:
            states_t = torch.tensor(states, dtype=torch.float32, device=device)

            with torch.no_grad(): # 행동 선택 및 가치 추정 
                actions_t, log_probs_t, values_t, _ = model.get_action_and_value(states_t)

            actions = actions_t.cpu().numpy()
            log_probs = log_probs_t.cpu().numpy()
            values = values_t.cpu().numpy()

            next_states, rewards, terminated, truncated, infos = env.step(actions)
            dones = np.logical_or(terminated, truncated)

            # Buffer에 vectorized 데이터 저장 (모든 환경 동시에)
            buffer.add(states, actions, log_probs, rewards, dones, values)

            episode_rewards += rewards
            global_step += num_envs

            # 종료된 에피소드 처리
            for i in range(num_envs):
                if dones[i]:
                    ep_reward = float(episode_rewards[i])
                    finished_rewards.append(ep_reward)

                    # 환경별 info 추출
                    info = {}
                    if isinstance(infos, dict):
                        for key, vals in infos.items():
                            try:
                                info[key] = vals[i]
                            except:
                                info[key] = None

                    # 통계 로깅 (PPO : deterministic policy. epsilon X)
                    logger.log_episode(
                        episode=completed_eps + 1,
                        total_reward=ep_reward,
                        length=int(info.get("time_step", 0)),
                        info=info,
                        epsilon=None,
                    )

                    episode_rewards[i] = 0.0
                    completed_eps += 1
                    pbar.update(1)

                    # 주기적 평가 및 Best 모델 저장
                    if completed_eps % 50 == 0 and len(finished_rewards) >= 100:
                        recent = finished_rewards[-100:]
                        avg_recent = float(np.mean(recent))
                        
                        # 최고 기록 갱신시 checkpoint 저장
                        if avg_recent > best_avg_reward:
                            best_avg_reward = avg_recent
                            checkpoint = {
                                "model_state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "scheduler_state_dict": scheduler.state_dict(),
                                "best_avg_reward": best_avg_reward,
                                "completed_episodes": completed_eps,
                                "global_step": global_step,
                                "seed": seed,
                            }
                            torch.save(checkpoint, model_best_path)
                            print(f"\n[BEST] Episode {completed_eps}: Avg reward = {avg_recent:.2f}")

            states = next_states
            steps_in_rollout += 1

        # PPO Update Phase (정책 학습)
        # 수집한 rollout data로 네트워크 업데이트
        (states_b, actions_b, old_log_probs_b, 
         rewards_b, dones_b, values_b) = buffer.get()

        # 마지막 상태의 가치 추정 (bootstrap)
        last_state_t = torch.tensor(states, dtype=torch.float32, device=device)
        with torch.no_grad():
            _, next_value_t = model(last_state_t)
        next_value = next_value_t.cpu().numpy()

        # GAE 계산 (advantage와 return 계산)
        advantages, returns = compute_gae(
            rewards_b, dones_b, values_b, next_value, gamma=gamma, lam=lam
        )

        # 데이터 Flatten
        b_states = states_b.reshape((-1,) + env.single_observation_space.shape)
        b_actions = actions_b.reshape(-1)
        b_log_probs = old_log_probs_b.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        # Tensor 변환 및 GPU 전송 
        states_t = torch.tensor(b_states, dtype=torch.float32, device=device)
        actions_t = torch.tensor(b_actions, dtype=torch.int64, device=device)
        old_log_probs_t = torch.tensor(b_log_probs, dtype=torch.float32, device=device)
        advantages_t = torch.tensor(b_advantages, dtype=torch.float32, device=device)
        returns_t = torch.tensor(b_returns, dtype=torch.float32, device=device)

        # Advantage 정규화
        # 평균 0, 표준편차 1 조정 
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        num_samples = states_t.size(0)
        indices = np.arange(num_samples)

        # Mini-batch SGD로 업데이트
        epoch_approx_kls = []

        for epoch in range(update_epochs):
            np.random.shuffle(indices)
            
            # 미니배치 단위로 업데이트
            for start in range(0, num_samples, mini_batch_size):
                end = start + mini_batch_size
                batch_idx = indices[start:end]

                # Forward pass
                logits, values_pred = model(states_t[batch_idx])
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(actions_t[batch_idx])
                entropy = dist.entropy().mean()

                # Policy loss (PPO clip)
                # ratio = π_new(a|s) / π_old(a|s)
                ratio = torch.exp(new_log_probs - old_log_probs_t[batch_idx])
                
                # 목적함수 계산 (둘 중 작은값으로 보수적 업데이트)
                surr1 = ratio * advantages_t[batch_idx]
                surr2 = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * advantages_t[batch_idx]
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.functional.mse_loss(values_pred, returns_t[batch_idx])

                # Total loss
                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

                # Metrics 수집
                with torch.no_grad():
                    approx_kl = (old_log_probs_t[batch_idx] - new_log_probs).mean().item()
                    epoch_approx_kls.append(approx_kl)

            # Early stopping (KL divergence가 너무 크면 중단)
            mean_kl = np.mean(epoch_approx_kls[-len(range(0, num_samples, mini_batch_size)):])
            if mean_kl > target_kl:
                print(f"[INFO] Early stopping at epoch {epoch+1}/{update_epochs} (KL={mean_kl:.4f})")
                break

        # Learning rate 업데이트
        scheduler.step()

    # 학습 종료 
    pbar.close()
    logger.save_csv()

    # 최종 모델 저장
    final_checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "completed_episodes": completed_eps,
        "global_step": global_step,
        "best_avg_reward": best_avg_reward,
        "seed": seed,
    }
    torch.save(final_checkpoint, model_final_path)

    print(f"\n{'='*60}")
    print(f"[TRAINING COMPLETE]")
    print(f"{'='*60}")
    print(f"Total Episodes: {completed_eps}")
    print(f"Total Steps: {global_step}")
    print(f"CSV saved: {logger.csv_path}")
    print(f"Final model: {model_final_path}")
    print(f"Best model: {model_best_path}")
    
    if len(finished_rewards) >= 200:
        print(f"Last 200 episodes avg reward: {np.mean(finished_rewards[-200:]):.2f}")
    if len(finished_rewards) >= 100:
        print(f"Last 100 episodes avg reward: {np.mean(finished_rewards[-100:]):.2f}")
    print(f"Best avg reward (100-ep window): {best_avg_reward:.2f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # 최적화된 하이퍼파라미터
    base_config = {
        # 학습 설정
        "num_episodes": 7000,
        "num_envs": 64,          
        "rollout_steps": 256,    
        "update_epochs": 6,     
        "mini_batch_size": 1024,
        
        # PPO 파라미터
        "gamma": 0.99,
        "lam": 0.95,
        "clip_coef": 0.2,
        "target_kl": 0.02, # Early stopping
        
        # 최적화
        "lr": 2.5e-4,
        "entropy_coef": 0.01,
        "value_coef": 0.5,
        "max_grad_norm": 0.5,
    }

    seed = 0 
    config = dict(base_config)
    config["seed"] = seed
    print(f"\n[INFO] Start PPO training (seed={seed})")
    train_ppo(config)