import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from gymnasium.vector import SyncVectorEnv
from subway_env_latency_test import SubwayEnv
from tqdm import tqdm
from stats_logger import StatsLogger


# ---------- 시드 설정 ----------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------- 모델 저장 경로 ----------
MODEL_FINAL_PATH = "dqn_vector_final.pth" # 학습 종료시 최종 모델
MODEL_BEST_PATH = "dqn_vector_best.pth" # 학습 중 최고 성능 모델


# ---------- Dueling DQN ----------
class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.feature = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
        ) # 공통 특징 추출 (상태를 고차원 표현으로 변환)

        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        ) # Value Stream : V(s)

        self.adv_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        ) # Adavantage Stream : A(s,a)

    def forward(self, x):
        """
        Q(s,a) = V(s) + (A(s,a)-mean(A(s,a)))
        안정적인 학습을 위해 mean을 빼서 정규화 효과를 냄
        """
        f = self.feature(x)
        V = self.value_stream(f)
        A = self.adv_stream(f)
        return V + (A - A.mean(dim=1, keepdim=True))


# ---------- Replay Buffer ----------
class ReplayBuffer: # 과거 경험 저장 & sampling
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # transition: (s,a,r,s',done)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # batch size개의 경험을 sampling
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return ( # tensor 형태로 반환
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.int64).unsqueeze(-1),
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32).unsqueeze(-1),
        )

    def __len__(self):
        return len(self.buffer)


# ---------- ε-greedy 행동 선택 ----------
# exploration vs exploitation 
def select_action(policy_net, states, epsilon, action_dim, device):
    if random.random() < epsilon:
        return np.random.randint(0, action_dim, size=(states.shape[0],))
    with torch.no_grad(): # gradient 계산없이 Q 최대인 행동 선택
        states_t = torch.tensor(states, dtype=torch.float32, device=device)
        q = policy_net(states_t)
        return q.argmax(dim=1).cpu().numpy()


# ---------- DQN 학습 main 함수 ----------
def train_dqn(config):
    """
    Double DQN + Dueling 아키텍쳐
    - 256개의 환경이 병렬로 실행하여 데이터 수집 
    - policy net: 행동 선택 / target net(고정): 평가 
    """
    # 시드 설정
    seed = config.get("seed", 0)
    set_seed(seed)

    # 하이퍼파라미터 설정
    num_episodes           = config.get("num_episodes", 4000) # 총 에피소드 수
    num_envs               = config.get("num_envs", 256) # 병렬 환경 개수
    buffer_capacity        = config.get("buffer_capacity", 2_000_000) 
    batch_size             = config.get("batch_size", 4096) 
    gamma                  = config.get("gamma", 0.99) # discount factor
    lr                     = config.get("lr", 1e-4) 
    start_learning         = config.get("start_learning", 20000)
    learn_every            = config.get("learn_every", 2) # N step마다 학습
    target_update_interval = config.get("target_update_interval", 5000)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device} (seed={seed})")

    # seed별로 CSV 이름 분리
    logger = StatsLogger(csv_path=f"dqn_vector_stats_seed{seed}.csv")

    # Vectorized environments
    def make_env(): 
        # 256개의 SubwayEnv 동시 실행 
        return SubwayEnv(render_mode=None)

    env = SyncVectorEnv([make_env for _ in range(num_envs)])
    # 각 env에 다른 seed 부여
    env_seeds = [seed + i for i in range(num_envs)]
    state_dim = env.single_observation_space.shape[0]
    action_dim = env.single_action_space.n

    # 네트워크 초기화 
    policy_net = DuelingDQN(state_dim, action_dim).to(device)
    target_net = DuelingDQN(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(buffer_capacity)

    # epsilon 1.0~0.1까지 50만 스텝에 걸쳐 선형 decay
    epsilon, epsilon_end, epsilon_decay_steps = 1.0, 0.01, 500_000
    global_step = 0

    # episode 추적 변수
    episode_rewards = np.zeros(num_envs)
    finished_rewards = []
    completed_eps = 0

    # 최고 모델 추적
    best_reward = -float("inf")

    # seed별 모델 저장 경로
    model_final_path = f"dqn_vector_final_seed{seed}.pth"
    model_best_path  = f"dqn_vector_best_seed{seed}.pth"

    # 초기 state
    states, infos = env.reset(seed=env_seeds)
    pbar = tqdm(total=num_episodes, desc="Training Progress")

    # main 학습 loop
    while completed_eps < num_episodes:
        global_step += num_envs # 256개 state 동시 진행
        epsilon = max(epsilon_end, 1.0 - global_step / epsilon_decay_steps)

        # 행동 선택 및 실행
        actions = select_action(policy_net, states, epsilon, action_dim, device)
        next_states, rewards, terminated, truncated, infos = env.step(actions)
        dones = np.logical_or(terminated, truncated)

        # transitions 저장 
        for i in range(num_envs):
            replay_buffer.push(states[i], actions[i], rewards[i], next_states[i], dones[i])

        episode_rewards += rewards

        # Learning step
        if global_step % learn_every == 0 and len(replay_buffer) >= start_learning:
            s, a, r, ns, d = replay_buffer.sample(batch_size)
            s, a, r, ns, d = s.to(device), a.to(device), r.to(device), ns.to(device), d.to(device)

            # 현재 Q값: Q(s,a)
            q_vals = policy_net(s).gather(1, a) # [batch, 1]

            # Double DQN target 계산 
            with torch.no_grad():
                # policy net으로 next state의 최선 행동 선택
                next_actions = policy_net(ns).argmax(dim=1, keepdim=True)
                # target net으로 해당 행동의 Q값 평가
                next_q = target_net(ns).gather(1, next_actions)
                # Bellman Equation
                target = r + (1.0 - d) * gamma * next_q

            loss = nn.functional.smooth_l1_loss(q_vals, target)

            # 역전파 및 가중치 업데이트
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # target network 주기적 업데이트 
            # 일정 스텝마다 policy net -> target net 가중치 복사
            if global_step % target_update_interval == 0:
                target_net.load_state_dict(policy_net.state_dict())

        # Episode termination 처리 
        for i in range(num_envs):
            if dones[i]:
                ep_reward = float(episode_rewards[i])
                finished_rewards.append(ep_reward)

                # 환경별 info 딕셔너리 추출 
                info = {}
                if isinstance(infos, dict):
                    for key, values in infos.items():
                        try:
                            info[key] = values[i]
                        except Exception:
                            info[key] = None

                # 최고 성능 모델 저장 
                if ep_reward > best_reward:
                    best_reward = ep_reward
                    checkpoint = {
                        "model_state_dict": policy_net.state_dict(),
                        "target_state_dict": target_net.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_reward": best_reward,
                        "episode": completed_eps + 1,
                        "seed": seed,
                    }
                    torch.save(checkpoint, model_best_path)

                # 로컬 CSV 로그
                logger.log_episode(
                    episode=completed_eps + 1,
                    total_reward=ep_reward,
                    length=int(info.get("time_step", 0)),
                    epsilon=float(epsilon),
                    info=info,
                )

                episode_rewards[i] = 0.0
                completed_eps += 1
                pbar.update(1)

        states = next_states

    # 학습 종료 
    pbar.close()
    logger.save_csv()

    # 최종 모델 저장
    final_checkpoint = {
        "model_state_dict": policy_net.state_dict(),
        "target_state_dict": target_net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "seed": seed,
    }
    torch.save(final_checkpoint, model_final_path)

    print(f"Training completed — CSV saved: {logger.csv_path}")
    print(f"Final Model saved to: {model_final_path}")
    print(f"Best Model saved to: {model_best_path}")
    if len(finished_rewards) > 0:
        print("Avg reward (last 200):", np.mean(finished_rewards[-200:]))


if __name__ == "__main__":
    base_config = {
        "num_episodes": 7000,
        "num_envs": 256,
        "buffer_capacity": 2_000_000,
        "batch_size": 8192,
        "gamma": 0.985,
        "lr": 1e-4,
        "start_learning": 50_000,
        "learn_every": 2,
        "target_update_interval": 5000,
    }

    seed=0
    config = dict(base_config)
    config["seed"] = seed
    print(f"\n[INFO] Start DQN training (seed={seed})")
    train_dqn(config)
