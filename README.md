# ⚡Thunder Rush: 3-Lane Infinite Runner RL Agent⚡

Cloud Rush는 **3-lane 무한 달리기 게임 환경**에서  
Agent가 **두 개의 강화학습(DQN, PPO) 알고리즘** 으로 장애물을 회피하며 최대한 오래 생존하도록 학습하는 프로젝트입니다.

환경 설계부터 RL 알고리즘(DQN, PPO) 비교·개선까지 직접 수행했습니다.

---

## 🎯 프로젝트 소개 (Project Overview)

- **목표:** Agent가 장애물을 회피하며 최장 생존 시간 달성
- **환경:** 3-Lane Infinite Runner (custom Gym-like environment)
- **알고리즘:** Double+​Dueling DQN, PPO+GAE
- **학습 방식:** Vectorized Parallel Training  
  - DQN: 256 environments  
  - PPO: 64 environments
    
---

## 🧠 RL Algorithms

이 프로젝트에서는 두 가지 강화학습 알고리즘을 비교/실험합니다.

### 1) Double DQN + Dueling Architecture (Vectorized)

- Double DQN:  
  - 행동 선택: policy_net  
  - Q값 평가: target_net  
  - max 연산으로 인한 Q overestimation 완화

- Dueling DQN:  
  - Q(s,a) = V(s) + A(s,a) - mean(A(s,·))  
  - 상태 가치와 행동 이점을 분리해서 더 안정적으로 학습
- Vectorized Env:  
  - 256개 환경 동시 실행로 매 step마다 256 transition 수집

### 2) PPO + GAE (Proximal Policy Optimization)

- on-policy actor–critic 알고리즘
- GAE(λ)를 사용해 bias–variance trade-off 조절
- Clipped objective로 정책 업데이트 폭을 제한해 안정적 학습
- 64개 환경을 병렬로 실행하여 rollout 기반 학습

---

## 🎥 시각 자료 

> 게임 플레이 이미지, 학습 곡선, 사망 분포 등 넣을 예정

| 게임 화면 | 학습 보상 곡선 |
|----------|----------------|
| ![game](assets/gameplay.gif) | ![reward](assets/reward_curve.png) |

(파일 추가 후 경로 맞춰 넣으면 됨)

---

## 🛠 설치 방법 (Installation)

```bash
git clone https://github.com/<your-id>/Cloud-Rush.git
cd Cloud-Rush

# 가상 환경 권장
# python -m venv venv
# source venv/bin/activate

pip install -r requirements.txt

---

## 🛠 사용법 (Usage)

1) DQN 학습
python python train_dqn_vector_seed_2.py

2) PPO 학습
python train_ppo_vector_seed_2.py

3) 게임 실행
python subway_env_latency_test.py

4)
python stats_logger.py




