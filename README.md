# RL_Project_Thunder_Rush_3-Lane_Infinite_Runner
## Cloud Rush: 3-Lane Infinite Runner RL Agent

Cloud Rush는 **3-lane 무한 달리기 게임 환경**에서  
Agent가 **두 개의 강화학습(DQN, PPO) 알고리즘** 으로 장애물을 회피하며 최대한 오래 생존하도록 학습하는 프로젝트입니다.

---

## 🎮 Environment Overview

- **장르**: 3-Lane Infinite Runner
- **행동(Action)**
- 행동은 discrete space(5):
  - 0: 유지 (stay)
  - 1: 왼쪽 이동 (left)
  - 2: 오른쪽 이동 (right)
  - 3: 점프 (jump)
  - 4: 슬라이드 (slide)


- **장애물 타입(Obstacles)**  
  - **A**: 점프로만 회피 가능  
  - **B**: 슬라이드로만 회피 가능  
  - **C**: 피할 수 없는 패턴 (unavoidable) / 좌우이동으로만 회피 가능

- **상태(State) 예시**

  ```python
  [player_lane, speed, time_ratio,
   lane0_exists, lane0_dist, lane0_type,
   lane1_exists, lane1_dist, lane1_type,
   lane2_exists, lane2_dist, lane2_type, ...]

- **보상 (Reward)**
  - 기본 생존 보상 
    매 time step 마다 +0.1
  - 장애물 회피 보상
    장애물이 dist ≤ 0에 도달하면 지나간 것으로 간주하여 보상을 부여
    - A/B 장애물
      적절한 action: +2.0
      다른 action으로 생존: +1.0
    - C 장애물
      적절한 action: +2.0
      틀린 action: episode terminate
  - Penalty
    - 이동 penalty : 불필요한 lane 이동시, -0.01
    - 충돌 penalty : 장애물 피하지 못할시 -10.0, episode terminate
			(사망사유는 통계로 저장됨 :  (A_no_jump, B_no_slide, C_unavoidable))
 
