# subway_env_latency_test.py

import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    import pygame
    pygame_available = True
except ImportError:
    pygame_available = False


class SubwayEnv(gym.Env):
    """
    Subway Surfer 강화학습 환경 
    - 3개 lane 
    - 3가지 장애물 type:
        - A : jump로만 회피 가능
        - B : slide로만 회피 가능
        - C : 좌우이동으로만 회피 가능 
    - time이 지날수록 speed 증가 + 난이도 상승 
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()

        # 0: 유지, 1: 왼쪽, 2: 오른쪽, 3: 점프, 4: 슬라이드
        self.action_space = spaces.Discrete(5)

        """
        One-hot encoding으로 state 차원 변경
        수정: 27차원 (type을 [is_A, is_B, is_C] 3차원으로)
        구조:
            [0] player_lane (0-2)
            [1] speed (0-3)
            [2] time_ratio (0-1)
            [3-6] lane0 첫번째 장애물: exists, dist, is_A, is_B, is_C
            [7-11] lane1 첫번째 장애물: exists, dist, is_A, is_B, is_C
            [12-16] lane2 첫번째 장애물: exists, dist, is_A, is_B, is_C
            [17-21] lane0 두번째 장애물: exists, dist, is_A, is_B, is_C
            [22-26] lane1 두번째 장애물: exists, dist, is_A, is_B, is_C
            [27-31] lane2 두번째 장애물: exists, dist, is_A, is_B, is_C
        """
        self.observation_space = spaces.Box(
            low=np.array([0] * 33, dtype=np.float32),
            high=np.array([
                2, 3, 1,  # player_lane, speed, time_ratio
                # exists, dist, is_A, is_B, is_C

                # 각 레인의 첫번째장애물
                1, 1, 1, 1, 1, # lane0
                1, 1, 1, 1, 1, # lane1
                1, 1, 1, 1, 1, # lane2
                
                # 각 레인의 두번째 장애물
                1, 1, 1, 1, 1, # lane0
                1, 1, 1, 1, 1, # lane1
                1, 1, 1, 1, 1, # lane2
            ], dtype=np.float32),
            dtype=np.float32
        )

        self.render_mode = render_mode
        self.max_time_steps = 3000

        # 사망 통계 추적
        # 어떤 상황에서 자주 죽는지 분석하여 학습 개선
        self.death_stats = {
            "collision_A_no_jump": 0, 
            "collision_B_no_slide": 0,
            "collision_C": 0,
            "total_deaths": 0,
            "survival_time": []
        }
        self.current_death_reason = None

        # 어떤 패턴에서 죽었는지 기록 
        self.current_pattern_group = None
        self.current_pattern_name = None

        self._init_patterns()

        # Pygame 렌더링 초기화 
        if self.render_mode == "human" and pygame_available:
            pygame.init()
            self.screen_width = 400
            self.screen_height = 600
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Subway RL Environment")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 24)
            self.small_font = pygame.font.Font(None, 20)

        # 플레이어 렌더링 params
        self.base_y = 500
        self.player_y_offset = 0
        self.player_height = 40
        
        # jump/slide 지속기간 
        self.jump_frame = 0
        self.slide_frame = 0
        self.max_jump_frames = 15
        self.max_slide_frames = 15

        self.reset()

    def _init_patterns(self):
        """
        장애물 패턴 정의
        패턴 구조 : [(레인, 타입), (레인, 타입), ...]
        - 레인 : 0, 1, 2
        - 타입 : A(점프), B(슬라이드), C(좌우이동)

        난이도 단계
        1. Easy : 단일 장애물
        2. Medium : 2개 장애물
        3. Hard : 3개 장애물 
        """
        self.patterns = {
            # Easy 패턴
            "easy_left_A": [(0, "A")],
            "easy_left_B": [(0, "B")],
            "easy_left_C": [(0, "C")],
            "easy_center_A": [(1, "A")],
            "easy_center_B": [(1, "B")],
            "easy_center_C": [(1, "C")],
            "easy_right_A": [(2, "A")],
            "easy_right_B": [(2, "B")],
            "easy_right_C": [(2, "C")],
            
            # Medium 패턴
            "medium_sides_A": [(0, "A"), (2, "A")],
            "medium_sides_B": [(0, "B"), (2, "B")],
            "medium_left_center": [(0, "A"), (1, "B")],
            "medium_right_center": [(1, "A"), (2, "B")],
            "medium_mixed_LR": [(0, "B"), (2, "A")],
            "medium_complex_1": [(0, "B"), (1, "C")],
            "medium_complex_2": [(0, "B"), (2, "C")],
            "medium_complex_3": [(1, "B"), (2, "C")],
            "medium_complex_4": [(0, "C"), (1, "A")],
            "medium_complex_5": [(0, "C"), (2, "A")],
            "medium_complex_6": [(1, "C"), (2, "A")],
            
            # Hard 패턴
            "hard_all_A": [(0, "A"), (1, "A"), (2, "A")],
            "hard_all_B": [(0, "B"), (1, "B"), (2, "B")],
            "hard_sides_block_1": [(0, "C"), (1, "C")],
            "hard_sides_block_2": [(1, "C"), (2, "C")],
            "hard_sides_block_3": [(0, "C"), (2, "C")],
            "hard_complex_1": [(0, "A"), (1, "C"), (2, "B")],
            "hard_complex_2": [(0, "B"), (1, "A"), (2, "C")],
            "hard_complex_3": [(0, "A"), (1, "C"), (2, "C")],
            "hard_complex_4": [(0, "C"), (1, "A"), (2, "C")],
            "hard_complex_5": [(0, "C"), (1, "C"), (2, "A")],
            "hard_complex_6": [(0, "B"), (1, "C"), (2, "C")],
            "hard_complex_7": [(0, "C"), (1, "B"), (2, "C")],
            "hard_complex_8": [(0, "C"), (1, "C"), (2, "B")],
            "hard_complex_9": [(0, "A"), (1, "B"), (2, "C")],
            "hard_complex_10": [(0, "C"), (1, "A"), (2, "B")],
        }

        # 패턴 그룹 정의 (난이도별 분류)
        self.pattern_groups = {
            "easy": ["easy_left_A", "easy_left_B", "easy_left_C", 
                     "easy_center_A", "easy_center_B", "easy_center_C",
                     "easy_right_A", "easy_right_B", "easy_right_C"],
            "medium": ["medium_sides_A", "medium_sides_B", "medium_left_center",
                       "medium_right_center", "medium_mixed_LR", "medium_complex_1",
                       "medium_complex_2", "medium_complex_3", "medium_complex_4",
                       "medium_complex_5", "medium_complex_6"],
            "hard": ["hard_all_A", "hard_all_B", "hard_sides_block_1",
                     "hard_sides_block_2", "hard_sides_block_3", "hard_complex_1",
                     "hard_complex_2", "hard_complex_3", "hard_complex_4",
                     "hard_complex_5", "hard_complex_6", "hard_complex_7",
                     "hard_complex_8", "hard_complex_9", "hard_complex_10"]
        }

    def _get_pattern_probabilities(self):
        """
        시간에 따른 동적 난이도 조절
        - 초반 : 쉬운 패턴 위주로 기본기 학습
        - 중반 : 중간 난이도 증가
        - 후반 : 어려운 패턴 비중 증가 
        """
        if self.time_step < 600: # 초반 20초 : 쉬운 패턴 70%
            return {"easy": 0.7, "medium": 0.25, "hard": 0.05}
        elif self.time_step < 1200: # 20-40초 : 균형
            return {"easy": 0.5, "medium": 0.35, "hard": 0.15}
        elif self.time_step < 1600: # 40-53초 : 중간 난이도 위주
            return {"easy": 0.35, "medium": 0.4, "hard": 0.25}
        else: # 53초 이후 : 어려운 패턴 위주
            return {"easy": 0.25, "medium": 0.3, "hard": 0.45}

    def spawn_obstacles(self):
        """
        패턴 기반 장애물 생성
        - 현재 시간에 맞는 난이도 확률 계산
        - 해당 난이도에서 무작위 패턴 선택
        - 패턴에 정의된 장애물들을 동시 생성
        """
        probs = self._get_pattern_probabilities()
        
        # 난이도 선택 (확률적)
        difficulty = np.random.choice(
            ["easy", "medium", "hard"],
            p=[probs["easy"], probs["medium"], probs["hard"]]
        )

        # 해당 난이도에서 패턴 무작위 선택 
        pattern_name = np.random.choice(self.pattern_groups[difficulty])
        pattern = self.patterns[pattern_name]

        # 현재 패턴 정보 저장 
        self.current_pattern_group = difficulty
        self.current_pattern_name = pattern_name

        # 패턴의 모든 장애물 생성         
        for lane, obs_type in pattern:
            self.obstacles.append({
                "lane": lane,
                "dist": 1.0,
                "type": obs_type
            })

    def reset(self, seed=None, options=None):
        """
        환경 초기화
        - 에피소드 시작 시 또는 게임 오버 후 호출
        """

        super().reset(seed=seed)
        self.player_lane = 1
        self.speed = 1.0
        self.time_step = 0
        self.obstacles = []
        self.current_death_reason = None
        self.jump_frame = 0
        self.slide_frame = 0

        self.current_pattern_group = None
        self.current_pattern_name = None
        
        # 첫번째 장애물 생성
        self.spawn_obstacles()
        return self._get_state(), {}

    def _get_state(self):
        """
        One-hot encoding으로 장애물 타입
        현재 상태를 관측 벡터로 변환 
        """
        player_info = [ # 플레이어 정보 
            float(self.player_lane),
            float(self.speed),
            float(min(self.time_step / self.max_time_steps, 1.0))
        ]
        
        # lane별 장애물 분류
        lane_obstacles = {0: [], 1: [], 2: []}
        for obs in self.obstacles:
            lane_obstacles[obs["lane"]].append(obs)
        
        # 각 lane의 장애물을 거리순으로 정렬 (가까운 순)
        for lane in lane_obstacles:
            lane_obstacles[lane].sort(key=lambda x: x["dist"])
        
        def get_obstacle_features(obs):
            # 장애물 하나를 [exists, dist, is_A, is_B, is_C]로 변환 (5차원 벡터)
            if obs is None: # 장애물 없음 
                return [0.0, 1.0, 0.0, 0.0, 0.0]
            
            # One hot encoding 타입 표현 
            is_A = 1.0 if obs["type"] == "A" else 0.0
            is_B = 1.0 if obs["type"] == "B" else 0.0
            is_C = 1.0 if obs["type"] == "C" else 0.0
            
            return [1.0, float(obs["dist"]), is_A, is_B, is_C]
        
        obstacle_info = []
        
        # 각 레인의 첫 번째 장애물 (15차원)
        for lane in [0, 1, 2]:
            if len(lane_obstacles[lane]) >= 1:
                obstacle_info.extend(get_obstacle_features(lane_obstacles[lane][0]))
            else:
                obstacle_info.extend(get_obstacle_features(None))
        
        # 각 레인의 두 번째 장애물 (15차원)
        for lane in [0, 1, 2]:
            if len(lane_obstacles[lane]) >= 2:
                obstacle_info.extend(get_obstacle_features(lane_obstacles[lane][1]))
            else:
                obstacle_info.extend(get_obstacle_features(None))
        
        # 최종 상태 벡터 (3 + 15 + 15 = 33차원)
        state = np.array(player_info + obstacle_info, dtype=np.float32)
        return state

    def step(self, action):
        """
        한 step 실행 
        - action : 0(유지), 1(왼쪽), 2(오른쪽), 3(점프), 4(슬라이드)
        - Returns:
            state: 다음 상태 (33차원)
            reward: 보상
            done: 에피소드 종료 여부
            truncated: 시간 초과 여부 (사용 안 함)
            info: 메타데이터
        """
        self.time_step += 1

        # 액션 실행 : lane 이동
        if action == 1 and self.player_lane > 0:
            self.player_lane -= 1
        elif action == 2 and self.player_lane < 2:
            self.player_lane += 1

        # 액션 실행 : jump / slide
        if action == 3:
            self.jump_frame = self.max_jump_frames
            self.slide_frame = 0
        elif action == 4:
            self.slide_frame = self.max_slide_frames
            self.jump_frame = 0
        
        # jump / slide state 업데이트 
        if self.jump_frame > 0:
            self.jump_frame -= 1
            self.player_y_offset = -30
            self.player_height = 40
        elif self.slide_frame > 0:
            self.slide_frame -= 1
            self.player_y_offset = 0
            self.player_height = 20
        else:
            self.player_y_offset = 0
            self.player_height = 40

        # 속도 증가 
        self.speed = min(self.speed + 0.001, 3.0)

        # 새 장애물 생성 (시간에 따라 주기 감소)
        if self.time_step < 600:
            spawn_interval = 50
        elif self.time_step < 1200:
            spawn_interval = 40
        elif self.time_step < 1800:
            spawn_interval = 30
        else:
            spawn_interval = 20

        if self.time_step % spawn_interval == 0:
            self.spawn_obstacles()

        # 장애물 이동 (속도에 비례)
        for obs in self.obstacles:
            obs["dist"] -= 0.03 * self.speed
        
        # 충돌 감지
        collision = False
        death_reason = None
        death_reason_code = None
        avoided_list = []
        remaining = []

        for obs in self.obstacles:
            same_lane = (self.player_lane == obs["lane"])

            # 충돌 판정 범위 : dist <= 0.15 (플레이어 근처) 
            if obs["dist"] <= 0.15 and same_lane:
                # A 타입 : jump하지 않으면 충돌
                if obs["type"] == "A" and self.jump_frame <= 0:
                    collision = True
                    death_reason = (
                        f"Collision with A (Jump obstacle) at lane {obs['lane']} - Failed to jump"
                    )
                    death_reason_code = "A_no_jump"
                    self.death_stats["collision_A_no_jump"] += 1
                
                # B 타입 : slide하지 않으면 충돌
                elif obs["type"] == "B" and self.slide_frame <= 0:
                    collision = True
                    death_reason = (
                        f"Collision with B (Slide obstacle) at lane {obs['lane']} - Failed to slide"
                    )
                    death_reason_code = "B_no_slide"
                    self.death_stats["collision_B_no_slide"] += 1
                
                # C 타입 : 무조건 충돌 (lane 변경으로만 회피)
                elif obs["type"] == "C":
                    collision = True
                    death_reason = f"Collision with C (Unavoidable) at lane {obs['lane']}"
                    death_reason_code = "C_unavoidable"
                    self.death_stats["collision_C"] += 1

            # 장애물이 플레이어를 지나쳤으면 제거 (회피 성공)
            if obs["dist"] <= 0:
                avoided_list.append(obs)
            else:
                remaining.append(obs)

        # 남은 장애물만 유지 
        self.obstacles = remaining

        reward = 0.1 # 기본 생존 보상 

        # 장애물 회피 성공 보상 
        for obs in avoided_list:
            if obs["type"] == "A":
                reward += 2.0 if action == 3 else 1.0
            elif obs["type"] == "B":
                reward += 2.0 if action == 4 else 1.0
            elif obs["type"] == "C":
                reward += 2.0

        # 불필요한 lane 변경 페널티
        if action in [1, 2]:
            reward -= 0.01

        # 충돌 시 처리 
        if collision:
            reward -= 10.0 # 큰 페널티 
            self.death_stats["total_deaths"] += 1
            self.death_stats["survival_time"].append(self.time_step)
            self.current_death_reason = death_reason

        done = collision

        # 메타데이터
        info = {
            "death_reason": death_reason if collision else None,
            "death_reason_code": death_reason_code if collision else None,
            "survival_time": self.time_step if collision else 0,  # ← 여기 꼭 수정
            "pattern_group": self.current_pattern_group or "",
            "pattern_name": self.current_pattern_name or "",
            "speed": float(self.speed),
            "time_step": int(self.time_step),
        }

        return self._get_state(), reward, done, False, info

    def get_death_statistics(self):
        """
        사망 통계 반환
        - 어떤 상황에서 자주 죽는지 파악
        - 학습 진행 상황 모니터링 
        """
        total = self.death_stats["total_deaths"]
        if total == 0:
            return self.death_stats
        
        stats = self.death_stats.copy()
        stats["death_percentages"] = {
            "A_no_jump": (self.death_stats["collision_A_no_jump"] / total) * 100,
            "B_no_slide": (self.death_stats["collision_B_no_slide"] / total) * 100,
            "C_unavoidable": (self.death_stats["collision_C"] / total) * 100
        }
        
        # 생존 시간 통계 
        if len(self.death_stats["survival_time"]) > 0:
            stats["avg_survival_time"] = np.mean(self.death_stats["survival_time"])
            stats["max_survival_time"] = np.max(self.death_stats["survival_time"])
        
        return stats

    def render(self): # pygame을 이용한 시각화 
        if not pygame_available:
            print("[WARN] pygame is not installed")
            return

        self.screen.fill((30, 30, 30))
        lane_x = [80, 200, 320]

        for x in lane_x:
            pygame.draw.line(self.screen, (100, 100, 100), (x, 0), (x, 600), 2)

        y = self.base_y + self.player_y_offset
        player_rect = pygame.Rect(lane_x[self.player_lane] - 20, y, 40, self.player_height)
        
        if self.jump_frame > 0:
            player_color = (0, 255, 255)
        elif self.slide_frame > 0:
            player_color = (255, 255, 0)
        else:
            player_color = (0, 255, 0)
        
        pygame.draw.rect(self.screen, player_color, player_rect)

        for obs in self.obstacles:
            if obs["type"] == "A":
                color = (50, 150, 255)
                label = "J"
            elif obs["type"] == "B":
                color = (255, 255, 0)
                label = "S"
            else:
                color = (255, 50, 50)
                label = "X"

            obs_y = int(600 - obs["dist"] * 600)
            obs_rect = pygame.Rect(lane_x[obs["lane"]] - 20, obs_y, 40, 40)
            pygame.draw.rect(self.screen, color, obs_rect)
            
            label_surface = self.font.render(label, True, (0, 0, 0))
            label_rect = label_surface.get_rect(center=(lane_x[obs["lane"]], obs_y + 20))
            self.screen.blit(label_surface, label_rect)

        if self.current_death_reason:
            text_surface = self.font.render(self.current_death_reason, True, (255, 0, 0))
            self.screen.blit(text_surface, (10, 10))

        stats_text = f"Time: {self.time_step} | Speed: {self.speed:.2f} | Lane: {self.player_lane}"
        stats_surface = self.small_font.render(stats_text, True, (255, 255, 255))
        self.screen.blit(stats_surface, (10, 570))
        
        if self.jump_frame > 0:
            action_text = "JUMPING"
            action_color = (0, 255, 255)
        elif self.slide_frame > 0:
            action_text = "SLIDING"
            action_color = (255, 255, 0)
        else:
            action_text = "NORMAL"
            action_color = (255, 255, 255)
        
        action_surface = self.small_font.render(action_text, True, action_color)
        self.screen.blit(action_surface, (10, 550))

        pygame.display.flip()

