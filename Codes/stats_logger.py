# stats_logger.py
# train process statics 기록 및 분석

from dataclasses import dataclass
from collections import Counter
from typing import List, Dict, Any, Optional
import csv


@dataclass
class EpisodeRecord: # 한 episode의 실행 결과 저장하는 data class
    episode: int # 에피 번호
    total_reward: float # 해당 에피의 총 보상 
    length: int # 에피 길이 (몇 스텝동안 생존)
    death_reason: Optional[str] # 종료 사유 (텍스트)
    death_reason_code: Optional[str] # 종료 사유 코드 (분류용)
    pattern_group: Optional[str] # 패턴 그룹
    pattern_name: Optional[str] # 구체적 패턴 이름
    speed: float # 게임 속도 
    time_step: int # 실제 시간 스텝
    epsilon: Optional[float] = None   # 탐험률 (ε-greedy)


class StatsLogger: # 학습과정의 statics를 수집하고 저장하는 메인 class
    def __init__(self, csv_path: str = "train_stats.csv"):
        self.records: List[EpisodeRecord] = [] # 모든 에피 기록 저장
        self.csv_path = csv_path

    def log_episode(
        self,
        episode: int,
        total_reward: float,
        length: int,
        info: Dict[str, Any], # 환경에서 반환한 추가 정보 (종료 사유, 페턴 정보 등)
        epsilon: Optional[float] = None, 
    ): # 한 에피가 끝날 때마다 호출해 결과 기록 
        rec = EpisodeRecord(
            episode=episode,
            total_reward=total_reward,
            length=length,
            death_reason=info.get("death_reason"),
            death_reason_code=info.get("death_reason_code"),
            pattern_group=info.get("pattern_group"),
            pattern_name=info.get("pattern_name"),
            speed=float(info.get("speed", 0.0)), # 기본값 0.0
            time_step=int(info.get("time_step", length)), # 없으면 length 사용
            epsilon=epsilon,
        )
        self.records.append(rec)

    def summary_last(self, n: int = 100) -> Dict[str, Any]:
        # 최근 n개 에피소드 기준 요약 통계 (학습 잘 되는지 확인용)
        if not self.records:
            return {}

        recent = self.records[-n:]
        avg_return = sum(r.total_reward for r in recent) / len(recent)
        avg_len = sum(r.length for r in recent) / len(recent)

        death_counter = Counter(
            r.death_reason_code for r in recent if r.death_reason_code is not None
        )
        pattern_counter = Counter(
            r.pattern_name for r in recent if r.pattern_name is not None
        )

        return {
            "episodes": len(recent), # 에피소드 개수
            "avg_return": avg_return, # 평균 보상 계산
            "avg_length": avg_len, # 평균 길이 계산 
            "death_counts": dict(death_counter), # 사망 원인별 카운트
            "pattern_counts": dict(pattern_counter), # 패턴별 카운트 
        }

    def save_csv(self): # csv 파일로 저장
        if not self.records:
            return

        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "episode",
                "total_reward",
                "length",
                "death_reason",
                "death_reason_code",
                "pattern_group",
                "pattern_name",
                "speed",
                "time_step",
                "epsilon",           
            ])
            for r in self.records:
                writer.writerow([
                    r.episode,
                    r.total_reward,
                    r.length,
                    r.death_reason,
                    r.death_reason_code,
                    r.pattern_group,
                    r.pattern_name,
                    r.speed,
                    r.time_step,
                    r.epsilon,      
                ])
