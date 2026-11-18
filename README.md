# ESP32 거미형 4족 로봇: MuJoCo 강화학습 환경

다리 4개, 다리당 2자유도(고관절 수평 회전 + 무릎 상하)인 거미형 로봇이 **목표 좌표까지 장애물과 단차를 넘어 도달**하도록
PPO(Stable-Baselines3)로 학습하는 시뮬레이션 환경입니다. 형상·구동·센서 값은 모두 `configs/default.yaml` 에서 바꿀 수 있고,
학습 중에 보고서용 지표를 최대한 많이 남깁니다.

## 빠른 시작

```bash
pip install -r requirements.txt
# GPU를 쓰려면 그다음 CUDA 빌드로 교체: pip install --force-reinstall --no-deps torch --index-url https://download.pytorch.org/whl/cu130
python -m pytest -q tests                        # 동작 확인 (약 5초)
python scripts/train.py --override configs/quick_test.yaml --tag quicktest   # 수 분짜리 시험 학습
python scripts/train.py --tag main               # 본 학습 (일어서기 단계부터 커리큘럼 자동 진행)
python scripts/plot_report.py runs/<실행폴더>      # 보고서용 그래프 PNG 일괄 생성
```

Windows 학교 PC(재부팅 시 초기화)는 [USB 휴대형 환경](#windows-학교-pc-usb-휴대형-환경)을 참고하세요.

## 구성

| 경로 | 내용 |
|---|---|
| `configs/default.yaml` | 로봇 형상·질량·모터·감속기·센서·통신·랜덤화·보상·커리큘럼·PPO 전체 설정 |
| `configs/school_pc.yaml`, `desktop.yaml`, `quick_test.yaml` | PC별 덮어쓰기 설정 |
| `spider_rl/model_builder.py` | 설정값 → MJCF(XML) 생성 (로봇 + 5×5 m 경기장 + 장애물 풀) |
| `spider_rl/terrain.py` | 출발점·목표(1~4 m) 샘플링, 상자벽·드럼통·의자·단차 배치 (재컴파일 없음) |
| `spider_rl/env.py` | Gymnasium 환경: 스테퍼 모델, 센서 모사, 보상, 종료, 에피소드 통계 |
| `spider_rl/callbacks.py` | 커리큘럼 자동 승급, 상세 로깅, 체크포인트, 시간 제한 |
| `scripts/train.py` | 학습 (이어받기 `--resume`, 단계 고정 `--stage k --fixed-stage`, 개별 설정 `--set a.b=값`) |
| `scripts/evaluate.py` | 정책 평가 + 보행 패턴·관절 궤적·토크-속도·궤적 그래프 + 영상 |
| `scripts/plot_report.py` | 학습 로그 → 보고서 그래프 30여 종, 여러 실행 비교 |
| `scripts/analyze_robot.py` | 설계 검토: 질량 예산, 지상고, 무릎 토크 여유, 일어서기 속도별 탈조 |
| `scripts/view_model.py` | MuJoCo 뷰어로 모델·정책 확인, MJCF 내보내기 |
| `windows/*.bat` | 휴대형 Python 설치, 학습, TensorBoard, 그래프 실행용 배치 파일 |
| `docs/robot_analysis/` | `analyze_robot.py` 결과 (토크 표, 그래프) |

## 로봇 모델

- **형상**: 몸체 20×20×7 cm, 고관절은 몸체 네 모서리에 대각선(45°) 방향, 링크1·링크2 각 8 cm.
  관절 0° = 링크가 모두 수평으로 뻗은 **엎드린 자세**(실물의 수동 원점과 동일). 무릎 + 방향이 발끝을 내려 몸을 들어 올림.
- **질량**: 프레임·전장 0.60 kg + 17HS48 모터 8×0.36 kg + 감속기 8×0.075 kg + 링크 → **총 4.44 kg**.
  고관절 모터는 몸체 안, 무릎 모터는 링크1 끝(무릎)에 질량을 둡니다.
- **액추에이터(NEMA17 17HS48 + 1:4 유성 감속기)**
  - 출력 정지 토크 = 0.44 N·m × 4 × 효율 0.8 = **1.41 N·m (14.4 kgf·cm)**, `motor_holding_torque` 로 변경
  - 토크-속도 곡선: 모터 600 rpm에서 정지 토크의 30%까지 선형 감소
  - 펌웨어 가감속: 최고 6 rad/s, 가속 40 rad/s² 사다리꼴 프로파일
  - **백래시**: 관절마다 에피소드별 1~2° 무작위 유격 (데드밴드)
  - **탈조**: 토크가 포화된 상태에서 추종 오차가 커지면 명령각과 실제각 사이에 영구 오차가 생김.
    정책은 명령각(개루프 추정값)만 관측하므로 실물과 같은 조건
  - 회전자 관성(6.8 g·cm²)×감속비² 를 관절 관성(armature)으로 반영
- **센서** (실물에서 얻을 수 있는 값만 관측)
  - 거리센서 4개: 수평 0°/±45° + 전방 45° 하향(단차 감지). 유효 200 cm, 빔 폭(±7.5°, 광선 3개 중 최솟값),
    쿨타임 66 ms, 지연 0~2스텝(+랜덤화), 노이즈, 측정 실패
  - IMU(MPU6050): 몸체 기준 중력 방향 + 자이로, 노이즈·에피소드별 바이어스
  - 목표: 출발점 기준 추정 위치(랜덤 워크 오차) → 목표까지 거리·방향
- **통신**: 행동 지연 0~1스텝(+랜덤화), 패킷 손실 시 이전 명령 유지
- **제어 30 Hz**, 물리 타임스텝 1/360 s (12 서브스텝)

관측 32차원 × 최근 3프레임 = 96차원, 행동 8차원(관절 목표각, -1~1).

### 설계 검토 결과 (`python scripts/analyze_robot.py`)

| 항목 | 값 |
|---|---|
| 총중량 | 4.44 kg |
| 무릎 필요 토크 (해석해, 4발 / 3발 지지 최대) | 0.87 / 1.16 N·m (8.9 / 11.8 kgf·cm) |
| 감속기 출력 정지 토크 | 1.41 N·m → 3발 지지 대비 **여유 1.21배** |
| 서기 몸체 높이 / 지상고 | 9.2 cm / **5.7 cm** |
| 일어서기 속도 | 엎드림→서기 명령을 0.5초 이내로 주면 탈조, 1초 이상이면 탈조 없음 (랜덤화 최대 시 약 2초 필요) |

> **주의: 10 cm 단차.** 링크 8 cm 기준 지상고가 약 5.7 cm라서 10 cm 단차는 몸체가 걸려 넘기 어렵습니다.
> `robot.link2_length`(예: 0.11) 를 늘리거나 `robot.hip_z` 를 음수로 두어 고관절을 낮추면 지상고가 늘어납니다.
> 형상을 바꾼 효과는 `python scripts/analyze_robot.py --set robot.link2_length=0.11` 로 바로 비교할 수 있습니다.

## 커리큘럼

모든 에피소드는 실물과 같이 **관절 0°로 엎드린 자세**에서 시작할 수 있고(단계별 비율 `init_prone_prob`),
몸체가 75° 이상 기울면 **뒤집힘 → 실패 종료**입니다.

| 단계 | 이름 | 내용 | 성공 조건 | 승급 기준 |
|---|---|---|---|---|
| 0 | stand_up | 엎드린 자세에서 일어서기 + 균형 (6 s) | 서기 자세 1초 유지 | 85% |
| 1 | walk_forward | 정면 2 m 목표 직진 (20 s) | 1 m 이상 전진 | 80% |
| 2 | goal_near | 무작위 목표 1~2 m (25 s) | 반경 15 cm 도달 | 80% |
| 3 | goal_far | 무작위 목표 1~4 m (40 s) | 도달 | 80% |
| 4 | randomized | 질량·마찰·토크·지연·노이즈 랜덤화 강화 | 도달 | 75% |
| 5 | obstacles | 상자벽·드럼통·의자 1~5개 | 도달 | 70% |
| 6 | steps | 단차 5→10 cm | 도달 | 60% |
| 7 | mixed | 장애물 + 단차 혼합 (최종) | 도달 | - |

- 최근 300 에피소드 성공률이 기준을 넘고 단계 진행도가 30% 이상이면 자동 승급
- 단계 내 진행도(0→1, 3000 에피소드)에 따라 랜덤화 수준·장애물 수·단차 높이가 점진적으로 올라감
- 이전 단계 에피소드를 15% 확률로 섞어 망각 방지
- 보상 가중치는 `reward:` 기본값 위에 단계별 `reward:` 로 덮어씀

## 로그와 보고서 그래프

실행 폴더 `runs/<날짜>_<PC이름>_<태그>/` 에 저장됩니다.

| 파일 | 내용 |
|---|---|
| `progress.csv`, `events.out.tfevents*` | SB3 로그 (TensorBoard: `tensorboard --logdir runs`) |
| `episodes.csv` | 에피소드마다 한 줄: 단계, 성공/뒤집힘/시간초과, 목표 거리, 단차·장애물, 지연, 에너지, 보상 항목별 합 |
| `curriculum_events.csv` | 단계 승급 시점 |
| `checkpoints/` | `step_*.zip`, `latest.zip`, `final.zip` + VecNormalize 통계 + 커리큘럼 상태 |
| `config.yaml`, `meta.json` | 사용한 설정과 PC 정보 (재현용) |

`python scripts/plot_report.py runs/<실행>` 이 만드는 그래프 (`runs/<실행>/figures/`):

- 성능: 성공률, 에피소드 보상(평균/최대/최소), 에피소드 길이, 종료 원인(도달/뒤집힘/시간 초과), 남은 거리, 목표 방향 이동 거리, 평균 속도, 일어서기 성공률·소요 시간, 에너지, 탈조 오차
- PPO 내부 지표: 엔트로피, 행동 표준편차, policy loss, value loss, approx KL, clip fraction, explained variance, 학습률 (묶음 + 개별)
- 보상 분해: 항목별 기여도 누적 영역 그래프, 항목별 소형 다중 그래프
- 커리큘럼: 단계 진행, 단계별 성공률, 단계별 결과 분포, 단계 × 목표거리 성공률 히트맵, 지형 난이도(단차 × 장애물 수) 히트맵
- Sim-to-Real: 통신 지연·센서 지연별 성공률, 일어서기 시간 분포
- 시스템: 학습 속도(FPS), CPU/RAM, 체크포인트 이력
- 실험 비교: `plot_report.py runs/A runs/B --labels 기본 엔트로피증가` 로 성공률·보상·단계·엔트로피·FPS 겹쳐보기

`python scripts/evaluate.py <체크포인트.zip> --stage k --episodes 50 --video` 는 로봇 거동 그래프
(발 접지 다이어그램, 관절 명령각 vs 실제각, 토크-속도 운용 영역, 토크·누적 에너지, 몸체 높이·기울기,
거리센서 판독값, 보상 누적, 장애물과 함께 그린 이동 궤적)와 영상, 성공률 요약을 만듭니다.
`--policy zero|random|stand` 로 비교용 기준선도 평가할 수 있습니다.

## Windows 학교 PC: USB 휴대형 환경

학교 PC는 재부팅하면 초기화되므로, Python과 패키지를 **USB 안의 저장소 폴더**에 설치해 들고 다닙니다.

1. 이 저장소를 USB에 복사 (예: `E:\ESP32-Spider-type-robot`)
2. 집이나 데스크탑에서 한 번 `windows\setup_portable.bat` 실행 → `python\` 폴더에 Python 3.11 + 패키지 설치
   (관리자 권한 불필요. GPU 버전 PyTorch가 필요하면 `setup_portable.bat -Cuda`, CUDA 13.0 빌드, 약 2.5 GB)
3. 학교 PC에서 `windows\quick_test.bat` 으로 확인 후 `windows\train_school_pc.bat` 실행
4. 7.5시간이 지나면 자동으로 `latest.zip` 저장 후 종료. 다음 세션은
   `windows\train_school_pc.bat runs\<이전 실행>\checkpoints\latest.zip` 으로 이어받기
5. 여러 대를 쓸 때는 PC마다 `--set ppo.ent_coef=0.005` 처럼 설정을 다르게 주고, 결과 폴더(`runs\`)를 모아
   `plot_report.py` 로 비교한 뒤 좋은 체크포인트를 다음 단계 학습에 이어받습니다.

참고: 소형 MLP 정책은 GPU보다 CPU가 빠른 경우가 많아 기본값은 CPU입니다. 병렬 환경 수(`ppo.n_envs`)를
CPU 코어 수에 맞추는 것이 속도에 가장 큰 영향을 줍니다. GPU 비교는 `--device cuda:0`.

## 주요 설정 바꾸기

```bash
# 모터 토크를 0.55 N·m로, 링크2를 11 cm로
python scripts/train.py --set robot.motor_holding_torque=0.55 --set robot.link2_length=0.11
# 장애물 단계만 집중 학습
python scripts/train.py --resume runs/A/checkpoints/latest.zip --stage 5 --fixed-stage
```

설정 파일 하나를 새로 만들어 `--override my.yaml` 로 넘기면 바뀐 값만 덮어씁니다.
