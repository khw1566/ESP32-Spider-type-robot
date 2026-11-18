"""설정값으로부터 거미형 4족 로봇 + 경기장 MJCF(XML)를 생성한다.

좌표계
- 몸체: x 전방, y 좌측, z 위
- 다리: 몸체 모서리에 대각선(45°) 방향으로 장착. 다리 프레임의 x축이 바깥쪽.
- 고관절(hip): z축 회전(수평 회전)
- 무릎(knee): 다리 프레임 y축 회전. + 방향이 발끝을 아래로 → 몸을 들어올림
- 관절 0° 자세: 링크1, 링크2 모두 수평으로 뻗은 '엎드린' 자세 (실물 수동 원점과 동일)

충돌 그룹
- 로봇 geom: contype=2, conaffinity=1, group=1  (로봇끼리는 충돌하지 않음)
- 월드 geom: contype=1, conaffinity=2, group=0  (거리센서 광선은 group 0만 검출)
"""
from __future__ import annotations

import math

LEG_NAMES = ["FL", "FR", "RL", "RR"]
LEG_YAW_DEG = {"FL": 45.0, "FR": -45.0, "RL": 135.0, "RR": -135.0}
JOINT_NAMES = [f"{leg}_{j}" for leg in LEG_NAMES for j in ("hip", "knee")]

# 장애물 풀 크기 (리셋 때 위치/크기만 바꾸어 재컴파일 없이 사용)
N_BOX_WALL = 5
N_DRUM = 3
N_CHAIR = 2
N_STEP = 2
PARK_POS = (40.0, 40.0)  # 사용하지 않는 장애물을 치워두는 위치

ROBOT_GEOM = 'contype="2" conaffinity="1" group="1"'
WORLD_GEOM = 'contype="1" conaffinity="2" group="0"'


def _f(*vals) -> str:
    return " ".join(f"{v:.6g}" for v in vals)


def output_torque(r: dict) -> float:
    """감속기 출력 정지 토크 (N·m)."""
    return r["motor_holding_torque"] * r["gear_ratio"] * r["gear_efficiency"]


def robot_mass_budget(cfg: dict) -> dict:
    r = cfg["robot"]
    act = r["motor_mass"] + r["gearbox_mass"]
    foot_mass = 0.01
    parts = {
        "body_frame_electronics": r["body_mass"],
        "hip_actuators(4)": 4 * act,
        "knee_actuators(4)": 4 * act,
        "link1(4)": 4 * r["link1_mass"],
        "link2+foot(4)": 4 * (r["link2_mass"] + foot_mass),
    }
    parts["total"] = sum(parts.values())
    return parts


def _robot_xml(cfg: dict) -> tuple[str, str]:
    r = cfg["robot"]
    sx, sy, sz = r["body_size"]
    L1, L2 = r["link1_length"], r["link2_length"]
    lr, fr = r["link_radius"], r["foot_radius"]
    act_mass = r["motor_mass"] + r["gearbox_mass"]
    armature = r["rotor_inertia"] * r["gear_ratio"] ** 2
    tau = output_torque(r)
    hip_lo, hip_hi = r["hip_range"]
    kn_lo, kn_hi = r["knee_range"]
    s = cfg["sensors"]
    com = r.get("com_offset", [0, 0, 0])
    z0 = sz / 2 + 0.002  # 엎드린 자세에서 몸체 중심 높이

    legs = []
    for leg in LEG_NAMES:
        yaw = LEG_YAW_DEG[leg]
        hx = math.copysign(sx / 2 - r["hip_inset"], math.cos(math.radians(yaw)))
        hy = math.copysign(sy / 2 - r["hip_inset"], math.sin(math.radians(yaw)))
        legs.append(f"""
      <!-- 고관절 모터(수직축)는 몸체 내부: 질량만 반영 -->
      <geom name="{leg}_hipmotor" type="cylinder" pos="{_f(hx, hy, r['hip_z'])}" size="0.021 0.03"
            mass="{act_mass:.4f}" contype="0" conaffinity="0" group="1" rgba="0.2 0.2 0.22 1"/>
      <body name="{leg}_link1" pos="{_f(hx, hy, r['hip_z'])}" euler="0 0 {yaw}">
        <joint name="{leg}_hip" type="hinge" axis="0 0 1" range="{hip_lo} {hip_hi}"
               armature="{armature:.3e}" damping="{r['joint_damping']}" frictionloss="{r['joint_frictionloss']}"/>
        <geom name="{leg}_l1" type="capsule" fromto="0 0 0 {_f(L1)} 0 0" size="{lr}" mass="{r['link1_mass']}"
              {ROBOT_GEOM} rgba="0.95 0.55 0.15 1"/>
        <geom name="{leg}_kneemotor" type="box" pos="{_f(L1, 0, 0)}" size="0.021 0.032 0.021" mass="{act_mass:.4f}"
              {ROBOT_GEOM} rgba="0.2 0.2 0.22 1"/>
        <body name="{leg}_link2" pos="{_f(L1, 0, 0)}">
          <joint name="{leg}_knee" type="hinge" axis="0 1 0" range="{kn_lo} {kn_hi}"
                 armature="{armature:.3e}" damping="{r['joint_damping']}" frictionloss="{r['joint_frictionloss']}"/>
          <geom name="{leg}_l2" type="capsule" fromto="0 0 0 {_f(L2 - fr)} 0 0" size="{lr * 0.8:.4f}" mass="{r['link2_mass']}"
                {ROBOT_GEOM} rgba="0.95 0.55 0.15 1"/>
          <geom name="{leg}_foot" type="sphere" pos="{_f(L2, 0, 0)}" size="{fr}" mass="0.01"
                friction="{_f(*r['foot_friction'])}" condim="4" {ROBOT_GEOM} rgba="0.1 0.1 0.1 1"/>
          <site name="{leg}_foot_site" pos="{_f(L2, 0, 0)}" size="0.005"/>
        </body>
      </body>""")

    body = f"""
    <body name="torso" pos="0 0 {z0:.4f}">
      <freejoint name="root"/>
      <camera name="track" mode="trackcom" pos="0.0 -1.1 0.75" xyaxes="1 0 0 0 0.56 0.83"/>
      <camera name="side" mode="trackcom" pos="0.9 -0.9 0.35" xyaxes="0.707 0.707 0 -0.2 0.2 0.96"/>
      <geom name="torso_box" type="box" size="{_f(sx / 2, sy / 2, sz / 2)}" pos="{_f(*com)}" mass="{r['body_mass']}"
            {ROBOT_GEOM} rgba="0.25 0.45 0.85 1"/>
      <site name="imu" pos="0 0 0" size="0.01" rgba="1 0 0 1"/>
      <site name="dist_front" pos="{_f(sx / 2, 0, s['sensor_height'])}" size="0.008" rgba="1 1 0 1"/>
      <site name="dist_down" pos="{_f(sx / 2, 0, -sz / 4)}" size="0.008" rgba="1 0 1 1"/>
      {''.join(legs)}
    </body>"""

    acts = []
    for jn in JOINT_NAMES:
        acts.append(
            f'<position name="act_{jn}" joint="{jn}" kp="{r["position_kp"]}" kv="{r["position_kv"]}" '
            f'forcelimited="true" forcerange="{-tau:.4f} {tau:.4f}" ctrllimited="false"/>')
    return body, "\n    ".join(acts)


def _obstacle_pool_xml(cfg: dict) -> str:
    a = cfg["arena"]
    px, py = PARK_POS
    out = []
    for i in range(N_BOX_WALL):
        out.append(f'<body name="wall{i}" pos="{px + i} {py} 0"><geom name="wall{i}_g" type="box" size="0.3 0.1 0.2" '
                   f'pos="0 0 0.2" {WORLD_GEOM} rgba="0.72 0.55 0.35 1"/></body>')
    for i in range(N_DRUM):
        out.append(f'<body name="drum{i}" pos="{px + i} {py + 3} 0"><geom name="drum{i}_g" type="cylinder" size="0.2 0.3" '
                   f'pos="0 0 0.3" {WORLD_GEOM} rgba="0.2 0.4 0.75 1"/></body>')
    c = a["chair"]
    sw, sd = c["seat"]
    sh, lr = c["seat_height"], c["leg_radius"]
    for i in range(N_CHAIR):
        legs = "".join(
            f'<geom type="cylinder" size="{lr} {sh / 2:.4f}" pos="{_f(dx * (sw / 2 - 0.03), dy * (sd / 2 - 0.03), sh / 2)}" '
            f'{WORLD_GEOM} rgba="0.3 0.3 0.3 1"/>' for dx in (-1, 1) for dy in (-1, 1))
        out.append(
            f'<body name="chair{i}" pos="{px + i} {py + 6} 0">'
            f'<geom name="chair{i}_seat" type="box" size="{_f(sw / 2, sd / 2, 0.02)}" pos="0 0 {sh:.3f}" {WORLD_GEOM} rgba="0.55 0.35 0.2 1"/>'
            f'<geom type="box" size="{_f(0.02, sd / 2, 0.2)}" pos="{_f(-sw / 2, 0, sh + 0.2)}" {WORLD_GEOM} rgba="0.55 0.35 0.2 1"/>'
            f'{legs}</body>')
    for i in range(N_STEP):
        out.append(f'<body name="step{i}" pos="{px + i} {py + 9} 0"><geom name="step{i}_g" type="box" size="0.5 0.3 0.025" '
                   f'pos="0 0 0.025" {WORLD_GEOM} rgba="0.6 0.6 0.55 1"/></body>')
    return "\n    ".join(out)


def build_mjcf(cfg: dict) -> str:
    sim = cfg["sim"]
    dt = 1.0 / (sim["control_hz"] * sim["substeps"])
    half = sim["arena_size"] / 2
    wh = sim["wall_height"]
    robot, actuators = _robot_xml(cfg)
    walls = "".join(
        f'<geom name="arena_wall{i}" type="box" size="{_f(*sz)}" pos="{_f(*p)}" {WORLD_GEOM} rgba="0.8 0.8 0.8 0.6"/>'
        for i, (sz, p) in enumerate([
            ((half + 0.05, 0.05, wh / 2), (0, half + 0.05, wh / 2)),
            ((half + 0.05, 0.05, wh / 2), (0, -half - 0.05, wh / 2)),
            ((0.05, half + 0.05, wh / 2), (half + 0.05, 0, wh / 2)),
            ((0.05, half + 0.05, wh / 2), (-half - 0.05, 0, wh / 2)),
        ]))
    return f"""<mujoco model="esp32_spider">
  <compiler angle="degree" autolimits="true"/>
  <option timestep="{dt:.6f}" integrator="{sim['integrator']}" gravity="0 0 -9.81"/>
  <size memory="20M"/>
  <visual>
    <global offwidth="1280" offheight="720"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.45 0.45 0.45" specular="0.1 0.1 0.1"/>
    <quality shadowsize="2048"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.82 0.82 0.80" rgb2="0.70 0.70 0.68" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="10 10" reflectance="0.05"/>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.6 0.75 0.9" rgb2="0.1 0.1 0.15" width="256" height="256"/>
  </asset>
  <worldbody>
    <light pos="1.5 -1.5 4" dir="-0.3 0.3 -1" directional="true" diffuse="0.55 0.55 0.55" castshadow="true"/>
    <camera name="top" pos="0 0 7.5" xyaxes="1 0 0 0 1 0"/>
    <geom name="floor" type="plane" size="{half + 1} {half + 1} 0.1" material="grid"
          friction="{_f(*sim['floor_friction'])}" {WORLD_GEOM}/>
    {walls}
    <site name="goal" type="cylinder" pos="1 0 0.002" size="{cfg['arena']['goal_radius']} 0.002" rgba="0.1 0.9 0.2 0.5"/>
    <site name="goal_pole" type="capsule" fromto="1 0 0 1 0 0.4" size="0.01" rgba="0.1 0.9 0.2 0.8"/>
    {_obstacle_pool_xml(cfg)}
    {robot}
  </worldbody>
  <actuator>
    {actuators}
  </actuator>
  <sensor>
    <gyro name="imu_gyro" site="imu"/>
    <accelerometer name="imu_acc" site="imu"/>
    <framequat name="imu_quat" objtype="site" objname="imu"/>
  </sensor>
</mujoco>
"""
