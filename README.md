# Soccer_Lab

基于 Isaac Lab 的 Unitree G1 足球实验项目。

## 1. 安装 Isaac Lab

请先按照 Isaac Lab 官方安装文档完成 Isaac Sim 与 Isaac Lab 环境安装：

- [Isaac Lab 官方安装文档](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)

建议先进入你已经准备好的 `env_isaaclab` 环境，再执行下面的项目安装命令。

## 2. 安装本项目

在仓库根目录执行：

```bash
python -m pip install -e source/Soccer_Lab
```

## 3. 启动仿真

### 4v4 G1 足球环境

```bash
python scripts/zero_agent.py \
  --task SoccerLab-G1-Soccer-4v4-Direct-v0 \
  --enable_cameras
```

### 单机器人 G1 足球环境

```bash
python scripts/zero_agent.py \
  --task SoccerLab-G1-Soccer-Single-Direct-v0 \
  --enable_cameras
```
