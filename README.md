# 城市末端医药物流配送优化（CVRPTW）

基于真实第三方物流（3PL）医药物流数据集，使用 Google OR-Tools 构建带容量和时间窗约束的车辆路径优化模型，并通过 Streamlit 部署为交互式 Web 应用。

## 🚀 在线 Demo
**https://vrp-medical-delivery-dubwqgxof6h27cnqbyexyv.streamlit.app**

## 📊 核心结果（Day 1，最可能场景）

| 指标 | 基线（最近邻） | OR-Tools | 改进 |
|------|----------------|----------|------|
| 总距离 | 386.7 km | **163.4 km** | **-57.7%** |
| 车辆数 | 4 | 4 | 持平 |
| 服务客户数 | 78/78 | 78/78 | 全部覆盖 |

![对比图](https://github.com/XJ12-web/vrp-medical-delivery/raw/main/routes_comparison.png)

## 📁 数据集
- 来源：[Delivering Data (Zenodo)](https://zenodo.org/records/15310106)
- 9 天真实配送任务，每天 78 个客户，含时间窗、重量、体积
- 三种交通场景：乐观 / 最可能 / 悲观

## 🛠 方法
- **基线 A**：纯最近邻（只看距离，不考虑时间窗）
- **基线 B**：最近邻 + 时间窗检查
- **优化模型**：OR-Tools CVRPTW，GUIDED_LOCAL_SEARCH 元启发式

## 🧰 技术栈
- Python 3.12
- Google OR-Tools（CVRPTW 求解）
- Streamlit（交互式仪表盘）
- NumPy（经典 MDS 从距离矩阵恢复坐标）
- Matplotlib（路线可视化）

## 📂 项目结构
