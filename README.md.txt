# 城市末端医药物流配送优化（CVRPTW）

基于真实第三方物流（3PL）医药物流数据集，使用 Google OR-Tools 构建带容量和时间窗约束的车辆路径优化模型。

## 数据集
- 来源：Delivering Data (Zenodo)
- 9 天真实配送任务，每天 78 个客户，含时间窗、重量、体积
- 三种交通场景：乐观 / 最可能 / 悲观

## 方法
- **基线 A**：纯最近邻（只看距离，不考虑时间窗）
- **基线 B**：最近邻 + 时间窗检查
- **优化模型**：OR-Tools CVRPTW，GUIDED_LOCAL_SEARCH 元启发式

## 核心结果（Day 1，悲观场景）

| 指标 | 基线 B | OR-Tools | 改进 |
|------|--------|----------|------|
| 总距离 | 427.8 km | 168.1 km | **-60.7%** |
| 车辆数 | 5 | 4 | **-1 辆** |
| 服务客户数 | 78/78 | 78/78 | 全部覆盖 |

## 技术栈
- Python 3.10
- Google OR-Tools
- Streamlit
- scikit-learn MDS
- matplotlib

## 运行

pip install -r requirements.txt
streamlit run app.py

## 在线 Demo
（部署后填写）
