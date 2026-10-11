import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import MDS

from data_loader import load_day_data
from baselines import nearest_neighbor_with_tw
from solver import solve_cvrptw, extract_routes, compute_total_distance

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="医药物流配送优化", layout="wide")
st.title("🚚 城市末端医药物流配送优化")
st.markdown("**数据集**: Delivering Data (Zenodo) | **求解器**: Google OR-Tools")


@st.cache_data
def load_all():
    return load_day_data(r"D:\python\data", day_index=1)


@st.cache_data
def compute_coords(dist_matrix):
    D = np.array(dist_matrix, dtype=float)
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0)
    mds = MDS(n_components=2, metric="precomputed", random_state=42,
              n_init=4, max_iter=300, init="random", normalized_stress=False)
    return mds.fit_transform(D)


def plot_routes(coords, routes, title, color_map):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(coords[0, 0], coords[0, 1], c="black", s=200,
               marker="s", zorder=5, label="Depot")
    for v, route in enumerate(routes):
        if len(route) <= 1:
            continue
        color = color_map[v % len(color_map)]
        xs = [coords[item["node"], 0] for item in route]
        ys = [coords[item["node"], 1] for item in route]
        ax.plot(xs, ys, "-o", color=color, linewidth=2,
                markersize=5, label=f"车辆 {v}", alpha=0.8)
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# ===== 侧边栏 =====
st.sidebar.header("参数设置")
scenario = st.sidebar.selectbox(
    "交通场景",
    ["most_likely", "optimistic", "pessimistic"],
    format_func=lambda x: {"most_likely": "最可能", "optimistic": "乐观", "pessimistic": "悲观"}[x]
)
num_vehicles = st.sidebar.slider("车辆数上限", 3, 6, 4)
time_limit = st.sidebar.slider("求解时间（秒）", 10, 120, 30)
run_btn = st.sidebar.button("开始优化")

data = load_all()
time_matrix = data[f"time_matrix_{scenario}"]
coords = compute_coords(data["distance_matrix"])
color_map = ["red", "blue", "green", "orange", "purple", "brown"]

if run_btn:
    # 基线
    with st.spinner("运行基线..."):
        routes_b, dist_b, _, _ = nearest_neighbor_with_tw(
            data["distance_matrix"], time_matrix,
            data["time_windows"], data["demands"],
            data["service_times"], data["vehicle_capacity_weight"]
        )
        routes_b_plot = [[{"node": n, "arrival": 0} for n in r] for r in routes_b]

    # OR-Tools
    with st.spinner(f"OR-Tools 求解中（{time_limit}秒）..."):
        solution, routing, manager = solve_cvrptw(
            data["distance_matrix"], time_matrix,
            data["time_windows"], data["demands"],
            data["service_times"], data["vehicle_capacity_weight"],
            num_vehicles=num_vehicles, time_limit=time_limit
        )

    if solution:
        routes_o, on_time_o, late_o = extract_routes(
            solution, routing, manager, num_vehicles,
            data["time_windows"], data["demands"]
        )
        dist_o = compute_total_distance(routes_o, data["distance_matrix"])
        used = sum(1 for r in routes_o if len(r) > 1)

        # 指标卡片
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("基线距离", f"{dist_b:.1f} km")
        col2.metric("优化距离", f"{dist_o:.1f} km",
                    delta=f"-{(dist_b-dist_o):.1f} km")
        col3.metric("基线车辆", len(routes_b))
        col4.metric("优化车辆", used, delta=f"{used - len(routes_b)}")

        st.markdown(f"**距离减少**: {(dist_b-dist_o)/dist_b*100:.1f}%")

        # 两图并排
        c1, c2 = st.columns(2)
        with c1:
            fig1 = plot_routes(coords, routes_b_plot,
                               f"基线（最近邻）{dist_b:.1f} km / {len(routes_b)} 辆",
                               color_map)
            st.pyplot(fig1)
        with c2:
            fig2 = plot_routes(coords, routes_o,
                               f"OR-Tools {dist_o:.1f} km / {used} 辆",
                               color_map)
            st.pyplot(fig2)
    else:
        st.error(f"OR-Tools 在 {num_vehicles} 辆车下无可行解，请增大车辆数。")
else:
    st.info("👈 在左侧设置参数，点击「开始优化」运行。")