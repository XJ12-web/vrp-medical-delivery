import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS

from data_loader import load_day_data
from baselines import nearest_neighbor_with_tw
from solver import solve_cvrptw, extract_routes, compute_total_distance

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def recover_coordinates(distance_matrix):
    """用 MDS 从距离矩阵反推 2D 坐标（含对称化处理）"""
    D = np.array(distance_matrix, dtype=float)
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0)

    mds = MDS(
        n_components=2,
        metric="precomputed",
        random_state=42,
        n_init=4,
        max_iter=300,
        init="random",
        normalized_stress=False,
    )
    coords = mds.fit_transform(D)
    return coords


def plot_routes(coords, routes, title, color_map):
    """在 2D 平面上画出配送路线"""
    fig, ax = plt.subplots(figsize=(12, 10))

    # Depot
    ax.scatter(coords[0, 0], coords[0, 1], c="black", s=200,
               marker="s", zorder=5, label="Depot")

    # 每条路线一种颜色
    for v, route in enumerate(routes):
        if len(route) <= 1:
            continue
        color = color_map[v % len(color_map)]
        xs = [coords[item["node"], 0] for item in route]
        ys = [coords[item["node"], 1] for item in route]
        ax.plot(xs, ys, "-o", color=color, linewidth=2, markersize=6,
                label=f"车辆 {v}", alpha=0.8)
        for item in route:
            node = item["node"]
            if node != 0:
                ax.annotate(f"N{node}", (coords[node, 0], coords[node, 1]),
                            fontsize=6, alpha=0.6)

    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def main():
    data = load_day_data(r"D:\python\data", day_index=1)

    print("\n正在用 MDS 从距离矩阵恢复坐标...")
    coords = recover_coordinates(data["distance_matrix"])
    print("坐标恢复完成。")

    color_map = ["red", "blue", "green", "orange", "purple", "brown"]

    # --- 基线 B ---
    routes_b, dist_b, late_b, on_time_b = nearest_neighbor_with_tw(
        data["distance_matrix"],
        data["time_matrix_pessimistic"],
        data["time_windows"],
        data["demands"],
        data["service_times"],
        data["vehicle_capacity_weight"]
    )

    routes_b_plot = []
    for r in routes_b:
        routes_b_plot.append([{"node": node, "arrival": 0} for node in r])

    fig1 = plot_routes(coords, routes_b_plot,
                       f"基线 B（最近邻+时间窗）— 总距离 {dist_b:.1f} km，{len(routes_b)} 辆车",
                       color_map)
    fig1.savefig("routes_baseline.png", dpi=150)
    print("已保存: routes_baseline.png")

    # --- OR-Tools ---
    solution, routing, manager = solve_cvrptw(
        data["distance_matrix"],
        data["time_matrix_pessimistic"],
        data["time_windows"],
        data["demands"],
        data["service_times"],
        data["vehicle_capacity_weight"],
        num_vehicles=4,
        time_limit=60
    )

    if solution:
        routes_o, on_time_o, late_o = extract_routes(
            solution, routing, manager, 4,
            data["time_windows"], data["demands"]
        )
        dist_o = compute_total_distance(routes_o, data["distance_matrix"])
        used = sum(1 for r in routes_o if len(r) > 1)

        fig2 = plot_routes(coords, routes_o,
                           f"OR-Tools CVRPTW — 总距离 {dist_o:.1f} km，{used} 辆车",
                           color_map)
        fig2.savefig("routes_ortools.png", dpi=150)
        print("已保存: routes_ortools.png")

        print(f"\n对比总结：")
        print(f"  基线B:     {dist_b:>8.1f} km, {len(routes_b)} 辆车")
        print(f"  OR-Tools:  {dist_o:>8.1f} km, {used} 辆车")
        print(f"  距离减少:  {dist_b - dist_o:.1f} km ({(dist_b-dist_o)/dist_b*100:.1f}%)")

    plt.show()


if __name__ == "__main__":
    main()