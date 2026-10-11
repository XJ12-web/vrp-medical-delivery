from data_loader import load_day_data
from baselines import nearest_neighbor_with_tw
from solver import solve_cvrptw, extract_routes, compute_total_distance


def main():
    data = load_day_data(r"D:\python\data", day_index=1)
    total_customer = data["n_nodes"] - 1

    print("\n" + "="*60)
    print("对比实验：基线 B vs OR-Tools CVRPTW（悲观场景）")
    print("="*60)

    # ----- 基线 B（悲观场景）-----
    routes_b, dist_b, late_b, on_time_b = nearest_neighbor_with_tw(
        data["distance_matrix"],
        data["time_matrix_pessimistic"],
        data["time_windows"],
        data["demands"],
        data["service_times"],
        data["vehicle_capacity_weight"]
    )
    rate_b = on_time_b / total_customer

    # 统计基线实际访问的客户数
    visited_b = set()
    for route in routes_b:
        for node in route:
            if node != 0:
                visited_b.add(node)

    print(f"\n基线B（最近邻+时间窗）— 悲观场景:")
    print(f"  总距离 = {dist_b:.2f} km")
    print(f"  使用车辆 = {len(routes_b)}")
    print(f"  实际访问客户数 = {len(visited_b)} / {total_customer}")
    print(f"  准时率 = {rate_b:.2%}")

    # ----- OR-Tools 尝试不同车辆数 -----
    for n_vehicles in [4, 5, 6]:
        print(f"\n尝试 OR-Tools，可用车辆上限 = {n_vehicles} ...")
        solution, routing, manager = solve_cvrptw(
            data["distance_matrix"],
            data["time_matrix_pessimistic"],
            data["time_windows"],
            data["demands"],
            data["service_times"],
            data["vehicle_capacity_weight"],
            num_vehicles=n_vehicles,
            time_limit=60
        )

        if solution:
            routes_o, on_time_o, late_o = extract_routes(
                solution, routing, manager, n_vehicles,
                data["time_windows"], data["demands"]
            )
            dist_o = compute_total_distance(routes_o, data["distance_matrix"])
            rate_o = on_time_o / total_customer
            vehicles_used = sum(1 for r in routes_o if len(r) > 1)

            # ===== 核心验证：检查所有客户是否都被访问 =====
            visited_nodes = set()
            for route in routes_o:
                for item in route:
                    if item["node"] != 0:
                        visited_nodes.add(item["node"])

            print(f"  OR-Tools 结果:")
            print(f"    总距离 = {dist_o:.2f} km")
            print(f"    使用车辆 = {vehicles_used}")
            print(f"    实际访问客户数 = {len(visited_nodes)} / {total_customer}")
            print(f"    准时率 = {rate_o:.2%}")

            if len(visited_nodes) < total_customer:
                missing = set(range(1, total_customer + 1)) - visited_nodes
                print(f"    ⚠️ 缺失客户数 = {len(missing)}，前 20 个: {sorted(missing)[:20]}")
                print(f"    ⚠️ 距离减少 {dist_b - dist_o:.2f} km 不可信（未服务全部客户）")
            else:
                print(f"    ✅ 所有客户均被访问，结果可信")
                print(f"    相比基线距离减少 = {dist_b - dist_o:.2f} km "
                      f"({(dist_b - dist_o) / dist_b * 100:.1f}%)")
        else:
            print(f"  无可行解（{n_vehicles} 辆车不够满足所有约束）")

    print("\n" + "="*60)


if __name__ == "__main__":
    main()