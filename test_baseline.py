from data_loader import load_day_data
from baselines import nearest_neighbor_pure, nearest_neighbor_with_tw


def main():
    data = load_day_data(r"D:\python\data", day_index=1)

    print("\n" + "="*60)
    print("基线算法结果")
    print("="*60)

    # 基线A
    routes_a, dist_a = nearest_neighbor_pure(
        data["distance_matrix"],
        data["demands"],
        data["vehicle_capacity_weight"]
    )
    print(f"\n基线A（纯最近邻）:")
    print(f"  总距离 = {dist_a:.2f} km")
    print(f"  使用车辆 = {len(routes_a)}")

    total_customer = len(data["demands"]) - 1

    # 基线B，三种场景
    scenarios = {
        "乐观 (optimistic)": data["time_matrix_optimistic"],
        "最可能 (most likely)": data["time_matrix_most_likely"],
        "悲观 (pessimistic)": data["time_matrix_pessimistic"],
    }

    for name, time_matrix in scenarios.items():
        routes_b, dist_b, late_b, on_time_b = nearest_neighbor_with_tw(
            data["distance_matrix"],
            time_matrix,
            data["time_windows"],
            data["demands"],
            data["service_times"],
            data["vehicle_capacity_weight"]
        )
        on_time_rate = on_time_b / total_customer
        print(f"\n基线B — {name}:")
        print(f"  总距离 = {dist_b:.2f} km")
        print(f"  使用车辆 = {len(routes_b)}")
        print(f"  准时率 = {on_time_rate:.2%}")


if __name__ == "__main__":
    main()