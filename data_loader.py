import pandas as pd
import numpy as np
import os

print(">>> data_loader.py 版本 v3.0（含 read_matrix 清理函数）<<<")


def read_matrix(file_path, n_node):
    """读取矩阵 Excel，自动清理表头、空行、空列、非数值内容。"""
    df = pd.read_excel(file_path, header=None)
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna(how='all', axis=0).dropna(how='all', axis=1)
    df = df.reset_index(drop=True)
    df = df.fillna(0)
    mat = df.values.astype(float)
    if mat.shape[0] < n_node or mat.shape[1] < n_node:
        raise ValueError(f"矩阵 {file_path} 清理后维度 {mat.shape} 小于节点数 {n_node}")
    mat = mat[:n_node, :n_node]
    return mat


def load_day_data(day_folder, day_index, vehicle_capacity_weight=600.0,
                  vehicle_capacity_volume=3.0, num_vehicles=6):
    # ===== 1. 读取订单 =====
    orders_path = os.path.join(day_folder, "orders", "orders.xlsx")
    if not os.path.exists(orders_path):
        raise FileNotFoundError(f"找不到订单文件: {orders_path}")

    sheet_name = f"Day {day_index}"
    orders = pd.read_excel(orders_path, sheet_name=sheet_name)

    demands = orders["WEIGHT"].values.astype(float).tolist()
    volumes = orders["VOLUME"].values.astype(float).tolist()
    service_times = orders["SERVICE_TIME"].values.astype(int).tolist()
    time_windows = list(zip(
        orders["EAT"].values.astype(int),
        orders["LAT"].values.astype(int)
    ))

    # 插入 Depot
    demands.insert(0, 0.0)
    volumes.insert(0, 0.0)
    service_times.insert(0, 0)
    time_windows.insert(0, (0, 10000))

    n_node = len(demands)
    print(f"订单数据：{n_node} 个节点（1个Depot + {n_node-1}个客户）")

    # ===== 2. 读取矩阵 =====
    matrix_base = os.path.join(day_folder, "time_and_distance_matrices")
    matrix_folder = os.path.join(matrix_base, f"day_{day_index}")

    if not os.path.exists(matrix_folder):
        raise FileNotFoundError(f"找不到矩阵子文件夹: {matrix_folder}")

    all_files = [f for f in os.listdir(matrix_folder) if f.endswith(".xlsx")]
    print(f"\n[day_{day_index}] 文件夹内共有 {len(all_files)} 个文件：")
    for f in all_files:
        print(f"  - {f}")

    def find_file(keyword_list):
        for f in all_files:
            if all(kw.lower() in f.lower() for kw in keyword_list):
                return os.path.join(matrix_folder, f)
        raise FileNotFoundError(f"找不到包含关键词 {keyword_list} 的文件")

    dist_file = find_file(["distance"])
    time_most_file = find_file(["time", "most"])
    time_pess_file = find_file(["time", "pess"])
    time_opt_file = find_file(["time", "opt"])

    print(f"\n自动匹配到的文件：")
    print(f"  距离矩阵: {os.path.basename(dist_file)}")
    print(f"  时间(most likely): {os.path.basename(time_most_file)}")

    # ===== 3. 用 read_matrix 清理并裁剪 =====
    print("\n开始读取并清理矩阵...")
    dist_matrix = read_matrix(dist_file, n_node)
    time_most = read_matrix(time_most_file, n_node)
    time_pess = read_matrix(time_pess_file, n_node)
    time_opt = read_matrix(time_opt_file, n_node)

    # ===== 4. 校验 =====
    assert dist_matrix.shape == (n_node, n_node)
    assert time_most.shape == (n_node, n_node)

    print(f"\n距离矩阵范围：min={dist_matrix.min():.2f}, max={dist_matrix.max():.2f}")
    print(f"时间矩阵(most likely)范围：min={time_most.min():.2f}, max={time_most.max():.2f}")
    print("请根据以上范围判断单位：距离是km还是m？时间是分钟还是秒？")

    # ===== 5. 组装数据 =====
    data = {
        "n_nodes": n_node,
        "demands": demands,
        "volumes": volumes,
        "time_windows": time_windows,
        "service_times": service_times,
        "distance_matrix": dist_matrix,
        "time_matrix_most_likely": time_most,
        "time_matrix_pessimistic": time_pess,
        "time_matrix_optimistic": time_opt,
        "vehicle_capacity_weight": vehicle_capacity_weight,
        "vehicle_capacity_volume": vehicle_capacity_volume,
        "depot": 0,
        "num_vehicles": num_vehicles,
    }

    print(f"\n成功加载 Day {day_index}: {data['n_nodes']} 个节点（含Depot）")
    return data


# ==================== 测试 ====================
if __name__ == "__main__":
    data = load_day_data(r"D:\python\data", day_index=1)
    print("\n--- 数据预览 ---")
    print("节点数:", data["n_nodes"])
    print("前5个节点重量:", data["demands"][:5])
    print("距离矩阵前3行前3列:\n", data["distance_matrix"][:3, :3])