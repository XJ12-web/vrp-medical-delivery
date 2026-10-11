def nearest_neighbor_pure(distance_matrix, demands, vehicle_capacity_weight, depot=0):
    """
    基线A：纯最近邻，完全不考虑时间窗，只看距离和重量容量。

    返回: routes（路线列表）, total_distance（总距离）
    """
    n = len(distance_matrix)
    unvisited = set(range(1, n))
    routes = []
    current_route = [depot]
    current_load = 0
    current_node = depot
    total_distance = 0
    max_iter = n * n  # 防止死循环的保护机制
    iter_count = 0

    while unvisited:
        iter_count += 1
        if iter_count > max_iter:
            print("警告：纯最近邻陷入死循环，强制退出！")
            break

        # 核心修复：检查是否存在单节点需求大于车辆载重的情况
        max_single_demand = max([demands[x] for x in unvisited])
        if max_single_demand > vehicle_capacity_weight:
            print(f"警告：存在客户需求{max_single_demand} > 车辆载重{vehicle_capacity_weight}，该实例不可行！")
            break

        nearest = min(unvisited, key=lambda x: distance_matrix[current_node][x])

        # 如果装不下，就返回depot，派新车
        if current_load + demands[nearest] > vehicle_capacity_weight:
            total_distance += distance_matrix[current_node][depot]
            routes.append(current_route + [depot])
            current_route = [depot]
            current_load = 0
            current_node = depot
            continue

        total_distance += distance_matrix[current_node][nearest]
        current_load += demands[nearest]
        current_route.append(nearest)
        unvisited.remove(nearest)
        current_node = nearest

    if len(current_route) > 1:
        total_distance += distance_matrix[current_node][depot]
        routes.append(current_route + [depot])

    return routes, total_distance


def nearest_neighbor_with_tw(distance_matrix, time_matrix, time_windows,
                             demands, service_times, vehicle_capacity_weight, depot=0):
    """
    基线B：最近邻 + 时间窗检查（修复了 late_count 统计漏洞和死循环风险）
    返回: routes, total_distance, late_count, on_time_count
    """
    n = len(distance_matrix)
    unvisited = set(range(1, n))
    routes = []
    current_route = [depot]
    current_load = 0
    current_node = depot
    current_time = 0
    total_distance = 0
    late_count = 0
    on_time_count = 0
    max_iter = n * n * 2  # 防止死循环
    iter_count = 0

    while unvisited:
        iter_count += 1
        if iter_count > max_iter:
            print("警告：基线B陷入死循环，强制退出！")
            break

        feasible = []
        for node in unvisited:
            travel = time_matrix[current_node][node]
            arrival = current_time + travel
            tw_start, tw_end = time_windows[node]
            service_start = max(arrival, tw_start)  # 到达后需等待到开门时间

            # 核心修复：必须用 service_start 判断是否超过最晚服务时间 tw_end
            if service_start <= tw_end and demands[node] + current_load <= vehicle_capacity_weight:
                feasible.append((node, travel, arrival, service_start))

        if feasible:
            node, travel, arrival, service_start = min(feasible, key=lambda x: x[1])
            tw_start, tw_end = time_windows[node]

            # 核心修复：准确统计晚到（服务开始时间超过最晚时间窗）
            if service_start > tw_end:
                late_count += 1
            else:
                on_time_count += 1

            total_distance += distance_matrix[current_node][node]
            # 核心修复：使用每个节点独立的 service_times
            current_time = service_start + service_times[node]
            current_load += demands[node]
            current_route.append(node)
            unvisited.remove(node)
            current_node = node

        else:
            # 当前车辆无法继续服务，尝试等待或返回
            waiting = []
            for node in unvisited:
                travel = time_matrix[current_node][node]
                arrival = current_time + travel
                tw_start, tw_end = time_windows[node]
                if arrival < tw_start and demands[node] + current_load <= vehicle_capacity_weight:
                    waiting.append((node, travel, tw_start - arrival))

            if waiting:
                node, travel, wait = min(waiting, key=lambda x: x[1])
                current_time += wait
                tw_start, tw_end = time_windows[node]
                service_start = max(current_time, tw_start)

                # 核心修复：等待后依然要判断是否超时
                if service_start > tw_end:
                    late_count += 1
                else:
                    on_time_count += 1

                total_distance += distance_matrix[current_node][node]
                current_time = service_start + service_times[node]
                current_load += demands[node]
                current_route.append(node)
                unvisited.remove(node)
                current_node = node
            else:
                # 没有可服务的，必须返回depot派新车
                total_distance += distance_matrix[current_node][depot]
                routes.append(current_route + [depot])
                current_route = [depot]
                current_load = 0
                current_time = 0
                current_node = depot

    if len(current_route) > 1:
        total_distance += distance_matrix[current_node][depot]
        routes.append(current_route + [depot])

    return routes, total_distance, late_count, on_time_count