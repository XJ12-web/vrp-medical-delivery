from ortools.constraint_solver import routing_enums_pb2, pywrapcp


def solve_cvrptw(distance_matrix, time_matrix, time_windows, demands,
                 service_times, vehicle_capacity, num_vehicles, depot=0,
                 time_limit=60):
    manager = pywrapcp.RoutingIndexManager(len(distance_matrix), num_vehicles, depot)
    routing = pywrapcp.RoutingModel(manager)

    # --- 距离回调 ---
    def distance_callback(from_idx, to_idx):
        from_node = manager.IndexToNode(from_idx)
        to_node = manager.IndexToNode(to_idx)
        return int(distance_matrix[from_node][to_node] * 1000)

    dist_cb = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(dist_cb)

    # --- 容量约束 ---
    def demand_callback(from_idx):
        return int(demands[manager.IndexToNode(from_idx)])

    demand_cb = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_cb, 0, [int(vehicle_capacity)] * num_vehicles, True, "Capacity"
    )

    # --- 时间维度 ---
    max_tw_end = max([tw[1] for tw in time_windows])
    horizon = max_tw_end + 120

    def time_callback(from_idx, to_idx):
        from_node = manager.IndexToNode(from_idx)
        to_node = manager.IndexToNode(to_idx)
        return int(time_matrix[from_node][to_node] * 1000 + service_times[to_node] * 1000)

    time_cb = routing.RegisterTransitCallback(time_callback)
    routing.AddDimension(time_cb, 0, horizon * 1000, False, "Time")
    time_dim = routing.GetDimensionOrDie("Time")

    for node, (tw_start, tw_end) in enumerate(time_windows):
        if node == depot:
            continue
        idx = manager.NodeToIndex(node)
        time_dim.CumulVar(idx).SetRange(int(tw_start * 1000), int(tw_end * 1000))

    # --- 强制所有客户必须被访问 ---
    for node in range(1, len(distance_matrix)):
        idx = manager.NodeToIndex(node)
        routing.AddDisjunction([idx], 10**9)

    # --- 搜索参数 ---
    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_params.time_limit.seconds = time_limit

    solution = routing.SolveWithParameters(search_params)
    return solution, routing, manager


def extract_routes(solution, routing, manager, num_vehicles, time_windows, demands):
    routes = []
    late_count = 0
    on_time_count = 0
    time_dim = routing.GetDimensionOrDie("Time")

    for v in range(num_vehicles):
        index = routing.Start(v)
        route = []
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            arrival = solution.Min(time_dim.CumulVar(index)) / 1000.0
            tw_start, tw_end = time_windows[node]
            if node != 0:
                if arrival > tw_end:
                    late_count += 1
                else:
                    on_time_count += 1
            route.append({"node": node, "arrival": arrival})
            index = solution.Value(routing.NextVar(index))
        routes.append(route)
    return routes, on_time_count, late_count


def compute_total_distance(routes, distance_matrix):
    total = 0
    for route in routes:
        for i in range(len(route) - 1):
            total += distance_matrix[route[i]["node"]][route[i + 1]["node"]]
    return total