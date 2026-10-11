import os
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data_loader import load_day_data
from baselines import nearest_neighbor_with_tw
from solver import solve_cvrptw, extract_routes, compute_total_distance

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

st.set_page_config(page_title="VRP Optimization", layout="wide")
st.title("City Last-Mile Medical Delivery Optimization")
st.markdown("**Dataset**: Delivering Data (Zenodo) | **Solver**: Google OR-Tools")


@st.cache_data
def load_all():
    return load_day_data(DATA_DIR, day_index=1)


@st.cache_data
def compute_coords(dist_matrix):
    """Classic MDS implemented with numpy (no sklearn dependency)."""
    D = np.array(dist_matrix, dtype=float)
    D = (D + D.T) / 2.0
    np.fill_diagonal(D, 0.0)

    n = D.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ (D ** 2) @ H

    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    eigvals = np.maximum(eigvals[:2], 0)
    return eigvecs[:, :2] * np.sqrt(eigvals)


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
                markersize=5, label=f"Vehicle {v}", alpha=0.8)
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


st.sidebar.header("Parameter Settings")
scenario = st.sidebar.selectbox(
    "Traffic Scenario",
    ["most_likely", "optimistic", "pessimistic"],
    format_func=lambda x: {"most_likely": "Most Likely",
                            "optimistic": "Optimistic",
                            "pessimistic": "Pessimistic"}[x]
)
num_vehicles = st.sidebar.slider("Max Vehicles", 3, 6, 4)
time_limit = st.sidebar.slider("Time Limit (s)", 10, 120, 30)
run_btn = st.sidebar.button("Run Optimization")

data = load_all()
time_matrix = data[f"time_matrix_{scenario}"]
coords = compute_coords(data["distance_matrix"])
color_map = ["red", "blue", "green", "orange", "purple", "brown"]

if run_btn:
    with st.spinner("Running baseline..."):
        routes_b, dist_b, _, _ = nearest_neighbor_with_tw(
            data["distance_matrix"], time_matrix,
            data["time_windows"], data["demands"],
            data["service_times"], data["vehicle_capacity_weight"]
        )
        routes_b_plot = [[{"node": n, "arrival": 0} for n in r] for r in routes_b]

    with st.spinner(f"Solving with OR-Tools ({time_limit}s)..."):
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

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Baseline Distance", f"{dist_b:.1f} km")
        col2.metric("Optimized Distance", f"{dist_o:.1f} km",
                    delta=f"-{(dist_b - dist_o):.1f} km")
        col3.metric("Baseline Vehicles", len(routes_b))
        col4.metric("Optimized Vehicles", used)

        st.markdown(f"**Distance Reduction**: {(dist_b - dist_o) / dist_b * 100:.1f}%")

        c1, c2 = st.columns(2)
        with c1:
            fig1 = plot_routes(coords, routes_b_plot,
                               f"Baseline (Nearest Neighbor) — {dist_b:.1f} km / {len(routes_b)} vehicles",
                               color_map)
            st.pyplot(fig1)
        with c2:
            fig2 = plot_routes(coords, routes_o,
                               f"OR-Tools — {dist_o:.1f} km / {used} vehicles",
                               color_map)
            st.pyplot(fig2)
    else:
        st.error(f"No feasible solution with {num_vehicles} vehicles. Try increasing.")
else:
    st.info("Set parameters on the left, then click 'Run Optimization'.")