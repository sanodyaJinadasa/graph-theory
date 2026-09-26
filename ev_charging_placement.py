"""
Time-Dependent Facility Location for EV Charging Station Placement
====================================================================

This script:
1. Downloads a real road network for a city (via OSMnx).
2. Builds a STATIC p-median model (classic facility location, fixed travel times).
3. Builds a TIME-DEPENDENT p-median model (travel times scaled by simulated
   congestion multipliers, e.g. rush-hour vs. off-peak).
4. Compares the two solutions and shows how much worse the "static-optimal"
   placement performs once you factor in peak-hour congestion.
5. Visualizes both placements on an interactive map (Folium).

SETUP (run this once in a terminal, NOT in this file):
    pip install osmnx networkx matplotlib folium pulp numpy pandas

USAGE:
    python ev_charging_placement.py
"""

import random
import numpy as np
import networkx as nx
import osmnx as ox
import folium
import pulp

# ----------------------------------------------------------------------
# 1. CONFIGURATION -- change these to fit your project
# ----------------------------------------------------------------------

CITY_NAME = "Colombo, Sri Lanka"   # <-- change to your city of choice
NUM_STATIONS = 5                    # number of EV charging stations to place (p)
NUM_CANDIDATE_NODES = 40            # candidate sites sampled from the network (keep small for ILP speed)
NUM_DEMAND_NODES = 60               # demand points sampled from the network
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ----------------------------------------------------------------------
# 2. DOWNLOAD REAL ROAD NETWORK
# ----------------------------------------------------------------------

def get_road_network(city_name):
    print(f"Downloading road network for: {city_name} ...")
    G = ox.graph_from_place(city_name, network_type="drive")
    G = ox.add_edge_speeds(G)      # estimates speed (km/h) per edge from highway type
    G = ox.add_edge_travel_times(G)  # adds 'travel_time' (seconds) to each edge
    print(f"Network loaded: {len(G.nodes)} nodes, {len(G.edges)} edges")
    return G


# ----------------------------------------------------------------------
# 3. SAMPLE DEMAND POINTS AND CANDIDATE STATION SITES
# ----------------------------------------------------------------------

def sample_nodes(G, n, seed_offset=0):
    random.seed(RANDOM_SEED + seed_offset)
    nodes = list(G.nodes)
    return random.sample(nodes, min(n, len(nodes)))


# ----------------------------------------------------------------------
# 4. BUILD STATIC AND TIME-DEPENDENT TRAVEL TIME MATRICES
# ----------------------------------------------------------------------

def build_travel_time_matrix(G, demand_nodes, candidate_nodes, congestion_multiplier=None):
    """
    Returns a matrix [demand x candidate] of shortest-path travel times (seconds).
    If congestion_multiplier is given (a dict edge -> multiplier, or a global
    scalar), edge weights are scaled before computing shortest paths, simulating
    a specific time-of-day traffic condition.
    """
    if congestion_multiplier is not None:
        # Create a temporary weighted copy of the graph with scaled travel times
        G_temp = G.copy()
        for u, v, k, data in G_temp.edges(keys=True, data=True):
            base_time = data.get("travel_time", 1.0)
            if isinstance(congestion_multiplier, dict):
                mult = congestion_multiplier.get((u, v), 1.0)
            else:
                mult = congestion_multiplier
            data["weight"] = base_time * mult
        weight_key = "weight"
        graph_to_use = G_temp
    else:
        weight_key = "travel_time"
        graph_to_use = G

    matrix = np.zeros((len(demand_nodes), len(candidate_nodes)))
    for i, d in enumerate(demand_nodes):
        lengths = nx.single_source_dijkstra_path_length(graph_to_use, d, weight=weight_key)
        for j, c in enumerate(candidate_nodes):
            matrix[i, j] = lengths.get(c, 1e9)  # large penalty if unreachable
    return matrix


def simulate_peak_hour_multipliers(G, peak_multiplier_range=(1.5, 3.0)):
    """
    Simulates rush-hour congestion by assigning a random multiplier to each
    edge, biased by road type (major roads get worse congestion multipliers).
    Replace this with real traffic data if available for your city.
    """
    multipliers = {}
    for u, v, k, data in G.edges(keys=True, data=True):
        highway = data.get("highway", "")
        if isinstance(highway, list):
            highway = highway[0]
        # Major roads (primary/trunk) suffer more from rush-hour congestion
        if highway in ("primary", "trunk", "primary_link", "trunk_link"):
            mult = random.uniform(*peak_multiplier_range)
        elif highway in ("secondary", "secondary_link"):
            mult = random.uniform(1.2, 2.0)
        else:
            mult = random.uniform(1.0, 1.3)
        multipliers[(u, v)] = mult
    return multipliers


# ----------------------------------------------------------------------
# 5. P-MEDIAN MODEL (ILP via PuLP)
# ----------------------------------------------------------------------

def solve_p_median(cost_matrix, p):
    """
    Classic p-median ILP: choose p candidate sites minimizing total
    (demand-weighted) travel cost to the nearest chosen site.
    cost_matrix: shape [num_demand, num_candidates]
    Returns: list of chosen candidate indices, and total objective cost.
    """
    num_demand, num_candidates = cost_matrix.shape
    prob = pulp.LpProblem("p_median", pulp.LpMinimize)

    # x[j] = 1 if candidate j is chosen as a station
    x = pulp.LpVariable.dicts("x", range(num_candidates), cat="Binary")
    # y[i][j] = 1 if demand point i is served by candidate j
    y = pulp.LpVariable.dicts(
        "y", (range(num_demand), range(num_candidates)), cat="Binary"
    )

    # Objective: minimize total travel cost
    prob += pulp.lpSum(
        cost_matrix[i, j] * y[i][j]
        for i in range(num_demand)
        for j in range(num_candidates)
    )

    # Each demand point served by exactly one station
    for i in range(num_demand):
        prob += pulp.lpSum(y[i][j] for j in range(num_candidates)) == 1

    # A demand point can only be served by an open station
    for i in range(num_demand):
        for j in range(num_candidates):
            prob += y[i][j] <= x[j]

    # Exactly p stations opened
    prob += pulp.lpSum(x[j] for j in range(num_candidates)) == p

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    chosen = [j for j in range(num_candidates) if pulp.value(x[j]) > 0.5]
    total_cost = pulp.value(prob.objective)
    return chosen, total_cost


# ----------------------------------------------------------------------
# 6. EVALUATE A GIVEN STATION SET UNDER A DIFFERENT COST MATRIX
# ----------------------------------------------------------------------

def evaluate_placement(chosen_indices, cost_matrix):
    """Given a fixed set of open stations, compute total cost = sum over
    demand points of the min travel time to any open station, under the
    provided cost_matrix (e.g. peak-hour matrix)."""
    sub_matrix = cost_matrix[:, chosen_indices]
    return sub_matrix.min(axis=1).sum()


# ----------------------------------------------------------------------
# 7. VISUALIZE ON MAP
# ----------------------------------------------------------------------

def visualize_placements(G, demand_nodes, candidate_nodes,
                          static_chosen, dynamic_chosen, out_html="placement_map.html"):
    center_node = demand_nodes[0]
    center_lat = G.nodes[center_node]["y"]
    center_lon = G.nodes[center_node]["x"]

    m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

    # Demand points (small gray dots)
    for d in demand_nodes:
        folium.CircleMarker(
            location=[G.nodes[d]["y"], G.nodes[d]["x"]],
            radius=2, color="gray", fill=True, fill_opacity=0.5
        ).add_to(m)

    # Static-optimal stations (blue)
    for idx in static_chosen:
        node = candidate_nodes[idx]
        folium.Marker(
            location=[G.nodes[node]["y"], G.nodes[node]["x"]],
            popup="Static-optimal station",
            icon=folium.Icon(color="blue", icon="bolt", prefix="fa"),
        ).add_to(m)

    # Time-dependent-optimal stations (red)
    for idx in dynamic_chosen:
        node = candidate_nodes[idx]
        folium.Marker(
            location=[G.nodes[node]["y"], G.nodes[node]["x"]],
            popup="Time-dependent-optimal station",
            icon=folium.Icon(color="red", icon="bolt", prefix="fa"),
        ).add_to(m)

    m.save(out_html)
    print(f"Map saved to {out_html}")


# ----------------------------------------------------------------------
# 8. MAIN
# ----------------------------------------------------------------------

def main():
    G = get_road_network(CITY_NAME)

    demand_nodes = sample_nodes(G, NUM_DEMAND_NODES, seed_offset=1)
    candidate_nodes = sample_nodes(G, NUM_CANDIDATE_NODES, seed_offset=2)

    print("\nBuilding STATIC (free-flow) travel time matrix...")
    static_matrix = build_travel_time_matrix(G, demand_nodes, candidate_nodes)

    print("Simulating peak-hour congestion multipliers...")
    peak_multipliers = simulate_peak_hour_multipliers(G)

    print("Building TIME-DEPENDENT (peak-hour) travel time matrix...")
    peak_matrix = build_travel_time_matrix(
        G, demand_nodes, candidate_nodes, congestion_multiplier=peak_multipliers
    )

    print(f"\nSolving static p-median (p={NUM_STATIONS})...")
    static_chosen, static_cost = solve_p_median(static_matrix, NUM_STATIONS)
    print(f"Static-optimal stations (candidate indices): {static_chosen}")
    print(f"Static objective (free-flow total cost): {static_cost:.1f} sec")

    print(f"\nSolving time-dependent p-median (p={NUM_STATIONS})...")
    dynamic_chosen, dynamic_cost = solve_p_median(peak_matrix, NUM_STATIONS)
    print(f"Time-dependent-optimal stations (candidate indices): {dynamic_chosen}")
    print(f"Time-dependent objective (peak-hour total cost): {dynamic_cost:.1f} sec")

    # KEY COMPARISON: how does the static-optimal placement perform
    # once REAL peak-hour congestion is applied?
    static_placement_under_peak = evaluate_placement(static_chosen, peak_matrix)
    dynamic_placement_under_peak = evaluate_placement(dynamic_chosen, peak_matrix)

    gap_seconds = static_placement_under_peak - dynamic_placement_under_peak
    gap_percent = 100 * gap_seconds / dynamic_placement_under_peak

    print("\n" + "=" * 60)
    print("KEY RESULT")
    print("=" * 60)
    print(f"Static-optimal placement, evaluated under peak-hour congestion: "
          f"{static_placement_under_peak:.1f} sec total")
    print(f"Time-dependent-optimal placement, under peak-hour congestion: "
          f"{dynamic_placement_under_peak:.1f} sec total")
    print(f"=> Using the static model costs an extra {gap_seconds:.1f} sec "
          f"({gap_percent:.1f}%) in total driver travel time during peak hours.")

    visualize_placements(G, demand_nodes, candidate_nodes, static_chosen, dynamic_chosen)


if __name__ == "__main__":
    main()
