#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import all necessary modules
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from gurobipy import *
import numpy as np
from scipy.spatial import distance


# In[2]:


def read_vrp_file(path, filename):
    '''function for reading VRP instance file
    path = path
    filename = filename
    returns capacity (float), coordinates (dict), demand (dict)
    '''
    capacity = -1
    coordinates = {}
    demand = {}
    
    with open(path+filename, 'r') as file:
        current_section = None
        for line in file:
            line = line.strip()
            if line == '':
                continue
            if line.startswith("CAPACITY"):
                capacity_data = line.split()
                capacity = float(capacity_data[2])
            elif line == 'NODE_COORD_SECTION':
                current_section = 'coordinates'
            elif line == 'DEMAND_SECTION':
                current_section = 'demand'
            elif line == 'DEPOT_SECTION':
                current_section = 'depot'
            elif line == 'EOF':
                break                
            else:
                if current_section == 'coordinates':
                    node_data = line.split()
                    node_id = int(node_data[0])
                    x_coord = float(node_data[1])
                    y_coord = float(node_data[2])
                    coordinates[node_id - 1] = (x_coord, y_coord)  # Adjust the key
                elif current_section == 'demand':
                    demand_data = line.split()
                    node_id = int(demand_data[0])
                    demand_value = int(demand_data[1])
                    demand[node_id - 1] = demand_value  # Adjust the key

    return capacity, coordinates, demand

# If the instance file is not in the same folder as your Jupyter notebook, specify path
path = "Vrp-Set-X/Vrp-Set-X/X/"
# Name of the instance file
filename = "X-n331-k15.vrp"
# Read instance file and store data
capacity, coordinates, demand = read_vrp_file(path, filename)

print('Capacity:', capacity)
print('Coordinates:', coordinates)
print('Demand:', demand)


# In[3]:


# Function to generate distance matrix
def create_VRP_distM(euclidean_coordinates):
    '''function computing the distance matrix (using a function from scipy)
    euclidean_coordinates = euclidean coordinates (stored in a dictionary)
    '''
    positions = [coord for coord in euclidean_coordinates.values()]  # Use euclidean_coordinates
    distances = squareform(pdist(positions, 'euclidean'))
    return distances

# Create distance matrix using adjusted coordinates
distances = create_VRP_distM(coordinates)
print("Distance matrix:\n", distances)


# In[4]:


def plot_vrp_graph(coordinates, routes=None):
    '''
    Function for plotting nodes and routes for the VRP.
    coordinates: Node coordinates stored in a dictionary.
    routes: A list of routes, where each route is a list of node indices.
    '''
    x_coords = [coord[0] for coord in coordinates.values()]
    y_coords = [coord[1] for coord in coordinates.values()]

    plt.figure(figsize=(8, 6))

    # Depot (demand=0)
    plt.scatter(x_coords[0], y_coords[0], color='orange', marker='s', s=100, label='Depot')
    # Customers (demand>0)
    plt.scatter(x_coords[1:], y_coords[1:], color='blue', s=50, label='Customers')

    # If routes are provided, plot them
    if routes:
        for route in routes:
            for i in range(len(route) - 1):
                plt.plot([coordinates[route[i]][0], coordinates[route[i + 1]][0]],
                         [coordinates[route[i]][1], coordinates[route[i + 1]][1]], 'k-')
    
    plt.title('Locations and Routes')
    plt.legend()
    
    # Uncomment the following lines if you want to label nodes with their indices
    '''
    for node, coord in coordinates.items():
        plt.annotate(str(node), xy=coord, xytext=(5, 5), textcoords='offset points')
    '''
    
    plt.show()
plot_vrp_graph(coordinates)


# In[5]:


def capacity_constrained_kmeans(coordinates, demand, capacity, k, max_iter=100, depot_weight=0.5):
    depot_coord = np.array(coordinates[0]).reshape(1, -1)
    nodes = list(coordinates.keys())[1:]
    X = np.array([coordinates[node] for node in nodes])
    
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=0)
    kmeans.fit(X)
    centroids = kmeans.cluster_centers_
    
    for iteration in range(max_iter):
        clusters = {i: [] for i in range(k)}
        
        for node in nodes:
            coord = np.array(coordinates[node]).reshape(1, -1)
            distances_to_centroids = distance.cdist(coord, centroids, 'euclidean')[0]
            distance_to_depot = distance.cdist(coord, depot_coord, 'euclidean')[0][0]
            
            distances = depot_weight * distance_to_depot + (1 - depot_weight) * distances_to_centroids
            
            sorted_clusters = np.argsort(distances)
            
            for cluster_idx in sorted_clusters:
                if sum([demand[i] for i in clusters[cluster_idx]]) + demand[node] <= capacity:
                    clusters[cluster_idx].append(node)
                    break
        
        new_centroids = []
        for cluster, node_list in clusters.items():
            if node_list:
                avg_x = sum([coordinates[node][0] for node in node_list]) / len(node_list)
                avg_y = sum([coordinates[node][1] for node in node_list]) / len(node_list)
                new_centroids.append([avg_x, avg_y])
            else:
                new_centroids.append(centroids[cluster])
        
        if np.allclose(centroids, new_centroids, atol=1e-4):
            break
        centroids = np.array(new_centroids)
    
    return clusters

k = 15
clusters = capacity_constrained_kmeans(coordinates, demand, capacity, k, depot_weight=0.5)

# Check for unassigned nodes
assigned_nodes = [node for cluster_nodes in clusters.values() for node in cluster_nodes]
unassigned_nodes = set(coordinates.keys()) - set(assigned_nodes) - {0}  # Excluding depot from unassigned check

# Print clusters
for key, value in clusters.items():
    print(f"Cluster {key}: {value}")

# Print unassigned nodes
if unassigned_nodes:
    print(f"\nUnassigned Nodes: {sorted(list(unassigned_nodes))}")
else:
    print("\nAll nodes have been assigned!")


# In[6]:


# Calculate the total demand for each cluster
cluster_demands = {}

for cluster_id, node_list in clusters.items():
    cluster_demand = sum([demand[node] for node in node_list])
    cluster_demands[cluster_id] = cluster_demand

# Print the total demand for each cluster
for cluster_id, total_demand in cluster_demands.items():
    print(f"Total Demand for Cluster {cluster_id}: {total_demand}")


# In[7]:


# Convert dictionary of coordinates to a list
coords_list = list(coordinates.values())

# Number of clusters
num_clusters = 15

# KMeans clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(coords_list)
cluster_centers = kmeans.cluster_centers_

print("Cluster Centers:", cluster_centers)


# In[8]:


# Compute distance matrix for cluster centers
cluster_distance_matrix = create_VRP_distM({i: center for i, center in enumerate(cluster_centers)})
print("Cluster distance matrix:\n", cluster_distance_matrix)


# In[9]:


def nearest_neighbor_tsp(distance_matrix, start_node=0):
    """Solve TSP using the Nearest Neighbor algorithm."""
    num_nodes = len(distance_matrix)
    unvisited = set(range(num_nodes))
    current_node = start_node
    tour = [current_node]
    unvisited.remove(current_node)

    while unvisited:
        nearest_neighbor = min([(distance_matrix[current_node][node], node) for node in unvisited], key=lambda x: x[0])
        current_node = nearest_neighbor[1]
        unvisited.remove(current_node)
        tour.append(current_node)

    return tour

def two_opt_swap(route, i, k):
    """Perform the 2-opt swap."""
    new_route = route[:i]
    new_route.extend(reversed(route[i:k+1]))
    new_route.extend(route[k+1:])
    return new_route

def two_opt(route, distance_matrix):
    """Improve the route using the 2-opt algorithm."""
    improvement = True
    best_route = route
    best_distance = sum(distance_matrix[best_route[i]][best_route[i+1]] for i in range(len(best_route) - 1))

    while improvement:
        improvement = False
        for i in range(1, len(best_route) - 1):
            for k in range(i + 1, len(best_route)):
                new_route = two_opt_swap(best_route, i, k)
                new_distance = sum(distance_matrix[new_route[j]][new_route[j+1]] for j in range(len(new_route) - 1))
                if new_distance < best_distance:
                    best_distance = new_distance
                    best_route = new_route
                    improvement = True

    return best_route

# Solve TSP using Nearest Neighbor for the cluster centers
nn_route_centers = nearest_neighbor_tsp(cluster_distance_matrix)
    
# Refine the solution with 2-opt for the cluster centers
optimized_route_centers = two_opt(nn_route_centers, cluster_distance_matrix)

print(f"Optimized Route for Cluster Centers: {optimized_route_centers}")


# In[11]:


def two_opt_swap(route, i, k):
    """Perform 2-opt swap."""
    new_route = route[:i]
    new_route.extend(reversed(route[i:k + 1]))
    new_route.extend(route[k + 1:])
    return new_route

def two_opt(distances, cluster_nodes):
    """2-opt algorithm optimized."""
    best_route = cluster_nodes
    best_distance = compute_distance([0] + best_route + [0], distances)

    improvement = True
    while improvement:
        improvement = False
        for i in range(len(best_route) - 1):  # Exclude depot
            for k in range(i + 1, len(best_route)):
                new_route = two_opt_swap(best_route, i, k)
                new_distance = compute_distance([0] + new_route + [0], distances)
                if new_distance < best_distance:
                    best_route = new_route
                    best_distance = new_distance
                    improvement = True
    return [0] + best_route + [0]

def compute_distance(route, distances):
    """Compute total distance of the route."""
    return sum(distances[route[i]][route[i + 1]] for i in range(len(route) - 1))

# Store the optimized routes for each cluster
cluster_routes = {}

# Order of cluster centers
cluster_order = [0, 14, 6, 1, 10, 12, 9, 2, 11, 5, 8, 3, 7, 4, 13]
# Solve TSP for each cluster
for center in cluster_order:
    cluster_nodes = clusters[center]
    optimized_route = two_opt(distances, cluster_nodes)
    cluster_routes[center] = optimized_route

# Print the optimized routes for each cluster
for center in cluster_order:
    print(f"Cluster {center}: {cluster_routes[center]}")


# In[12]:


def compute_distance(route, distances):
    """Compute total distance of the route."""
    return sum(distances[route[i]][route[i+1]] for i in range(len(route)-1))

def solve_vrp_within_cluster(cluster_nodes, distances, depot=0, vehicle_capacity=None):
    """Solve VRP for nodes within a cluster without returning to the depot after each node."""
    # First, let's create an initial solution using Nearest Neighbor algorithm.
    current_node = depot
    unvisited = set(cluster_nodes)
    route = []
    current_load = 0

    while unvisited:
        next_node = min([(distances[current_node][node], node) for node in unvisited if current_load + demand[node] <= vehicle_capacity], key=lambda x: x[0], default=(None, None))
        
        if next_node[1] is None:  # if no next node fits or all nodes are visited, start new route from depot
            current_node = depot
            current_load = 0
        else:
            route.append(next_node[1])
            current_load += demand[next_node[1]]
            unvisited.remove(next_node[1])
            current_node = next_node[1]
    
    # Now, refine this route using 2-opt.
    optimized_route = two_opt(distances, route)

    return optimized_route

# Now let's solve the VRP for each cluster.
routes_for_clusters = {}
for cluster_id, nodes in clusters.items():
    routes_for_clusters[cluster_id] = solve_vrp_within_cluster(nodes, distances, depot=0, vehicle_capacity=capacity)

for cluster_id, route in routes_for_clusters.items():
    print(f"Route for Cluster {cluster_id}: {route}")


# In[13]:


def total_distance_for_routes(routes, distances):
    total = 0
    for route in routes.values():
        total += compute_distance(route, distances)
    return total

print(total_distance_for_routes(routes_for_clusters, distances))


# In[14]:


def plot_routes(routes, coordinates):
    plt.figure(figsize=(10,10))
    for route in routes.values():
        route_coords = [coordinates[node] for node in route]
        xs, ys = zip(*route_coords)
        plt.plot(xs, ys, '-o')
    plt.title("Routes Visualization")
    plt.show()

plot_routes(routes_for_clusters, coordinates)


# In[ ]:




