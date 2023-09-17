#!/usr/bin/env python
# coding: utf-8

# In[2]:


import math
import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from gurobipy import *

def read_vrp_file(path, filename):
    capacity = -1
    coordinates = {}
    demand = []

    with open(path + filename, 'r') as file:
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
                    coordinates[node_id] = (x_coord, y_coord)
                elif current_section == 'demand':
                    demand_data = line.split()
                    node_id = int(demand_data[0])
                    demand_value = int(demand_data[1])
                    demand.append(demand_value)

    # Duplicate the depot node
    duplicated_depot = max(coordinates.keys()) + 1
    coordinates[duplicated_depot] = coordinates[1]
    demand.append(demand[0])

    return capacity, coordinates, demand

# Specify the path and filename
path = "Vrp-Set-X/Vrp-Set-X/X/"
filename = 'X-n101-k25.vrp'

# Read instance file and store data
capacity, coordinates, demand = read_vrp_file(path, filename)
print('Capacity:', capacity)
print('Coordinates:', coordinates)
print('Demand:', demand)


# In[3]:


def create_VRP_distM(coordinates):
    positions = [coord for coord in coordinates.values()]
    distances = squareform(pdist(positions, 'euclidean'))
    return distances

# Create distance matrix
distances = create_VRP_distM(coordinates)

# Print the distance matrix
for row in distances:
    print('\t'.join(str(distance) for distance in row))


# In[4]:


def plot_vrp_graph(coordinates):
    '''function for plotting nodes
    coordinates = coordinates (stored in a dictionary)
    '''
    x_coords = [coord[0] for coord in coordinates.values()]
    y_coords = [coord[1] for coord in coordinates.values()]
    
    plt.figure(figsize=(8, 6))
    
    # Depot (demand=0)
    plt.scatter(x_coords[0], y_coords[0], color='orange', marker='s')
    # Duplicate depot
    plt.scatter(x_coords[-1], y_coords[-1], color='orange', marker='s')
    # Customers (demand>0)
    plt.scatter(x_coords[1:-1], y_coords[1:-1], color='blue')
    plt.title('Locations')
    
    plt.show()

plot_vrp_graph(coordinates)


# In[5]:


N = len(demand) # number of customers including the depots
n = N - 2 # number of customers
K = 25 # number of vehicles
Q = 206.0 # vehicle capacity


# In[6]:


# Create a new model
model = Model("Capacitated_VRP")


# In[7]:


# Create decision variables
x = model.addVars(N, N, vtype=GRB.BINARY, name="x")
y = model.addVars(N, lb=0, ub=Q, vtype=GRB.CONTINUOUS, name="y")


# In[8]:


# Set objective function
model.setObjective(quicksum(distances[i][j] * x[i, j] for i in range(N) for j in range(N)), GRB.MINIMIZE)


# In[9]:


# Add constraints (2)
model.addConstrs((quicksum(x[i, j] for j in range(1, N) if j != i) == 1 for i in range(1, n+1)), name="C2");


# In[10]:


# Add constraints (3)
model.addConstrs((quicksum(x[i, h] for i in range(N-1) if i != h) - quicksum(x[h, j] for j in range(1, N) if j != h) == 0
                  for h in range(1, n+1)), name="C3");


# In[11]:


# Add constraints (4)
model.addConstr(quicksum(x[0, j] for j in range(1, n+1)) <= K, name="C4")


# In[12]:


# Add constraints (5)
model.addConstrs((y[j] >= y[i] + demand[j] * x[i, j] - Q * (1 - x[i, j]) for i in range(N) for j in range(N)), name="C5");


# In[13]:


# Add constraints (6)
model.addConstrs((y[i] >= demand[i] for i in range(N)), name="C6a");


# In[ ]:


# Optimize the model
model.optimize()


# In[ ]:




