import random
import numpy as np
from numpy import pi
from itertools import permutations
import math
import operator
import networkx as nx
import matplotlib.pyplot as plt
from qiskit import Aer, QuantumRegister, QuantumCircuit, ClassicalRegister, execute
from qiskit.circuit.library import QFT
from qiskit_optimization.applications import Tsp
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit.visualization import plot_histogram, array_to_latex

# Define number of cities/nodes to base the TSP on - number of genes
n = 13


# Define a random initial state for the problem to be modelled
tsp = Tsp.create_random_instance(n)
qp = tsp.to_quadratic_program()
qp2qubo = QuadraticProgramToQubo()
qubo = qp2qubo.convert(qp)
qubitOp, offset = qubo.to_ising()
adj_matrix = nx.to_numpy_matrix(tsp.graph)

print("Orthogonal Unitary Matrix of Distance/Phase:\n", adj_matrix)
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_aspect('equal')
plt.imshow(adj_matrix, interpolation='nearest', cmap=plt.cm.plasma)
plt.colorbar()
plt.show()


# Instantiate networkx graph for map of cities/nodes
# Plots the initial city/node diagram
g = nx.DiGraph(directed=True)
pos = [tsp.graph.nodes[node]["pos"] for node in tsp.graph.nodes]
fig = plt.figure(1, figsize=(14, 10))
colors = ["g" for node in tsp.graph.nodes]
pos = [tsp.graph.nodes[node]["pos"] for node in tsp.graph.nodes]

def draw_graph(g, colors, pos):
    default_axes = plt.axes(frameon=True)
    nx.draw_networkx(g, node_color=colors, node_size=600, alpha=0.8, ax=default_axes, pos=pos)
    edge_labels = nx.get_edge_attributes(g, "weight")
    nx.draw_networkx_edge_labels(g, pos=pos, edge_labels=edge_labels)


draw_graph(tsp.graph, colors, pos)
plt.show()