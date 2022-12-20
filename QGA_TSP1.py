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
n = 5

# Define number of classical binary bits for representing chromosomes
bit_count = int((n+1) * 2)

# Define qubit count necessary based on amount of nodes
num_qubits = n**2
list_qubits = []
for Q in range(0, num_qubits):
    list_qubits.append(Q)

# Define a random initial state for the problem to be modelled
tsp = Tsp.create_random_instance(n)
qp = tsp.to_quadratic_program()
qp2qubo = QuadraticProgramToQubo()
qubo = qp2qubo.convert(qp)
qubitOp, offset = qubo.to_ising()
adj_matrix = nx.to_numpy_matrix(tsp.graph)

print("Orthogonal Unitary Matrix of Distance/Phase:\n", adj_matrix)

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


# Overall unitary is the tensor of all n * n matrices for n unitaries
# Total elements in overall unitary is n^n
unitary_elements = n**n
print('Total Number of Unitary Elements: ' + str(unitary_elements))

# The total number of Hamiltonian cycles in our 'n' city model
ham_cycle_total = math.factorial(n)
print('Total Hamiltonian Cycles: ' + str(ham_cycle_total))

# The number of distinct Hamiltonian cycles / eigenstates in our 'n' city model
ham_cycle_distinct = math.factorial(n-1)
print('Distinct Hamiltonian Cycles / Eigenstates: ' + str(ham_cycle_distinct))


# Calculate all possible route sequences and store in chromosome_list
sequences = list(permutations(range(n)))
print('Eigenstate Sequences Generated:')
print(sequences)
print(len(sequences))


# Convert Eigenstates to Binary
bin_sequences = []
for i in range(len(sequences)):
    bin_sequences.append((bin(int(str("".join(map(str, sequences[i])))))[2:]).zfill(bit_count))

print('Binary Eigenstate Sequences Generated:')
print(bin_sequences)

# Function to place appropriate corresponding gate according to eigenstates
def eigenstates(qc, eigen, index):
    for i in range(0, len(eigen)):
        if bin_sequences[index][i] == '1':
            qc.x(eigen[i])
        if bin_sequences[index][i] == '0':
            pass
    qc.barrier()
    return qc


# Catalog only sequences that match point of origin
origin = 0
origin_matched_sequences = []
for j in range(len(sequences)):
    if sequences[j][0] == origin:
        origin_matched_sequences.append(sequences[j])
print("Verified Distinct Hamiltonian Cycles / Eigenstates: ")
print(str(len(origin_matched_sequences)))

# Convert Eigenstates to Binary
bin_sequences = []
for i in range(len(origin_matched_sequences)):
    bin_sequences.append((bin(int(str("".join(map(str, origin_matched_sequences[i])))))[2:]).zfill(bit_count))

print('Binary Eigenstate Sequences Generated w/ Matched Origin:')
print(bin_sequences)

'''
# Classically calculate distance for sequences
sequential_distance_totals = []
print("The total distance for each origin matched sequence:")
for k in range(len(origin_matched_sequences)):
    sequence_distance = 0
    for l in range(len(origin_matched_sequences[k])):
        if l < len(origin_matched_sequences[k]) - 1:
            sequence_distance = sequence_distance + abs((adj_matrix[origin_matched_sequences[k][l], origin_matched_sequences[k][l + 1]]))
        else:
            sequence_distance = sequence_distance + abs((adj_matrix[origin_matched_sequences[k][l], origin_matched_sequences[k][0]]))
    sequential_distance_totals.append(sequence_distance)
print('The sequential distance totals are: ' + str(sequential_distance_totals))
'''

# Quantum method for calculating distance for sequences
# Build the whole circuit part by part, by unitaries, and by eigenstates. Creating a function to create CU_j.
# The unitary (U) is the crucial part of quantum phase estimation.
# Building the (U), means building the CU_1, CU_2, CU_3, CU_4... CU_n

# Define controlled_unitary(qc, qubits: list, phases: list):  # x,y,z = Specific Qubit; a,b,c,d...n = Phases
def controlled_unitary(qc, qubits: list, phases: list):
    for cunity in range(0, n-1):
        if cunity <= n-2:
            qc.cp(phases[cunity - cunity + 2] - phases[cunity - cunity], list_qubits[cunity], list_qubits[cunity + 1])  # controlled-U1(c-a)
            qc.p(phases[cunity - cunity], qubits[cunity - cunity])  # U1(a)
            qc.cp(phases[cunity - cunity + 1] - phases[cunity - cunity], list_qubits[cunity], list_qubits[cunity + 2])  # controlled-U1(b-a)
        elif cunity > n-2:
            qc.cp((phases[3] - phases[2] + phases[0] - phases[1]) / 2, qubits[1], qubits[2])
            qc.cx(qubits[0], qubits[1])
            qc.cp(-(phases[3] - phases[2] + phases[0] - phases[1]) / 2, qubits[1], qubits[2])
            qc.cx(qubits[0], qubits[1])
            qc.cp((phases[3] - phases[2] + phases[0] - phases[1]) / 2, qubits[0], qubits[2])

            qc.cp((phases[cunity] - phases[cunity - 1] + phases[cunity - cunity] - phases[cunity - cunity + 1]) / 2, qubits[cunity - cunity + 1], qubits[cunity - cunity + 2])
            qc.cx(qubits[cunity - cunity], qubits[cunity - cunity + 1])
            qc.cp(-(phases[cunity] - phases[cunity - 1] + phases[cunity - cunity] - phases[cunity - cunity + 1]) / 2, qubits[cunity - cunity + 1], qubits[cunity - cunity + 2])
            qc.cx(qubits[cunity - cunity], qubits[cunity - cunity + 1])
            qc.cp((phases[cunity] - phases[cunity - cunity - 1] + phases[cunity - cunity] - phases[cunity - cunity + 1]) / 2, qubits[cunity - cunity], qubits[cunity - cunity + 2])

# Makes the Global Unitary from tensor of local unitary phase values
# a,b,c = phases for U1; d,e,f = phases for U2; g,h,i = phases for U3; j,k,l = phases for U4; m_list=[m, n, o, p, q, r, s, t, u, a, b, c, d, e, f, g, h, i, j, k, l]
def U(times, qc, unit, eigen, phases: list):
    for unity_phases in range(0, (2*n) - 1):
        controlled_unitary(qc, [unit[unity_phases - unity_phases]]+eigen[unity_phases - unity_phases:unity_phases - unity_phases + 2], [unity_phases - unity_phases]+phases[unity_phases - unity_phases:unity_phases - unity_phases + 3])

def final_U(times, eigen, phases: list):
    unit = QuantumRegister(1, 'unit')
    qc = QuantumCircuit(unit, eigen)
    for _ in range(0, 2**times):
        U(times, qc, unit, eigen, phases)
    return qc.to_gate(label='U'+'_'+(str(2**times)))

# Function to place appropriate corresponding gate according to eigenstates
def eigenstates(qc, eigen, index):
    for i in range(0, len(eigen)):
        if bin_sequences[index][i] == '1':
            qc.x(eigen[i])
        if bin_sequences[index][i] == '0':
            pass
    qc.barrier()
    return qc

# Initialization
unit = QuantumRegister(2*(n - 1), 'unit')
eigen = QuantumRegister(2*n, 'eigen')
unit_classical = ClassicalRegister(2*(n-1), 'unit_classical')
qc = QuantumCircuit(unit, eigen, unit_classical)

# Setting one eigenstate
# Playing with the first eigenstate here i.e. 11000110 from eigen_values list.
# (Try to play with other eigenstates from the eigen_values list)
eigenstates(qc, eigen, 0)
#

# Hadamard on the 'unit' qubits
qc.h(unit[:])
qc.barrier()
#

# Controlled Unitary
# The phases are normalized to be bound within [0, 2π] once we know the range of distances between the cities.
phases = []
variable_phase_count = (3*n) # Ex. phases = [pi / 2, pi / 8, pi / 4, pi / 2, pi... a, b, c, d,...n
for o in range(0, variable_phase_count):
    phases.append(pi / random.randrange(1,2*(n-1),2))
print(phases)

eigenstate_phase_array = np.array(list(zip(bin_sequences,phases)))
print('Eigenstate Phase Array')
print(eigenstate_phase_array)


for i in range(0, 2*(n-1)):
    qc.append(final_U(i, eigen, phases), [unit[(n+1)-i]] + eigen[:])

# Inverse QFT
qc.barrier()
qft = QFT(num_qubits=len(unit), inverse=True, insert_barriers=True, do_swaps=False, name='Inverse QFT')
qc.append(qft, qc.qubits[:len(unit)])
qc.barrier()
#

# Measure
qc.measure(unit, unit_classical)
#


# Draw
qc.draw()
plt.show()

backend = Aer.get_backend('qasm_simulator')
job = execute(qc, backend, shots=8192)
counts = job.result().get_counts()
rankedCounts = sorted(counts.items(), key = operator.itemgetter(1), reverse = True)
print('Ranked Counts: ' + str(rankedCounts))
plot_histogram(counts, color='b', figsize=(10,10),bar_labels=False, title='FGATSP', sort='value_desc', number_to_keep=20, filename='FQGATSP10.png')
plt.show()


# Then perform selection operation
# Then perform recombination
# Then probability of mutation to some degree
# Then iterate over next generation

print('Complete')