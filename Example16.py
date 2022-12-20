import numpy as np
from qiskit import BasicAer
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import DensityMatrix
from qiskit import execute
from qiskit.visualization import plot_state_city
import matplotlib.pyplot as plt

# Build the Quantum Model
q = QuantumRegister(3, 'q')
circ = QuantumCircuit(q)
circ.h(q[0])
circ.cx(q[0], q[1])
circ.cx(q[0], q[2])
print(circ)

# Run the simulation
backend = BasicAer.get_backend('statevector_simulator')
job = execute(circ, backend)
result = job.result()
outputstate = DensityMatrix(result.get_statevector(circ, decimals=3))
print(outputstate)

# Visualization toolbox
plot_state_city(outputstate, color=['midnightblue', 'crimson'], title="New State City")
plt.show()