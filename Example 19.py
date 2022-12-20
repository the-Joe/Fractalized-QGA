import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import execute
from qiskit import BasicAer
import matplotlib.pyplot as plt


#Build the Quantum Model
q = QuantumRegister(1, 'q')
circ = QuantumCircuit(q)
circ.id(q[0])
print(circ)

#Run the simulation
from qiskit import BasicAer
backend = BasicAer.get_backend('statevector_simulator')
job = execute(circ, backend)
result = job.result()
vector = result.get_statevector(circ, decimals=3)
print(vector)

from qiskit.tools.visualization import plot_bloch_vector
p = np.outer(vector,vector.conj())
px = np.array([[0,1],[1,0]])
py = np.array([[0,-1j],[1j,0]])
pz = np.array([[1,0],[0,-1]])

x = float(np.trace(np.dot(px,p)))
y = float(np.trace(np.dot(py,p)))
z = float(np.trace(np.dot(pz,p)))
vec=[x,y,z]
plot_bloch_vector(vec)
plt.show()

#Build the Quantum Model
q = QuantumRegister(1, 'q')
circ = QuantumCircuit(q)
circ.x(q[0])
print(circ)

#Run the simulation
backend = BasicAer.get_backend('statevector_simulator')
job = execute(circ, backend)
result = job.result()
vector = result.get_statevector(circ, decimals=3)
print(vector)

p = np.outer(vector,vector.conj())
px = np.array([[0,1],[1,0]])
py = np.array([[0,-1j],[1j,0]])
pz = np.array([[1,0],[0,-1]])

x = float(np.trace(np.dot(px,p)))
y = float(np.trace(np.dot(py,p)))
z = float(np.trace(np.dot(pz,p)))
vec=[x,y,z]
plot_bloch_vector(vec)

plt.show()

#Build the Quantum Model
q = QuantumRegister(1, 'q')
circ = QuantumCircuit(q)
circ.h(q[0])
print(circ)

#Run the simulation
backend = BasicAer.get_backend('statevector_simulator')
job = execute(circ, backend)
result = job.result()
vector = result.get_statevector(circ, decimals=3)
print(vector)
p = np.outer(vector,vector.conj())
px = np.array([[0,1],[1,0]])
py = np.array([[0,-1j],[1j,0]])
pz = np.array([[1,0],[0,-1]])

x = float(np.trace(np.dot(px,p)))
y = float(np.trace(np.dot(py,p)))
z = float(np.trace(np.dot(pz,p)))
vec=[x,y,z]
plot_bloch_vector(vec)
plt.show()