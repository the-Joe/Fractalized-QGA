import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import execute

q = QuantumRegister(3, 'q')

circ = QuantumCircuit(q)

circ.h(q[0])

circ.cx(q[0], q[1])

circ.cx(q[0], q[2])

print(circ)