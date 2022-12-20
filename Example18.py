import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import BasicAer, execute
import matplotlib.pyplot as plt

q0 = QuantumRegister(2, 'q0')
c0 = ClassicalRegister(2, 'c0')
q1 = QuantumRegister(2, 'q1')
c1 = ClassicalRegister(2, 'c1')
q_test = QuantumRegister(2, 'q0')

print(q0.name)
print(q0.size)

circ = QuantumCircuit(q0, q1)
circ.x(q0[1])
circ.x(q1[0])
print(circ)

meas = QuantumCircuit(q0, q1, c0, c1)
meas.measure(q0, c0)
meas.measure(q1, c1)
qc = circ + meas
print(qc)

plt.show()