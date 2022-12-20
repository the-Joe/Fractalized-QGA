from qiskit import QuantumCircuit, execute, Aer, IBMQ,QuantumRegister,ClassicalRegister
from qiskit.compiler import transpile, assemble
from qiskit.tools.monitor import job_monitor
from qiskit.tools.jupyter import *
from qiskit.visualization import *
import networkx as nx
import operator
from numpy import flip,array,binary_repr,insert

import math

NO_OF_NODES = 5


def init_graph(no_of_nodes, edge_probability):
    # graph = nx.fast_gnp_random_graph(
    #     no_of_nodes,
    #     edge_probability,seed=None,directed=False)
    graph = nx.erdos_renyi_graph(no_of_nodes, edge_probability, seed=None, directed=False)
    adjacency_matrix = nx.to_numpy_matrix(graph).astype(int)

    return adjacency_matrix

GRAPH = init_graph(NO_OF_NODES,0.4).tolist()

'''
NO_OF_COLORS -- the maximum number of colors used for coloring the graph.
INVALID_COLORS_LIST -- the list of invalid colors. Binary colors are specified as tuples ( e.g. [(1,1),(0,1)] for marking the color configurations 11 and 01 as invalid).
NO_OF_QUBITS_FITNESS -- the number of qubits needed for representing the fitness value.
NO_OF_QUBITS_PER_COLOR -- the number of qubits needed for representing a color configuration.
NO_OF_QUBIT_INDIVIDUAL -- the total number of qubits needed for representing an individual. Thus, considering that an individual is made of chromosomes that represents the color configurations for each node, the total number of qubits is represented by the product between NO_OF_NODES and NO_OF_QUBITS_PER_COLOR.
POPULATION_SIZE -- all possible configurations for coloring the graph
NO_OF_MAX_GROVER_ITERATIONS -- the maximum number of Grover iterations in which a solution should be found.
'''

NO_OF_COLORS = 3# 2 bits needed
INVALID_COLORS_LIST = [(1,1)]
NO_OF_QUBITS_FITNESS = 4
NO_OF_QUBITS_PER_COLOR = 2
NO_OF_QUBITS_INDIVIDUAL = NO_OF_QUBITS_PER_COLOR*NO_OF_NODES
POPULATION_SIZE = 2**NO_OF_QUBITS_INDIVIDUAL #2*NO_OF_NODES#
NO_OF_MAX_GROVER_ITERATIONS = int(math.sqrt(2**NO_OF_QUBITS_FITNESS))

'''
to_binary -- used for getting the two's complement representation.
pairs_colors -- returns a list of tuples representing the color configuration for each node.
check_edges_validity -- returns the number of edges between adjacent nodes colored using different colors.
get_number_of_edges -- returns the number of edges in graph
'''

def to_binary(value, number_of_bits, lsb=False):
    """
    Function return two's complement representation
    :param value: value in decimal representation
    :param number_of_bits: number of bits used for representation
    :returns: np.array that represents the binary representation
    >>> to_binary(10,4)
    array([1, 0, 1, 0])
    >>> to_binary(10,4,True)
    array([0, 1, 0, 1])
    """
    if lsb == True:
        return flip(array(list(binary_repr(value, number_of_bits)), dtype=int))
    return array(list(binary_repr(value, number_of_bits)), dtype=int)

def pairs_colors(colors_list):
    """
    Function returns a list of colors from the binary representation of the individual
    :param colors_list: binary representation of the individual
    :returns: list of pairs representing the color configuration for each node.
    """
    pairs=list()
    for i in range(0,len(colors_list),2):
        pairs.append((colors_list[i],colors_list[i+1]))
    return pairs

def check_edges_validity(graph, colors):
    """
    Function return the number of edges between adjacent nodes colored using different colors
    :param graph: adjacency matrix
    :param colors: list of colors
    :returns: number of edges between adjacent nodes colored using different colors
    """
    no_of_valid_edges = 0
    for color in colors:
        if color in INVALID_COLORS_LIST:
            return -1
    for i in range(NO_OF_NODES):
        for j in range(i + 1, NO_OF_NODES):
            if graph[i][j]:#daca am legatura
                if colors[j]==colors[i]:
                    continue
                else:
                    no_of_valid_edges +=1

    return no_of_valid_edges

def get_number_of_edges(graph):
    """
    Function return the number of edges in graph
    :param graph: adjacency matrix
    :returns: number of edges in graph
    """
    no_of_edges = 0
    for i in range(NO_OF_NODES):
        for j in range(i + 1, NO_OF_NODES):
            if graph[i][j]:
                no_of_edges +=1
    return no_of_edges

def get_ufit_instruction():
    #define and initialize the individual quantum register
    ind_qreg = QuantumRegister(NO_OF_QUBITS_INDIVIDUAL,"ind_qreg")
    #define and initialize the fitness quantum register.
    fit_qreg = QuantumRegister(NO_OF_QUBITS_FITNESS+1,"fit_qreg")
    #create the ufit subcircuit
    qc = QuantumCircuit(ind_qreg,fit_qreg,name="U$_fit$")
    for i in range(0,POPULATION_SIZE):
        """
        For each individual in population get the two's complement representation and 
        set the qubits on 1 using X-gate, according to the binary representation
        """
        individual_binary = to_binary(i, NO_OF_QUBITS_INDIVIDUAL, True)
        for k in range(0,NO_OF_QUBITS_INDIVIDUAL):
            if individual_binary[k] == 0:
                qc.x(ind_qreg[k])
        """
        Create list of colors from individual binary representation, calculate the fitness value
        and get the two's complement representation of the fitness value.
        """
        colors = pairs_colors(individual_binary)
        #calculate valid score
        valid_score = check_edges_validity(GRAPH,colors)
        valid_score_binary = to_binary(valid_score,NO_OF_QUBITS_FITNESS,True)

        """
        Set the fitness value in fitness quantum register for each individual and mark it valid or invalid
        """
        for k in range(0,NO_OF_QUBITS_FITNESS):
            if valid_score_binary[k]==1:
                qc.mct([ind_qreg[j] for j in range(0,NO_OF_QUBITS_INDIVIDUAL)],fit_qreg[k])
        #if fitness value si greater than 0 then set the valid qubit to 1
        if valid_score > 0:
            qc.mct([ind_qreg[j] for j in range(0,NO_OF_QUBITS_INDIVIDUAL)],fit_qreg[NO_OF_QUBITS_FITNESS])
        #reset individual
        for k in range(0,NO_OF_QUBITS_INDIVIDUAL):
            if individual_binary[k] == 0:
                qc.x(ind_qreg[k])
        qc.barrier()
    return qc.to_instruction()


def get_oracle_instruction(positive_value_array):
    # define and initialize fitness quantum register
    fit_reg = QuantumRegister(NO_OF_QUBITS_FITNESS, "fqreg")
    # define and initialize max quantum register
    no_of_edges_reg = QuantumRegister(NO_OF_QUBITS_FITNESS, "noqreg")
    # define and initialize carry quantum register
    carry_reg = QuantumRegister(3, "cqreg")
    # define and initialize oracle workspace quantum register
    oracle = QuantumRegister(1, "oqreg")
    # create Oracle subcircuit
    oracle_circ = QuantumCircuit(fit_reg, no_of_edges_reg, carry_reg, oracle, name="O")

    # define majority operator
    def majority(circ, a, b, c):
        circ.cx(c, b)
        circ.cx(c, a)
        circ.ccx(a, b, c)

    # define unmajority operator
    def unmaj(circ, a, b, c):
        circ.ccx(a, b, c)
        circ.cx(c, a)
        circ.cx(a, b)

    # define the Quantum Ripple Carry Adder
    def adder_4_qubits(p, a0, a1, a2, a3, b0, b1, b2, b3, cin, cout):
        majority(p, cin, b0, a0)
        majority(p, a0, b1, a1)
        majority(p, a1, b2, a2)
        majority(p, a2, b3, a3)
        p.cx(a3, cout)
        unmaj(p, a2, b3, a3)
        unmaj(p, a1, b2, a2)
        unmaj(p, a0, b1, a1)
        unmaj(p, cin, b0, a0)

    """
    Subtract max value. We start by storing the max value in the quantum register. Such, considering that 
    all qubits are |0>, if on position i in positive_value_array there's 0, then qubit i will be negated. Otherwise, 
    if on position i in positive_value_array there's a 1, by default will remain 0 in no_of_edges_reg quantum
    register. For performing subtraction, carry-in will be set to 1.
    """
    for i in range(0, NO_OF_QUBITS_FITNESS):
        if positive_value_array[i] == 0:
            oracle_circ.x(no_of_edges_reg[i])
    oracle_circ.x(carry_reg[0])

    adder_4_qubits(oracle_circ, no_of_edges_reg[0], no_of_edges_reg[1], no_of_edges_reg[2], no_of_edges_reg[3],
                   fit_reg[0], fit_reg[1], fit_reg[2], fit_reg[3],
                   carry_reg[0], carry_reg[1]);

    oracle_circ.barrier()
    """
    Reset the value in no_of_edges_reg and carry-in
    """
    oracle_circ.x(no_of_edges_reg)
    oracle_circ.x(carry_reg[0])

    """
    Mark the corresponding basis states by shifting their amplitudes.
    """

    oracle_circ.h(oracle[0])
    oracle_circ.mct([fit_reg[i] for i in range(0, NO_OF_QUBITS_FITNESS)], oracle[0])
    oracle_circ.h(oracle[0])

    """
    Restore the fitness value by adding max value.
    """
    adder_4_qubits(oracle_circ, no_of_edges_reg[0], no_of_edges_reg[1], no_of_edges_reg[2], no_of_edges_reg[3],
                   fit_reg[0], fit_reg[1], fit_reg[2], fit_reg[3],
                   carry_reg[0], carry_reg[2]);
    return oracle_circ.to_instruction()

def get_grover_iteration_subcircuit():
    #define and initialize fitness quantum register
    fit_qreg = QuantumRegister(NO_OF_QUBITS_FITNESS+1,"fqreg")
    #define and initialize oracle workspace quantum register
    oracle_ws = QuantumRegister(1,"ows")
    #create grover diffuser subcircuit
    grover_circ = QuantumCircuit(fit_qreg,oracle_ws,name ="U$_s$")

    grover_circ.h(fit_qreg)
    grover_circ.x(fit_qreg)

    grover_circ.h(oracle_ws[0])

    grover_circ.mct(list(range(NO_OF_QUBITS_FITNESS+1)), oracle_ws[0])  # multi-controlled-toffoli

    grover_circ.h(oracle_ws[0])


    grover_circ.x(fit_qreg)
    grover_circ.h(fit_qreg)
    grover_circ.h(oracle_ws)

    return grover_circ.to_instruction()


def run_algorithm():
    # IBMQ Account Token
    IBMQ.save_account('4fb6f7554c27ba6c06c3f086d4c8d699d78b6be0e36493cdafe3ad924fa17a615a345700fd76b16f0f3ec51935db72afb85f8f13c946cead9b4ffd5ff89ec576')
    # Load IBMQ account
    IBMQ.load_account()
    # calculate the number of edges in graph
    pos_no_of_edges = get_number_of_edges(GRAPH)
    print("No of edges:{0}".format(pos_no_of_edges))
    # define a list for storing the results
    final_results = []

    # Start with one Grover iteration and run the algorithm with different number of iterations
    for iterations in range(1, NO_OF_MAX_GROVER_ITERATIONS + 1):
        print("Running with {0} iterations".format(iterations))

        print("Preparing quantum registers and creating quantum circuit...")
        ind_qreg = QuantumRegister(NO_OF_QUBITS_INDIVIDUAL, "ireg")
        fit_qreg = QuantumRegister(NO_OF_QUBITS_FITNESS + 1, "freg")  # 8 qubits fitness + 1 valid
        carry_qreg = QuantumRegister(9, "qcarry")
        oracle = QuantumRegister(1, "oracle")
        creg = ClassicalRegister(NO_OF_QUBITS_INDIVIDUAL, "reg")
        no_of_edges_qreg = QuantumRegister(NO_OF_QUBITS_FITNESS, "pos_max_qreg")

        print("Creating quantum circuit...")

        qc = QuantumCircuit(ind_qreg, fit_qreg, carry_qreg, oracle, no_of_edges_qreg, creg)

        print("Creating superposition of individuals...")
        qc.h(ind_qreg)
        qc.h(oracle)

        print("Getting maximum number of edges {0} binary representation...".format(pos_no_of_edges))
        pos_value_bin = to_binary(pos_no_of_edges, NO_OF_QUBITS_FITNESS, True)

        print("Getting ufit, oracle and grover iterations subcircuits...")
        ufit_instr = get_ufit_instruction()
        oracle_instr = get_oracle_instruction(pos_value_bin)
        grover_iter_inst = get_grover_iteration_subcircuit()

        print("Append Ufit instruction to circuit...")
        qc.append(ufit_instr, [ind_qreg[q] for q in range(0, NO_OF_QUBITS_INDIVIDUAL)] +
                  [fit_qreg[q] for q in range(0, NO_OF_QUBITS_FITNESS + 1)]
                  )

        for it in range(0, iterations):
            print("Append Oracle instruction to circuit...")

            qc.append(oracle_instr, [fit_qreg[q] for q in range(0, NO_OF_QUBITS_FITNESS)] +
                      [no_of_edges_qreg[q] for q in range(0, NO_OF_QUBITS_FITNESS)] +
                      [carry_qreg[0], carry_qreg[2 * it + 1], carry_qreg[2 * it + 2]] +
                      [oracle[0]])
            print("Append Grover Diffuser to circuit...")
            qc.append(grover_iter_inst, [fit_qreg[q] for q in range(0, NO_OF_QUBITS_FITNESS + 1)] + [oracle[0]])

        print("Measure circuit...")
        qc.measure(ind_qreg, creg)

        simulation_results = []

        provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
        backend = provider.get_backend('ibmq_qasm_simulator')

        # Perform 10 measurements for each circuit
        for run in range(0, 10):
            print("Setup simulator...")
            shots = 16
            try:
                print("Starting simulator...")
                mapped_circuit = transpile(qc, backend=backend)
                qobj = assemble(mapped_circuit, backend=backend, shots=shots)
                runner = backend.run(qobj)
                job_monitor(runner)
                results = runner.result()
                answer = results.get_counts()
                # Get the result with the maximum number of counts
                max_item = max(answer.items(), key=operator.itemgetter(1))
                print(max_item[0])
                # Store the result.
                simulation_results.append(max_item)
            except Exception as e:
                print(str(e))
                print("Error on run {0} with {1} grover iterations".format(run, iterations))
        final_results.append((iterations, simulation_results))
    print(final_results)
run_algorithm()