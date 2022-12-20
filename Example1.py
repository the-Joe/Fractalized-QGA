import numpy as np

def And(x1, x2):
    x = np.array([1, x1, x2])
    w = np.array([-1.5, 1, 1])
    y = np.sum(w*x)
    if y <= 0:
        return 0
    else:
        return 1

def Or(x1, x2):
    x = np.array([1, x1, x2])
    w = np.array([-0.5, 1, 1])
    y = np.sum(w*x)
    if y <= 0:
        return 0
    else:
        return 1

def Nand(x1, x2):
    x = np.array([1, x1, x2])
    w = np.array([1.5, 1, 1])
    y = np.sum(w*x)
    if y <= 0:
        return 0
    else:
        return 1

if __name__ == '__main__':
    input = [(0, 0), (1, 0), (0, 1), (1, 1)]

    print("AND")
    for x in input:
        y = And(x[0], x[1])
        print(str(x) + " -> " +str(y))

    print("OR")
    for x in input:
        y = Or(x[0], x[1])
        print(str(x) + " -> " + str(y))

    print("NAND")
    for x in input :
        y = Nand(x[0], x[1])
        print(str(x) + " -> " + str(y))
