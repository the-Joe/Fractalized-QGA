import numpy as np, random, operator, pandas as pd, matplotlib.pyplot as plt
import matplotlib


class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self, city):
        xDis = abs(self.x - city.x)
        yDis = abs(self.y - city.y)
        distance = np.sqrt((xDis ** 2) + (yDis ** 2))
        return distance

    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"


class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness = 0.0

    def routeDistance(self):
        if self.distance == 0:
            pathDistance = 0
            for i in range(0, len(self.route)):
                fromCity = self.route[i]
                toCity = None
                if i + 1 < len(self.route):
                    toCity = self.route[i + 1]
                else:
                    toCity = self.route[0]
                pathDistance += fromCity.distance(toCity)
            self.distance = pathDistance
        return self.distance

    def routeFitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.routeDistance())
        return self.fitness


def createRoute(cityList):
    route = random.sample(cityList, len(cityList))
    return route


def initialPopulation(popSize, cityList):
    population = []
    for i in range(0, popSize):
        population.append(createRoute(cityList))
    return population


def rankRoutes(population):
    fitnessResults = {}
    for i in range(0, len(population)):
        fitnessResults[i] = Fitness(population[i]).routeFitness()
    return sorted(fitnessResults.items(), key=operator.itemgetter(1), reverse=True)


def selection(popRanked, eliteSize):
    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns=["Index", "Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100 * df.cum_sum / df.Fitness.sum()
    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])
    for i in range(0, len(popRanked) - eliteSize):
        pick = 100 * random.random()
        for i in range(0, len(popRanked)):
            if pick <= df.iat[i, 3]:
                selectionResults.append(popRanked[i][0])
                break
    return selectionResults


def matingPool(population, selectionResults):
    matingpool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    return matingpool


def crossover(parent1, parent2):
    child = []
    childP1 = []
    childP2 = []
    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))
    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)
    for i in range(startGene, endGene):
        childP1.append(parent1[i])
    childP2 = [item for item in parent2 if item not in childP1]
    child = childP1 + childP2
    return child


def crossoverPopulation(matingpool, eliteSize):
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))
    for i in range(0, eliteSize):
        children.append(matingpool[i])
    for i in range(0, length):
        child = crossover(pool[i], pool[len(matingpool) - i - 1])
        children.append(child)
    return children


def mutate(individual, mutationRate):
    for swapped in range(len(individual)):
        if (random.random() < mutationRate):
            swapWith = int(random.random() * len(individual))
            city1 = individual[swapped]
            city2 = individual[swapWith]
            individual[swapped] = city2
            individual[swapWith] = city1
    return individual


def nextGeneration(currentGen, eliteSize, mutationRate):
    popRanked = rankRoutes(currentGen)
    selectionResults = selection(popRanked, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    children = crossoverPopulation(matingpool, eliteSize)
    nextGen = []
    for ind in range(0, len(children)):
        mutatedInd = mutate(children[ind], mutationRate)
        nextGen.append(mutatedInd)
    return nextGen


def GA(population, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, population)
    print("Initial distance: " + str(1 / rankRoutes(pop)[0][1]))
    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)
    print("Final distance: " + str(1 / rankRoutes(pop)[0][1]))
    bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]
    print(bestRoute)
    draw_line(bestRoute)
    return bestRoute


def GAPlot(population, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, population)
    progress = []
    progress.append(1 / rankRoutes(pop)[0][1])
    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)
        progress.append(1 / rankRoutes(pop)[0][1])
    plt.plot(progress)
    plt.title("GA for TSP progress", fontsize=10)
    plt.ylabel("Distance", fontsize=10)
    plt.xlabel("Generation", fontsize=10)
    plt.show()


# Draw multiple points.
def draw_multiple_points(coord_list):
    for i in range(0, len(coord_list)):
        plt.scatter(coord_list[i].x, coord_list[i].y, s=10)
    plt.title("City Coord ", fontsize=10)
    plt.xlabel("x Coord", fontsize=10)
    plt.ylabel("y Coord", fontsize=10)
    plt.show()


# Plot a line based on the x and y axis value list.
def draw_line(coord_list):
    lenList = len(coord_list)
    x_number_values = []
    y_number_values = []
    for i in range(0, lenList):
        x_number_values.append(coord_list[i].x)
        y_number_values.append(coord_list[i].y)
    plt.plot(x_number_values, y_number_values, 'o-', linewidth=1)
    plt.title("Best Route", fontsize=10)
    plt.xlabel("x Coord", fontsize=10)
    plt.ylabel("y Coord", fontsize=10)
    plt.show()


if __name__ == "__main__":
    # GA Parameters
    popSize = 3628800
    eliteSize = 20
    mutationRate = 0.001
    generations = 200
    cityList = []
    num_city = 11

    # Route Plan
    for i in range(0, num_city):
        newCity = City(x=int(random.random() * 200), y=int(random.random() * 200))
        cityList.append(newCity)

    print("city list:", cityList)
    draw_multiple_points(cityList)

    # Start GA procedure
    GA(population=cityList,
       popSize=popSize,
       eliteSize=eliteSize,
       mutationRate=mutationRate,
       generations=generations)

    # Plot GA procedure
    GAPlot(population=cityList,
           popSize=popSize,
           eliteSize=eliteSize,
           mutationRate=mutationRate,
           generations=generations)





