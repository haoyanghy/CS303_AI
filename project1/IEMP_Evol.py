import argparse
import time
import numpy as np
import networkx as nx
from collections import deque
import random
import copy


def createGraph(filePath):
    G = nx.DiGraph()
    with open(filePath, "r") as f:
        n, m = map(int, f.readline().split())
        for _ in range(m):
            u, v, p1, p2 = map(
                float, f.readline().split()
            )  # source, target, weight1, weight2
            G.add_edge(int(u), int(v), weight1=p1, weight2=p2)
    return G


def loadSeed(filePath, budget):
    with open(filePath, "r") as f:
        k1, k2 = map(int, f.readline().split())
        seeds1 = [int(f.readline().strip()) for _ in range(k1)]
        seeds2 = [int(f.readline().strip()) for _ in range(k2)]
    return seeds1, seeds2


def diffusion(G, seedSet, valueType):
    currentActivated = deque(seedSet)
    returnActivated = set(seedSet)
    exposedNodes = set(seedSet)

    while currentActivated:
        current = currentActivated.popleft()
        for nextV in G.successors(current):
            nextProb = G[current][nextV][f"weight{valueType}"]
            exposedNodes.add(nextV)
            if nextV not in returnActivated:
                if np.random.rand() < nextProb:
                    returnActivated.add(nextV)
                    currentActivated.append(nextV)

    return exposedNodes, returnActivated


def evaluateFitness(G, S1, S2, I1, I2, TIMES):
    totalPhi = 0
    for _ in range(TIMES):
        U1 = set(I1) | set(S1)
        U2 = set(I2) | set(S2)
        r1_exposed, r1_activated = diffusion(G, list(U1), 1)
        r2_exposed, r2_activated = diffusion(G, list(U2), 2)

        symDiff = r1_exposed.symmetric_difference(r2_exposed)
        phi = len(G.nodes) - len(symDiff)
        totalPhi += phi
    return totalPhi / TIMES


def initialPopulation(G, budget, POPULATION_SIZE):
    population = []
    V = len(G.nodes)
    for _ in range(POPULATION_SIZE):
        individual = [False] * (2 * V)
        tempBudget = budget
        nodes = list(G.nodes)
        random.shuffle(nodes)
        for node in nodes:
            if tempBudget == 0:
                break
            if random.random() < 0.5:
                if not individual[node]:
                    individual[node] = True
                    tempBudget -= 1
            else:
                if not individual[V + node]:
                    individual[V + node] = True
                    tempBudget -= 1
        population.append(individual)
    return population


# avoid local minimum / premature convergence
def tournamentSelect(population, fitnesses, TOURNAMENT_SIZE):
    # select tournament size amount of best population
    selected = random.sample(list(zip(population, fitnesses)), TOURNAMENT_SIZE)
    selected = sorted(
        selected, key=lambda x: x[1], reverse=True
    )  # sort descending by fitness
    return copy.deepcopy(selected[0][0])


def twoPointCrossover(parent1, parent2):
    size = len(parent1)
    if size < 2:
        return copy.deepcopy(parent1), copy.deepcopy(parent2)

    point1, point2 = sorted(random.sample(range(size), 2))
    offspring1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
    offspring2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
    return offspring1, offspring2


def bitFlipMutation(individual, MUTATION_RATE):
    mutated = individual.copy()
    for i in range(len(mutated)):
        if random.random() < MUTATION_RATE:
            mutated[i] = not mutated[i]
    return mutated


# ensure len(S1) + len(S2) <= budget
def repair(individual, budget, V):
    S1 = [i for i in range(V) if individual[i]]
    S2 = [i for i in range(V) if individual[V + i]]
    total = len(S1) + len(S2)

    if total > budget:
        while total > budget:
            if random.random() < 0.5 and S1:  # 'and S1' to ensure S1 is not empty
                remove = random.choice(S1)
                individual[remove] = False
                S1.remove(remove)
            elif S2:
                remove = random.choice(S2)
                individual[V + remove] = False
                S2.remove(remove)
            total = len(S1) + len(S2)
    return individual


def EA(G, seedSet1, seedSet2, budget):
    TIMES = 5
    POPULATION_SIZE = 20
    NUM_GENERATIONS = 10
    TOURNAMENT_SIZE = 3
    CROSSOVER_RATE = 0.8
    MUTATION_RATE = 0.01

    V = len(G.nodes)
    I1 = seedSet1
    I2 = seedSet2

    population = initialPopulation(G, budget, POPULATION_SIZE)

    # calculate fitness for each individual
    fitnesses = []
    for individual in population:
        S1 = [node for node in range(V) if individual[node]]
        S2 = [node for node in range(V) if individual[V + node]]
        if len(S1) + len(S2) <= budget:
            fitness = evaluateFitness(G, S1, S2, I1, I2, TIMES)
        else:
            fitness = -(len(S1) + len(S2))
        fitnesses.append(fitness)

    for generation in range(NUM_GENERATIONS):
        newPopulation = []

        while len(newPopulation) < POPULATION_SIZE:
            parent1 = tournamentSelect(population, fitnesses, TOURNAMENT_SIZE)
            parent2 = tournamentSelect(population, fitnesses, TOURNAMENT_SIZE)

            if random.random() < CROSSOVER_RATE:
                offspring1, offspring2 = twoPointCrossover(parent1, parent2)
            else:
                offspring1, offspring2 = copy.deepcopy(parent1), copy.deepcopy(parent2)

            offspring1 = bitFlipMutation(offspring1, MUTATION_RATE)
            offspring2 = bitFlipMutation(offspring2, MUTATION_RATE)

            offspring1 = repair(offspring1, budget, V)
            offspring2 = repair(offspring2, budget, V)

            newPopulation.extend([offspring1, offspring2])

        # trim because 2 offsprings every generation
        if len(newPopulation) > POPULATION_SIZE:
            newPopulation = newPopulation[:POPULATION_SIZE]

        newFitnesses = []
        for individual in newPopulation:
            S1 = [node for node in range(V) if individual[node]]
            S2 = [node for node in range(V) if individual[V + node]]
            if len(S1) + len(S2) <= budget:
                phi = evaluateFitness(G, S1, S2, I1, I2, TIMES)
                fitness = phi
            else:
                fitness = -(len(S1) + len(S2))
            newFitnesses.append(fitness)

        # elitism
        combinedPopulation = population + newPopulation
        combinedFitnesses = fitnesses + newFitnesses
        # select top POPULATION_SIZE individuals
        sortedIndices = np.argsort(combinedFitnesses)[::-1]  # sort descendingly
        population = [combinedPopulation[i] for i in sortedIndices[:POPULATION_SIZE]]
        fitnesses = [combinedFitnesses[i] for i in sortedIndices[:POPULATION_SIZE]]

        bestFitness = fitnesses[0]
        print(f"Generation {generation+1}: Best Fitness = {bestFitness}")

    bestIndividual = population[0]
    S1 = [node for node in range(V) if bestIndividual[node]]
    S2 = [node for node in range(V) if bestIndividual[V + node]]

    return S1, S2


def main(args):
    start = time.time()
    G = createGraph(args.network)
    seedSet1, seedSet2 = loadSeed(args.initial_seed_set, args.budget)
    S1, S2 = EA(G, seedSet1, seedSet2, args.budget)
    end = time.time()

    print(f"Time: {end - start:.2f}")

    with open(args.balanced_seed_set, "w") as f:
        f.write(f"{len(S1)} {len(S2)}\n")
        for node in S1:
            f.write(f"{node}\n")
        for node in S2:
            f.write(f"{node}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--network",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-i",
        "--initial_seed_set",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-b",
        "--balanced_seed_set",
        type=str,
        required=True,
    )
    parser.add_argument("-k", "--budget", type=int, required=True)

    args = parser.parse_args()
    main(args)


# python IEMP_Evol.py -n C:\Users\aaron\Documents\CS303_AI\project1\Testcase\Evolutionary\map1\dataset1 -i C:\Users\aaron\Documents\CS303_AI\project1\Testcase\Evolutionary\map1\seed -b C:\Users\aaron\Documents\CS303_AI\project1\Testcase\Evolutionary\map1\seed_balanced -k 10
# python IEMP_Evol.py -n C:\Users\aaron\Documents\CS303_AI\project1\Testcase\Evolutionary\map2\dataset2 -i C:\Users\aaron\Documents\CS303_AI\project1\Testcase\Evolutionary\map2\seed -b C:\Users\aaron\Documents\CS303_AI\project1\Testcase\Evolutionary\map2\seed_balanced -k 15
# python IEMP_Evol.py -n C:\Users\aaron\Documents\CS303_AI\project1\Testcase\Evolutionary\map3\dataset3 -i C:\Users\aaron\Documents\CS303_AI\project1\Testcase\Evolutionary\map3\seed -b C:\Users\aaron\Documents\CS303_AI\project1\Testcase\Evolutionary\map3\seed_balanced -k 15
