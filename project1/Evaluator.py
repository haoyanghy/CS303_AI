import argparse
import numpy as np
import networkx as nx
from collections import deque


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
        # if len(seeds1) + len(seeds2) > budget:
        #     raise ValueError("OUt of budget")
    return seeds1, seeds2


def loadBalancedSeed(filePath):
    with open(filePath, "r") as f:
        k1, k2 = map(int, f.readline().split())
        balancedSeeds1 = [int(f.readline().strip()) for _ in range(k1)]
        balancedSeeds2 = [int(f.readline().strip()) for _ in range(k2)]
    return balancedSeeds1, balancedSeeds2


def diffusion(G, seedSet, valueType):
    currentActivated = deque(seedSet)
    returnActivated = set(seedSet)
    exposedNodes = set(seedSet)

    while currentActivated:
        current = currentActivated.popleft()
        for nextV in G.successors(current):
            nextProb = G[current][nextV][valueType]
            exposedNodes.add(nextV)
            if nextV not in returnActivated:
                if np.random.rand() < nextProb:
                    returnActivated.add(nextV)
                    currentActivated.append(nextV)

    return exposedNodes


def monteCarlo(G, seedSet1, seedSet2, TIMES):
    totalPhi = 0.0

    for _ in range(TIMES):
        exposed1 = diffusion(G, seedSet1, "weight1")
        exposed2 = diffusion(G, seedSet2, "weight2")

        symmetricDiff = exposed1 ^ exposed2
        bothORnone = len(G.nodes) - len(symmetricDiff)
        totalPhi += bothORnone

    return totalPhi / TIMES


def main(args):
    TIMES = 500

    G = createGraph(args.network)
    seedSet1, seedSet2 = loadSeed(args.initial_seed_set, args.budget)
    balancedSeed1, balancedSeed2 = loadBalancedSeed(args.balanced_seed_set)
    seedSet1.extend(balancedSeed1)
    seedSet2.extend(balancedSeed2)
    finalValue = monteCarlo(G, seedSet1, seedSet2, TIMES)

    print(f"Final Value: {finalValue:.2f}")
    with open(args.output_path, "w") as f:
        f.write(f"{finalValue:.2f}\n")


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
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        required=True,
    )

    args = parser.parse_args()
    main(args)

# python Evaluator.py -n C:\Users\aaron\Documents\CS303_AI\project1\Testcase\Evaluator\map1\dataset1 -i C:\Users\aaron\Documents\CS303_AI\project1\Testcase\Evaluator\map1\seed -b C:\Users\aaron\Documents\CS303_AI\project1\Testcase\Evaluator\map1\seed_balanced -k 100 -o C:\Users\aaron\Documents\CS303_AI\project1\Testcase\Evaluator\map1\result.txt
# python Evaluator.py -n C:\Users\aaron\Documents\CS303_AI\project1\Testcase\Evaluator\map2\dataset2 -i C:\Users\aaron\Documents\CS303_AI\project1\Testcase\Evaluator\map2\seed -b C:\Users\aaron\Documents\CS303_AI\project1\Testcase\Evaluator\map2\seed_balanced -k 100 -o C:\Users\aaron\Documents\CS303_AI\project1\Testcase\Evaluator\map2\result.txt

# python Evaluator.py -n C:\Users\aaron\Documents\CS303_AI\project1\Testcase\Heuristic\map1\dataset1 -i C:\Users\aaron\Documents\CS303_AI\project1\Testcase\Heuristic\map1\seed -b C:\Users\aaron\Documents\CS303_AI\project1\Testcase\Heuristic\map1\seed_balanced -k 10 -o C:\Users\aaron\Documents\CS303_AI\project1\test_result.txt
# python Evaluator.py -n C:\Users\aaron\Documents\CS303_AI\project1\Testcase\Heuristic\map2\dataset2 -i C:\Users\aaron\Documents\CS303_AI\project1\Testcase\Heuristic\map2\seed -b C:\Users\aaron\Documents\CS303_AI\project1\Testcase\Heuristic\map2\seed_balanced -k 15 -o C:\Users\aaron\Documents\CS303_AI\project1\test_result.txt

# python Evaluator.py -n C:\Users\aaron\Documents\CS303_AI\project1\Testcase\Evolutionary\map1\dataset1 -i C:\Users\aaron\Documents\CS303_AI\project1\Testcase\Evolutionary\map1\seed -b C:\Users\aaron\Documents\CS303_AI\project1\Testcase\Evolutionary\map1\seed_balanced -k 10 -o C:\Users\aaron\Documents\CS303_AI\project1\test_result.txt
