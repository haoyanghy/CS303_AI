import argparse
import time
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
    return seeds1, seeds2


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

    return exposedNodes, returnActivated


def greedyBestFirstSearch(G, seedSet1, seedSet2, budget, TIMES):
    N = len(G.nodes)
    S1, S2 = [], []
    h1Values = np.zeros(N)
    h2Values = np.zeros(N)

    while len(S1) + len(S2) < budget:

        for i in range(TIMES):
            # print(f"{i} length: ", len(S1) + len(S2))
            exposed1, activated1 = diffusion(G, seedSet1, "weight1")
            exposed2, activated2 = diffusion(G, seedSet2, "weight2")

            symmetricDiff = exposed1 ^ exposed2
            bothORnone = N - len(symmetricDiff)

            for node in G.nodes:
                if node not in activated1:
                    exposed1_v, activated1_v = diffusion(G, {node}, "weight1")
                    symmetricDiff1_v = exposed1_v.union(exposed1) ^ exposed2
                    bothORnone1_v = N - len(symmetricDiff1_v)
                    h1Values[node] += bothORnone1_v - bothORnone
                else:
                    continue

            for node in G.nodes:
                if node not in activated2:
                    exposed2_v, activated2_v = diffusion(G, {node}, "weight2")
                    symmetricDiff2_v = exposed2_v.union(exposed2) ^ exposed1
                    bothORnone2_v = N - len(symmetricDiff2_v)
                    h2Values[node] += bothORnone2_v - bothORnone
                else:
                    continue

        h1Values /= TIMES
        h2Values /= TIMES

        if np.max(h1Values) > np.max(h2Values):
            bestNode = np.argmax(h1Values)
            S1.append(bestNode)
            seedSet1.append(bestNode)
            h1Values[bestNode] = -np.inf
        else:
            bestNode = np.argmax(h2Values)
            S2.append(bestNode)
            seedSet2.append(bestNode)
            h2Values[bestNode] = -np.inf

        # print(S1)
        # print(S2)

    return S1, S2


def main(args):
    start = time.time()
    TIMES = 2
    G = createGraph(args.network)
    seedSet1, seedSet2 = loadSeed(args.initial_seed_set, args.budget)
    S1, S2 = greedyBestFirstSearch(G, seedSet1, seedSet2, args.budget, TIMES)
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

# python IEMP_Heur.py -n C:\Users\aaron\Documents\CS303_AI\project1\Testcase\Heuristic\map1\dataset1 -i C:\Users\aaron\Documents\CS303_AI\project1\Testcase\Heuristic\map1\seed -b C:\Users\aaron\Documents\CS303_AI\project1\Testcase\Heuristic\map1\seed_balanced -k 10
# python IEMP_Heur.py -n C:\Users\aaron\Documents\CS303_AI\project1\Testcase\Heuristic\map2\dataset2 -i C:\Users\aaron\Documents\CS303_AI\project1\Testcase\Heuristic\map2\seed -b C:\Users\aaron\Documents\CS303_AI\project1\Testcase\Heuristic\map2\seed_balanced -k 15
# python IEMP_Heur.py -n C:\Users\aaron\Documents\CS303_AI\project1\Testcase\Heuristic\map3\dataset3 -i C:\Users\aaron\Documents\CS303_AI\project1\Testcase\Heuristic\map3\seed -b C:\Users\aaron\Documents\CS303_AI\project1\Testcase\Heuristic\map3\seed_balanced -k 15
