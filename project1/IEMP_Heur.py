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
    S1, S2 = [], []
    h1Values = {}
    h2Values = {}
    h1 = 0
    h2 = 0
    while len(S1) + len(S2) <= budget:
        for _ in range(TIMES):
            print(len(S1) + len(S2))
            exposed1, activated1 = diffusion(G, seedSet1, "weight1")
            exposed2, activated2 = diffusion(G, seedSet2, "weight2")

            symmetricDiff = exposed1 ^ exposed2
            bothORnone = len(G.nodes) - len(symmetricDiff)
            for node in G.nodes:
                if node in activated1:
                    continue
                else:
                    exposed1_v, activated1_v = diffusion(G, {node}, "weight1")
                if node in activated2:
                    continue
                else:
                    exposed2_v, activated2_v = diffusion(G, {node}, "weight2")

                symmetricDiff1_v = exposed1_v ^ exposed2
                bothORnone1_v = len(G.nodes) - len(symmetricDiff1_v)
                h1 += bothORnone1_v - bothORnone

                symmetricDiff2_v = exposed2_v ^ exposed1
                bothORnone2_v = len(G.nodes) - len(symmetricDiff2_v)
                h2 += bothORnone2_v - bothORnone
            h1Values[node] = h1 / TIMES
            h2Values[node] = h2 / TIMES

        S1.append(max(h1Values, key=h1Values.get))
        S2.append(max(h2Values, key=h2Values.get))
        print(S1)
        print(S2)

    return S1, S2


def main(args):

    TIMES = 10

    G = createGraph(args.network)
    seedSet1, seedSet2 = loadSeed(args.initial_seed_set, args.budget)

    # finalValue = monteCarlo(G, seedSet1, seedSet2, TIMES)

    S1, S2 = greedyBestFirstSearch(G, seedSet1, seedSet2, args.budget, TIMES)

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
