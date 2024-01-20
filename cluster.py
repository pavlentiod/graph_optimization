import itertools
import numpy as np
import matplotlib.pyplot as plt

def calculate_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

def find_combinations(cluster, target_distance):
    num_points = round(len(cluster) / 2- 0.3)
    combinations = []
    max_distance_sum = 0

    # Generate all combinations of points
    for r in range(1, num_points + 1):
        for combination in itertools.combinations(cluster, r):
            distance_sum = sum(calculate_distance(combination[i], combination[i + 1]) for i in range(len(combination) - 1))
            if abs(distance_sum - target_distance) < abs(max_distance_sum - target_distance):
                combinations = [combination]
                max_distance_sum = distance_sum
            elif abs(distance_sum - target_distance) == abs(max_distance_sum - target_distance):
                combinations.append(combination)
    return combinations

# Cluster of points
cluster = np.array([[1.5601864, 1.5599452],
                    [1.81824967, 1.8340451],
                    [3.04613769, 0.97672114],
                    [0.88492502, 1.95982862],
                    [0.45227289, 3.25330331]])

# Target distance
target_distance = 4

# Find combinations of points

def plot_combinations(cluster, target_distance):
    cmb = find_combinations(cluster, target_distance)
    # print(len(cmb))
    # Plot the combinations
    plt.figure()
    plt.title('Point Combinations')
    plt.xlabel('X')
    plt.ylabel('Y')

    for combination in cmb:
        combination = np.array(combination)
        plt.plot(combination[:, 0], combination[:, 1], marker='o', linestyle='-', markersize=5)
    # Plot the original points
    plt.scatter(cluster[:, 0], cluster[:, 1], color='red', marker='x')
    return plt

