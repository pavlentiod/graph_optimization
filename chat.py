import numpy as np
from numpy import array
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from itertools import permutations
from scipy.spatial import distance
import itertools
import math
from scipy.spatial import distance
from itertools import combinations
import matplotlib.image as mpimg

def find_combinations(sequences):
    combinations = []
    num_sequences = len(sequences)

    # Iterate over the sequences
    for i in range(num_sequences):
        for j in range(num_sequences):
            # Skip if the same sequence is selected
            if i == j:
                continue

            # Get the first and last elements from the sequences
            first_element = sequences[i][0]
            last_element = sequences[j][-1]

            # Add the combination to the list
            combinations.append((first_element, last_element))

    return combinations



def ceq_duets(cmb):
    new_seq = []
    for line in cmb:
        point1, point2 = line
        for seq in clust:
            if list(seq[0]) == list(point1):
                seq1 = seq
            if list(seq[-1]) == list(point2):
                seq2 = seq
        new_seq.append(seq1 + seq2)
    return new_seq


def find_sequences_with_desired_angles(sequences):
    result = []

    for sequence in sequences:
        angles = []

        # Calculate angles between consecutive points in the sequence
        for i in range(len(sequence) - 2):
            vector1 = sequence[i + 1] - sequence[i]
            vector2 = sequence[i + 2] - sequence[i + 1]

            # Calculate the angle between the two vectors
            angle = np.arccos(np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2)))

            # Convert the angle from radians to degrees
            angle_deg = np.degrees(angle)

            angles.append(angle_deg)

        # Check if all angles are within the desired range
        if all(40 <= angle <= 150 for angle in angles):
            result.append(sequence)

    return result


def max_sum_sequence(sequences):
    max_combination = []
    max_sum = 0

    for combination in itertools.combinations(sequences, 4):
        flattened_points = [tuple(point) for point in combination]
        unique_points = []
        if len(unique_points) != 9:
            continue

        angles = []
        for i in range(len(combination) - 2):
            v1 = combination[i + 1] - combination[i]
            v2 = combination[i + 2] - combination[i + 1]
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle = np.arccos(cos_angle) * 180 / np.pi
            angles.append(angle)

        if all(40 <= angle <= 150 for angle in angles):
            sequence_sum = sum(combination, np.array([0, 0]))[-1]
            if sequence_sum > max_sum:
                max_combination = combination
                max_sum = sequence_sum

    return max_combination, max_sum



# print(good_seq)


def get_sequence_intersection(seq1, seq2):
    intersection = []

    for point in seq1:
        if any(np.all(point == p) for p in seq2) and not any(np.all(point == p) for p in intersection):
            intersection.append(point)
    return intersection


# [print(len(i)) for i in good_seq]

def generate_new_sequences(sequences):
    new_sequences = []

    for i in range(len(sequences)):
        for j in range(i + 1, len(sequences)):
            sequence1 = sequences[i]
            sequence2 = sequences[j]

            # Check if sequences have at least one common point
            if len(get_sequence_intersection(sequence1, sequence2)) > 0:
                combined_sequence = np.concatenate((sequence1, sequence2))
                unique_points = list(set(map(tuple, combined_sequence)))

                angles = []
                for k in range(len(combined_sequence) - 2):
                    v1 = combined_sequence[k + 1] - combined_sequence[k]
                    v2 = combined_sequence[k + 2] - combined_sequence[k + 1]
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    angle = np.arccos(cos_angle) * 180 / np.pi
                    angles.append(angle)

                # Check if angles are within the range of 50 to 150 degrees
                if all(40 <= angle <= 160 for angle in angles):
                    new_sequences.append(combined_sequence)
    return new_sequences



def plot_sequences(sequences):
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    # Цвета для каждой последовательности
    plt.figure()

    # Загрузка изображения из файла
    img = mpimg.imread('иноземцево_дист.jpg')
    plt.imshow(img, extent=[0, 1, 0, 1])

    for i, sequence in enumerate(sequences):
        x = [point[0] for point in sequence]
        y = [point[1] for point in sequence]

        # Соединяем точки линиями
        plt.plot(x, y, color=colors[i], linestyle='-', marker='o')

        # Добавляем красную звездочку и соединяем ее с первой точкой последовательности
        plt.plot([x[0], 0.08], [y[0], 0.08], color='red', linestyle='-', marker='')
        plt.plot([y[0], 0.08],color='red', marker='^', markersize=10)
        # Добавляем красную звездочку и соединяем ее с последней точкой последовательности
        plt.plot([x[-1], 0.8], [y[-1], 0.1], color='red', linestyle='-', marker='*')

    # plt.xlabel('X')
    # plt.ylabel('Y')
    #
    # plt.title('Последовательности с соединенными точками')
    plt.show()



def select_longest_subsequence(sequences):
    longest_subsequences = []

    for sequence in sequences:
        unique_points = []
        longest_subsequence = []

        for point in sequence:
            # Check if point is already used in the subsequence
            if point.tolist() not in unique_points:
                unique_points.append(point.tolist())
                longest_subsequence.append(point)

        longest_subsequences.append(longest_subsequence)

    return longest_subsequences

def generate_points():
    points = np.random.rand(30, 2)  # Генерация 30 случайных точек в диапазоне [0, 1)
    return points


def cluster_points(points):
    kmeans = KMeans(n_clusters=5, random_state=0)
    clusters = kmeans.fit_predict(points)
    return clusters


def calculate_angle(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2
    dot_product = np.dot(v1, v2)
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    angle = np.arccos(dot_product / norm_product)
    return np.degrees(angle)


def find_optimal_routes(points, clusters, target_distance):
    optimal_routes = []

    for cluster_id in range(5):
        cluster_points = points[clusters == cluster_id]
        num_points = len(cluster_points)

        if num_points < 3:
            continue

        min_distance = np.inf
        optimal_route = None

        # Перебор всех возможных комбинаций точек
        for route in permutations(cluster_points, 4):
            current_distance = sum(distance.euclidean(route[i], route[i + 1]) for i in range(3))
            angle1 = calculate_angle(route[0], route[1], route[2])
            angle2 = calculate_angle(route[1], route[2], route[3])

            # Проверка на близость к требуемой сумме расстояний и внутренним углам
            if (abs(current_distance - target_distance) < abs(min_distance - target_distance) and
                    50 <= angle1 <= 150 and 50 <= angle2 <= 150):
                min_distance = current_distance
                optimal_route = route[:3]  # Берем только первые 3 точки для маршрута

        if optimal_route is not None:
            optimal_routes.append(optimal_route)

    return optimal_routes


def calculate_total_lengths(sequences):
    total_lengths = []

    for sequence in sequences:
        lengths = [distance.euclidean(sequence[i], sequence[i + 1]) for i in range(2)]
        total_length = sum(lengths)
        total_lengths.append(total_length)

    return total_lengths


def plot_routes(points, clusters, optimal_routes):
    colors = ['b', 'g', 'r', 'c', 'm']

    for cluster_id in range(5):
        cluster_points = points[clusters == cluster_id]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors[cluster_id])

        if cluster_id < len(optimal_routes):
            route = optimal_routes[cluster_id]
            route = np.array(route)
            plt.plot(route[:, 0], route[:, 1], color=colors[cluster_id])

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Optimal Routes')
    plt.show()




# def find_combination(seq, target_sum):

def run(total_dist):
    points = generate_points()
    clusters = cluster_points(points)
    optimal_routes = find_optimal_routes(points, clusters, target_distance=0.7)
    clusters_sum = sum(calculate_total_lengths(optimal_routes))
    good_seq = find_sequences_with_desired_angles(ceq_duets(cmb))
    filt_seq = generate_new_sequences(good_seq)
    longest_seqs = select_longest_subsequence(filt_seq)
    n = 0
    plot_sequences(longest_seqs[n:n + 1])
    target_sum = total_dist - clusters_sum
    # combination = find_combination(optimal_routes, target_sum)
    # plot_routes(points, clusters, optimal_routes)


# Запуск программы
run(15)
