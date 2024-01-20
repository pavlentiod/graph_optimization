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


clust = [(array([0.21900583, 0.39355051]), array([0.06576036, 0.42246549]), array([0.16786326, 0.63323533])),
         (array([0.16982689, 0.74491166]), array([0.34321507, 0.89535719]), array([0.38393874, 0.71927355])),
         (array([0.69256899, 0.50525033]), array([0.91209316, 0.48424888]), array([0.93856181, 0.30524044])),
         (array([0.01932954, 0.11562687]), array([0.36286632, 0.09007086]), array([0.41483319, 0.00808337])),
         (array([0, 0]), array([0, 0.1]), array([0.2, 0.]))]

cmb = find_combinations(clust)


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


good_seq = find_sequences_with_desired_angles(ceq_duets(cmb))


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


filt_seq = generate_new_sequences(good_seq)

longest_seqs = select_longest_subsequence(filt_seq)
n =0
plot_sequences(longest_seqs[n:n+1])
