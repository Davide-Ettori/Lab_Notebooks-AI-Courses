# Authors: Jessica Su, Wanzi Zhou, Pratyaksh Sharma, Dylan Liu, Ansh Shukla
#Modified: Alex Porter
import numpy as np
import random
import time
import pdb
import unittest
from PIL import Image
import os
import matplotlib.pyplot as plt

# Finds the L1 distance between two vectors
# u and v are 1-dimensional np.array objects
def l1(u, v):
    return np.sum(np.abs(u - v))

# Loads the data into a np array, where each row corresponds to
# an image patch -- this step is sort of slow.
# Each row in the data is an image, and there are 400 columns.
def load_data(filename):
    return np.genfromtxt(filename, delimiter=',')

# Creates a hash function from a list of dimensions and thresholds.
def create_function(dimensions, thresholds):
    def f(v):
        boolarray = [v[dimensions[i]] >= thresholds[i] for i in range(len(dimensions))]
        return "".join(map(str, map(int, boolarray)))
    return f

# Creates the LSH functions (functions that compute L K-bit hash keys).
# Each function selects k dimensions (i.e. column indices of the image matrix)
# at random, and then chooses a random threshold for each dimension, between 0 and
# 255.  For any image, if its value on a given dimension is greater than or equal to
# the randomly chosen threshold, we set that bit to 1.  Each hash function returns
# a length-k bit string of the form "0101010001101001...", and the L hash functions 
# will produce L such bit strings for each image.
def create_functions(k, L, num_dimensions=400, min_threshold=0, max_threshold=255):
    functions = []
    for i in range(L):
        dimensions = np.random.randint(low = 0, 
                                   high = num_dimensions,
                                   size = k)
        thresholds = np.random.randint(low = min_threshold, 
                                   high = max_threshold + 1, 
                                   size = k)

        functions.append(create_function(dimensions, thresholds))
    return functions

# Hashes an individual vector (i.e. image).  This produces an array with L
# entries, where each entry is a string of k bits.
def hash_vector(functions, v):
    return np.array([f(v) for f in functions])

# Hashes the data in A, where each row is a datapoint, using the L
# functions in "functions."
def hash_data(functions, A):
    return np.array(list(map(lambda v: hash_vector(functions, v), A)))

# Retrieve all of the points that hash to one of the same buckets 
# as the query point.  Do not do any random sampling (unlike what the first
# part of this problem prescribes).
# Don't retrieve a point if it is the same point as the query point.
def get_candidates(hashed_A, hashed_point, query_index):
    return filter(lambda i: i != query_index and \
        any(hashed_point == hashed_A[i]), range(len(hashed_A)))

# Sets up the LSH.  You should try to call this function as few times as 
# possible, since it is expensive.
# A: The dataset in which each row is an image patch.
# Return the LSH functions and hashed data structure.
def lsh_setup(A, k = 24, L = 10):
    functions = create_functions(k = k, L = L)
    hashed_A = hash_data(functions, A)
    return (functions, hashed_A)

# Run the entire LSH algorithm
def lsh_search(A, hashed_A, functions, query_index, num_neighbors = 10):
    hashed_point = hash_vector(functions, A[query_index, :])
    candidate_row_nums = get_candidates(hashed_A, hashed_point, query_index)
    
    distances = map(lambda r: (r, l1(A[r], A[query_index])), candidate_row_nums)
    best_neighbors = sorted(distances, key=lambda t: t[1])[:num_neighbors]

    return [t[0] for t in best_neighbors]

# Plots images at the specified rows and saves them each to files.
def plot(A, row_nums, base_filename):
    for row_num in row_nums:
        patch = np.reshape(A[row_num, :], [20, 20])
        im = Image.fromarray(patch)
        if im.mode != 'RGB':
            im = im.convert('RGB')
        im.save(base_filename + "-" + str(row_num) + ".png")

# Finds the nearest neighbors to a given vector, using linear search.
def linear_search(A, query_index, num_neighbors):
    distances = [(i, l1(A[query_index], A[i])) for i in range(len(A)) if i != query_index]
    distances.sort(key=lambda x: x[1])
    return [index for index, _ in distances[:num_neighbors]]

def compute_error(x1, x2, v): # x1 is prediction and x2 is ground truth
    total_error = 0
    for vec1, vec2, base in zip(x1, x2, v):
        dist1 = np.sum([l1(elem, base) for elem in vec1])
        dist2 = np.sum([l1(elem, base) for elem in vec2])
        total_error += dist1 / dist2
    return total_error / len(v)

def point_1():
    indices = list(range(100, 1100, 100))
    num_neighbors = 3

    # LSH setup
    functions, hashed_A = lsh_setup(A)
    
    lsh_times = []
    linear_times = []
    lsh_neighbors = []
    linear_neighbors = []

    for index in indices:
        # LSH search
        start_time = time.time()
        lsh_result = lsh_search(A, hashed_A, functions, index, num_neighbors)
        lsh_times.append(time.time() - start_time)
        lsh_neighbors.append(lsh_result)

        # Linear search
        start_time = time.time()
        linear_result = linear_search(A, index, num_neighbors)
        linear_times.append(time.time() - start_time)
        linear_neighbors.append(linear_result)

    avg_lsh_time = np.mean(lsh_times)
    avg_linear_time = np.mean(linear_times)

    print(f"Average LSH search time: {avg_lsh_time:.6f} s")
    print(f"Average linear search time: {avg_linear_time:.6f} s")

    '''for i, index in enumerate(indices):
        print(f"Patch {index}:")
        print(f"  LSH neighbors: {lsh_neighbors[i]}")
        print(f"  Linear neighbors: {linear_neighbors[i]}")'''
    


def point_2():
    indices = list(range(100, 1100, 100))
    L_values = list(range(10, 22, 2))
    num_neighbors = 3

    errors = []

    for L in L_values:
        functions, hashed_A = lsh_setup(A, k=24, L=L)
        lsh_neighbors = []
        linear_neighbors = []

        for index in indices:
            lsh_result = lsh_search(A, hashed_A, functions, index, num_neighbors)
            linear_result = linear_search(A, index, num_neighbors)
            lsh_neighbors.append(lsh_result)
            linear_neighbors.append(linear_result)

        error = compute_error(lsh_neighbors, linear_neighbors, [A[i] for i in indices])
        errors.append(error)

    print("\nClose the plot to continue...")
    plt.plot(L_values, errors, marker='o')
    plt.xlabel('L')
    plt.ylabel('Error')
    plt.title('Error as a function of L')
    plt.grid(True)
    plt.show()
    print("\nPlot closed")

    K_values = list(range(16, 26, 2))
    errors = []

    for K in K_values:
        functions, hashed_A = lsh_setup(A, k=K, L=10)
        lsh_neighbors = []
        linear_neighbors = []

        for index in indices:
            lsh_result = lsh_search(A, hashed_A, functions, index, num_neighbors)
            linear_result = linear_search(A, index, num_neighbors)
            lsh_neighbors.append(lsh_result)
            linear_neighbors.append(linear_result)

        error = compute_error(lsh_neighbors, linear_neighbors, [A[i] for i in indices])
        errors.append(error)
        
    print("\nClose the plot to continue...")
    plt.plot(K_values, errors, marker='o')
    plt.xlabel('K')
    plt.ylabel('Error')
    plt.title('Error as a function of K')
    plt.grid(True)
    plt.show()
    print("\nPlot closed")

def point_3():
    query_index = 100
    num_neighbors = 10

    functions, hashed_A = lsh_setup(A, k=24, L=10)
    lsh_neighbors = lsh_search(A, hashed_A, functions, query_index, num_neighbors)
    linear_neighbors = linear_search(A, query_index, num_neighbors)

    common_neighbors = set(lsh_neighbors).intersection(set(linear_neighbors))
    percentage_common = (len(common_neighbors) / num_neighbors) * 100
    print(f"\nPercentage of common neighbors: {percentage_common:.2f}%")

    output_dir = "imgs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for file in os.listdir(output_dir):
        file_path = os.path.join(output_dir, file)
        if os.path.isfile(file_path):
            os.unlink(file_path)
    plot(A, [query_index] + lsh_neighbors, "imgs/lsh_neighbors")
    plot(A, [query_index] + linear_neighbors, "imgs/linear_neighbors")

# TODO: Solve Problem 4
def problem4():
    point_1()
    point_2()
    point_3()

    print("\n\nProblem number 4 is solved\n")

#### TESTS #####

class TestLSH(unittest.TestCase):
    def test_l1(self):
        u = np.array([1, 2, 3, 4])
        v = np.array([2, 3, 2, 3])
        self.assertEqual(l1(u, v), 4)

    def test_hash_data(self):
        f1 = lambda v: sum(v)
        f2 = lambda v: sum([x * x for x in v])
        A = np.array([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(f1(A[0,:]), 6)
        self.assertEqual(f2(A[0,:]), 14)

        functions = [f1, f2]
        self.assertTrue(np.array_equal(hash_vector(functions, A[0, :]), np.array([6, 14])))
        self.assertTrue(np.array_equal(hash_data(functions, A), np.array([[6, 14], [15, 77]])))

    ### TODO: Write your tests here (they won't be graded, 
    ### but you may find them helpful)

A = None
if __name__ == '__main__':
    #unittest.main()

    if not os.path.exists("patches.npy"):
        A = load_data("data/patches.csv")
        np.save("patches.npy", A)

    A = np.load("patches.npy")
    problem4()