import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import pairwise_distances
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.manifold import _utils
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from Clustering_evaluation import NMI, Accuracy, db_index, SC, CH
import time

MACHINE_EPSILON = np.finfo(np.double).eps

def squared_dist_mat(X):
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    return D

def _joint_probabilities(distances, desired_perplexity, verbose):
    distances = distances.astype(np.float32, copy=False)
    conditional_P = _utils._binary_search_perplexity(
        distances, desired_perplexity, verbose
    )
    P = conditional_P + conditional_P.T
    sum_P = np.maximum(np.sum(P), MACHINE_EPSILON)
    P = np.maximum(squareform(P) / sum_P, MACHINE_EPSILON)
    return P

def _kl_divergence(params, P, degrees_of_freedom, n_samples, n_components, skip_num_points=0, compute_error=True):
    X_embedded = params.reshape(n_samples, n_components)
    dist = pdist(X_embedded, "sqeuclidean")
    dist /= degrees_of_freedom
    dist += 1.0
    dist **= (degrees_of_freedom + 1.0) / -2.0
    Q = np.maximum(dist / (2.0 * np.sum(dist)), MACHINE_EPSILON)
    if compute_error:
        kl_divergence = np.dot(P, np.log(np.maximum(P, MACHINE_EPSILON) / Q))
    else:
        kl_divergence = np.nan
    grad = np.ndarray((n_samples, n_components), dtype=params.dtype)
    PQd = squareform((P - Q) * dist)
    for i in range(skip_num_points, n_samples):
        grad[i] = np.dot(np.ravel(PQd[i], order="K"), X_embedded[i] - X_embedded)
    grad = grad.ravel()
    c = 2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom
    grad *= c
    return kl_divergence, grad

def objective_function(params, P, degrees_of_freedom, n_samples, n_components, compute_error=True):
    return _kl_divergence(params, P, degrees_of_freedom, n_samples, n_components, compute_error=True)

def initialize_population(X, n_components, perplexity, population_size):
    population = np.array([TSNE(n_components=n_components, perplexity=perplexity).fit_transform(X) for _ in range(population_size)])
    return population

def calculate_fitness(population, P, degrees_of_freedom, n_samples, n_components):
    fitness = np.zeros(population.shape[0])
    for i in range(population.shape[0]):
        fitness[i], _ = objective_function(population[i], P, degrees_of_freedom, n_samples, n_components)
    return fitness

def selection(population, fitness, population_size):
    # Normalize fitness values to ensure they sum to 1
    fitness_sum = np.sum(fitness)
    if fitness_sum == 0:
        fitness_sum = 1
    fitness = fitness / fitness_sum

    #print("Shape of population:", population.shape)
    #print("Shape of fitness:", fitness.shape)

    selected_indices = np.random.choice(np.arange(len(population)), size=population_size, p=fitness)
    return population[selected_indices]

def crossover(parent1, parent2):
    n_samples, n_components = parent1.shape
    crossover_point = np.random.randint(1, n_samples-1)
    child1 = np.vstack((parent1[:crossover_point, :], parent2[crossover_point:, :]))
    child2 = np.vstack((parent2[:crossover_point, :], parent1[crossover_point:, :]))
    return child1, child2

def mutation(offspring, mutation_rate=0.1):
    for i in range(offspring.shape[0]):
        if np.random.rand() < mutation_rate:
            mutation_idx = np.random.randint(offspring.shape[1])
            offspring[i, mutation_idx] += np.random.randn()
    return offspring

def genetic_algorithm(X, n_components, verbose, perplexity, n_neighbors, population_size, max_iter,
                      mutation_rate=0.1, crossover_rate=0.9):
    n_samples = X.shape[0]
    distances = pairwise_distances(X, metric="euclidean")
    P = _joint_probabilities(distances, perplexity, verbose)
    degrees_of_freedom = 1

    population = initialize_population(X, n_components, perplexity, population_size)
    best_fitness = float('inf')
    best_solution = None

    for iteration in tqdm(range(max_iter), "Iterations"):
        fitness = calculate_fitness(population, P, degrees_of_freedom, n_samples, n_components)

        if np.min(fitness) < best_fitness:
            best_fitness = np.min(fitness)
            best_solution = population[np.argmin(fitness)]

        selected_population = selection(population, fitness, population_size)

        new_population = []
        for i in range(0, population_size, 2):
            parent1, parent2 = selected_population[i], selected_population[(i + 1) % population_size]
            child1, child2 = crossover(parent1, parent2)
            new_population.append(child1)
            new_population.append(child2)

        population = np.array(new_population)
        population = mutation(population, mutation_rate)

    return best_solution

# Load dataset and scale
digits = load_digits()
d = scale(digits.data)
y = digits.target

# Parameters for GA
population_size = 10
max_iter = 50
mutation_rate = 0.1
crossover_rate = 0.9

# Run GA-based t-SNE
NMIRecord = []
ACCRecord = []
DBRecord = []
SCRecord = []
TimeRecord = []

for i in range (5):
    #if i % 5 == 0:
    print(i)
    start_time=time.time()
    gbest = genetic_algorithm(d, 2, verbose=1, perplexity=15, n_neighbors=30, population_size=population_size,
                              max_iter=max_iter, mutation_rate=mutation_rate, crossover_rate=crossover_rate)
    end_time=time.time()-start_time

    km = KMeans(n_clusters=10)
    prediction = km.fit_predict(gbest)

    TimeRecord.append(end_time)
    ACCRecord.append(Accuracy(y, prediction))
    NMIRecord.append(NMI(y, prediction))
    SCRecord.append(SC(gbest, prediction))
    DBRecord.append(db_index(gbest, prediction))

    #CHRecord.append(CH(X_embedded, prediction))

print("Execution time ", np.mean(TimeRecord), "std: ", np.std(TimeRecord))
print("Accuracy: ",np.mean(ACCRecord), "std: ", np.std(ACCRecord))
print("NMI: ",np.mean(NMIRecord), "std: ", np.std(NMIRecord))
print("silhouette score: ", np.mean(SCRecord), "std: ", np.std(SCRecord))
print("DBI: ", np.mean(DBRecord), "std: ", np.std(DBRecord))

# Visualize the result
df = pd.DataFrame(gbest)
df['target'] = y
df['x'] = gbest[:, 0]
df['y'] = gbest[:, 1]
plt.figure(figsize=(12, 8))
sns.scatterplot(x='x', y='y', hue='target', palette=sns.color_palette("hsv", 10), data=df)
plt.savefig("t-SNE-GA-vis.png")
