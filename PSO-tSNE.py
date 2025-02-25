import umap
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import _utils
from sklearn.metrics.pairwise import pairwise_distances
from Clustering_evaluation import NMI, Accuracy, db_index, SC
from sklearn.cluster import KMeans
import json
MACHINE_EPSILON = np.finfo(np.double).eps
from time import time
import os

os.makedirs("data/t-SNE-PSO", exist_ok=True)
os.makedirs("data/t-SNE", exist_ok=True)
os.makedirs("data/UMAP", exist_ok=True)
os.makedirs("Results", exist_ok=True)

def squared_dist_mat(X):
    """calculates the squared eucledian distance matrix

    function source: https://lvdmaaten.github.io/tsne/
    Parameters:
    X : ndarray of shape (n_samples, n_features)

    Returns:
    D: Squared eucledian distance matrix of shape (n_samples, n_samples)

    """
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


def _kl_divergence(
        params,
        P,
        degrees_of_freedom,
        n_samples,
        n_components,
        skip_num_points=0,
        compute_error=True):
    X_embedded = params.reshape(n_samples, n_components)

    # Q is a heavy-tailed distribution: Student's t-distribution
    dist = pdist(X_embedded, "sqeuclidean")  # (||yi-yj||**2)
    dist /= degrees_of_freedom
    dist += 1.0  # (1+||yi-yj||**2)
    dist **= (degrees_of_freedom + 1.0) / -2.0  # (1+||yi-yj||**2)^^-1, which is degrees of freedom=1, so 1+1/-2=-1
    Q = np.maximum(dist / (2.0 * np.sum(dist)), MACHINE_EPSILON)  # (1+||yi-yj||**2)^-1/sum((1+||yk-yl||**2)^-1
    # Optimization trick below: np.dot(x, y) is faster than
    # np.sum(x * y) because it calls BLAS
    # compute_error=True
    # Objective: C (Kullback-Leibler divergence of P and Q)
    if compute_error:
        kl_divergence = np.dot(P, np.log(np.maximum(P, MACHINE_EPSILON) / Q))
    else:
        kl_divergence = np.nan
    #     print("kl_divergence")
    #     print(kl_divergence)
    grad = np.ndarray((n_samples, n_components), dtype=params.dtype)
    PQd = squareform((P - Q) * dist)
    for i in range(skip_num_points, n_samples):
        grad[i] = np.dot(np.ravel(PQd[i], order="K"), X_embedded[i] - X_embedded)
    grad = grad.ravel()
    c = 2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom
    grad *= c

    return kl_divergence, grad


def objective_function(params, P, degrees_of_freedom, n_samples, n_components, compute_error=True, ):
    return _kl_divergence(params,
                          P,
                          degrees_of_freedom,
                          n_samples,
                          n_components,
                          compute_error=True)

def _gradient_descent(
        p0,
        P, degrees_of_freedom, n_samples, n_components,
        momentum=0.8,
        learning_rate=200.0,
        min_gain=0.01,
):
    p = p0.copy().ravel()
    update = np.zeros_like(p)
    gains = np.ones_like(p)

    error, grad = _kl_divergence(p, P, degrees_of_freedom, n_samples, n_components, compute_error=True)

    inc = update * grad < 0.0
    dec = np.invert(inc)
    gains[inc] += 0.2
    gains[dec] *= 0.8
    np.clip(gains, min_gain, np.inf, out=gains)
    grad *= gains
    update = momentum * update - learning_rate * grad
    p += update

    return p, error

def pso(X, n_components, verbose, perplexity, num_particles, max_iter, h, f, w, compute_error=True):
    n_samples = X.shape[0]
    distances = pairwise_distances(X, metric="euclidean")
    # distances = squared_dist_mat(X)
    P = _joint_probabilities(distances, perplexity, verbose)
    print("shape of P", np.shape(P))
    degrees_of_freedom = 1
    EPSILON = 1e-12
    best_positions = np.zeros((n_samples, n_components))  # best particle positions pbest positions #p
    best_fitness = np.zeros(num_particles)  # best particle function values pbest values #fp
    swarm_best_position = []  # best swarm position gbest positions #g
    swarm_best_fitness = 1e100  # artificial best swarm position starting value gbest values #fg
    particles = []
    debug = False
    #w = 1
    c1 = 1
    c2 = 1

    # Initialize particles and velocities
    for i in range(num_particles):
        #particle = TSNE(n_components=n_components, perplexity=perplexity).fit_transform(X)
        particle = umap.UMAP(n_neighbors=perplexity, min_dist=0.0, n_components=n_components).fit_transform(X)

        best_fitness[i], grad = objective_function(particle, P, degrees_of_freedom, n_samples, n_components, compute_error=True, )

        #grad = grad.reshape(n_samples, n_components)
        velocities = particle  # rand.multivariate_normal(np.zeros(n_components), np.identity(n_components)*10e-4 ,size=n_samples)#*10 # Initialize the particle's velocity
        # Initialize the best positions and fitness values
        best_positions = np.copy(particle)  # pbest positions
          # pbest value

        if i == 0:
            swarm_best_position = best_positions.copy()

        if best_fitness[i] < swarm_best_fitness:
            swarm_best_fitness = best_fitness[i]

            swarm_best_position = best_positions.copy()

        particles.append({
            'Y': np.copy(particle),
            'Yold': np.copy(particle),
            'velocity': np.copy(velocities),
            'pbest': np.copy(best_positions),
            'pbest_value': best_fitness,
            'gbest': np.copy(swarm_best_position),
            'gbest_value': swarm_best_fitness
        })

    # Iterate until termination criterion met ##################################
    it = 1
    for it in tqdm(range(max_iter), "Iterations"):
        rp = np.random.uniform(0, 1)  # np.random.rand(1)
        rg = np.random.uniform(0, 1)  # np.random.rand(1)
        i = 0
        for p in particles:
            # Update the particle's velocity
            p['velocity'] = w * p['velocity'] + c1 * rp * (p['pbest'] - p['Y']) + c2 * rg * (p['gbest'] - p['Y'])

            #params = p['Y'].ravel()
            #params, fitness = _gradient_descent(params, P, degrees_of_freedom, n_samples, n_components,momentum=0.8, learning_rate=200.0, min_gain=0.01,)
            #p['Y'] = params.reshape(n_samples, n_components)
            p['Y'] = p['Y'] + p['velocity']

            fitness, grad = objective_function(p['Y'], P, degrees_of_freedom, n_samples, n_components, compute_error=True, )
            # grad = grad.reshape(n_samples, n_components)

            if fitness < p['pbest_value'][i]:  # and is_feasible(Y[i, :]):
                p['pbest'] = p['Y'].copy()
                p['pbest_value'][i] = fitness

                if fitness < swarm_best_fitness:
                    if debug:
                        print('New best for swarm at iteration {:}: {:} {:}'.format(it, p['Y'], fitness))

                    tmp = p['Y'].copy()

                    stepsize = np.sqrt(np.sum((p['gbest'] - tmp) ** 2))
                    if np.abs(swarm_best_fitness - fitness) <= EPSILON:
                        print('Stopping search: Swarm best objective change less than {:}'.format(EPSILON))
                        return tmp, fitness
                    elif stepsize <= EPSILON:
                        print('Stopping search: Swarm best position change less than {:}'.format(EPSILON))
                        return tmp, fitness
                    else:
                        swarm_best_position = tmp.copy()
                        swarm_best_fitness = fitness

            i += 1
        if debug:
            print('Best after iteration {:}: {:} {:}'.format(it, swarm_best_position, swarm_best_fitness))

        it += 1
        c1 = h - (h / (1 + (f / it)))
        c2 = h / (1 + (f / it))

    print('Stopping search: maximum iterations reached --> {:}'.format(max_iter))
    return swarm_best_position

from keras.datasets import mnist,fashion_mnist

digits = load_digits()
d = scale(digits.data)
#d=d[:180]
y = digits.target
#y=y[:180]
for i in range(5, 100, 5):
    print(i)
    start_time=time()
    gbest = pso(d, n_components=3, compute_error=True, verbose=1, perplexity=i, 
                num_particles=5, max_iter=100, h=1e-20, f=1e-21, w=1e-20)
    end_time=time()-start_time
    # Save results as JSON
    data = {
        "x": gbest[:, 0].tolist(),
        "y": gbest[:, 1].tolist(),
        "z": gbest[:, 2].tolist(),
        "labels": y.tolist()
    }
    with open(f"data/t-SNE-PSO/tsne_pso_perp{i}.json", "w") as f:
        json.dump(data, f)

    km = KMeans(n_clusters=10)
    prediction = km.fit_predict(gbest)

    print("NMI: ",NMI(y, prediction))
    print("Accuracy: ",Accuracy(y, prediction))
    print("DBI: ", db_index(gbest, prediction))
    print("silhouette score: ", SC(gbest, prediction))
    print("Execution time: ", end_time)

    df = pd.DataFrame(gbest)
    df['target'] = y
    df['x'] = gbest[:, 0]
    df['y'] = gbest[:, 1]
    df['z'] = gbest[:, 2]
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='x', y='y', hue='target', palette=sns.color_palette("hsv", 10), data=df)
    plt.savefig(f"Results/PenDigit_t-SNE_vis_perp{i}.png")

    tsne_results = TSNE(n_components=3, perplexity=i).fit_transform(d)
    tsne_data = {
        "x": tsne_results[:, 0].tolist(),
        "y": tsne_results[:, 1].tolist(),
        "z": tsne_results[:, 2].tolist(),
        "labels": y.tolist()
    }
    with open(f"data/t-SNE/tsne_perp{i}.json", "w") as f:
        json.dump(tsne_data, f)

    print(f"Generating UMAP data for perplexity {i}")
    umap_results = umap.UMAP(n_neighbors=i, min_dist=0.1, n_components=3).fit_transform(d)
    umap_data = {
        "x": umap_results[:, 0].tolist(), 
        "y": umap_results[:, 1].tolist(),
        "z": umap_results[:, 2].tolist(),
        "labels": y.tolist()
    }
    with open(f"data/UMAP/umap_perp{i}.json", "w") as f:
        json.dump(umap_data, f)

    