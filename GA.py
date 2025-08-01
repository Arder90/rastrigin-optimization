import numpy as np
from rastrigin import rastrigin
import pandas as pd

np.random.seed(42)
POP_SIZE = 50
GENS = 100
MUT_RATE = 0.01
CROSS_RATE = 0.7
BOUNDS = [-5.12, 5.12]

def initialize_population():
    return np.random.uniform(BOUNDS[0], BOUNDS[1], (POP_SIZE, 2))

def tournament_selection(pop, fitness, k=3):
    indices = np.random.choice(range(POP_SIZE), k)
    return pop[indices[np.argmin(fitness[indices])]]

def crossover(parent1, parent2):
    if np.random.rand() < CROSS_RATE:
        alpha = np.random.rand()
        return alpha * parent1 + (1 - alpha) * parent2
    return parent1

def mutate(individual):
    for i in range(len(individual)):
        if np.random.rand() < MUT_RATE:
            individual[i] += np.random.normal(0, 0.1)
            individual[i] = np.clip(individual[i], BOUNDS[0], BOUNDS[1])
    return individual

def ga():
    pop = initialize_population()
    fitness = np.array([rastrigin(ind) for ind in pop])
    best_fitness = []

    for gen in range(GENS):
        new_pop = []
        for _ in range(POP_SIZE):
            parent1 = tournament_selection(pop, fitness)
            parent2 = tournament_selection(pop, fitness)
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_pop.append(child)

        pop = np.array(new_pop)
        fitness = np.array([rastrigin(ind) for ind in pop])
        best_fitness.append(np.min(fitness))
        print(f"Gen {gen}: Best Fitness = {best_fitness[-1]:.4f}")

    pd.DataFrame(best_fitness, columns=["Fitness"]).to_csv("data/ga_convergence.csv", index=False)

if __name__ == "__main__":
    ga()
