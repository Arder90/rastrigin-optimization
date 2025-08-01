import numpy as np
import pandas as pd
from rastrigin import rastrigin

np.random.seed(42)

# Parameters
POP_SIZE = 50
GENS = 100
F = 0.8  # Differential weight
CR = 0.9  # Crossover probability
BOUNDS = [-5.12, 5.12]

def initialize_population():
    return np.random.uniform(BOUNDS[0], BOUNDS[1], (POP_SIZE, 2))

def de():
    pop = initialize_population()
    fitness = np.array([rastrigin(ind) for ind in pop])
    best_fitness = []

    for gen in range(GENS):
        new_pop = []
        for i in range(POP_SIZE):
            indices = list(range(POP_SIZE))
            indices.remove(i)
            a, b, c = pop[np.random.choice(indices, 3, replace=False)]
            mutant = np.clip(a + F * (b - c), BOUNDS[0], BOUNDS[1])

            trial = np.copy(pop[i])
            for j in range(2):
                if np.random.rand() < CR:
                    trial[j] = mutant[j]

            trial_fit = rastrigin(trial)
            if trial_fit < fitness[i]:
                pop[i] = trial
                fitness[i] = trial_fit

        best_fitness.append(np.min(fitness))
        print(f"Gen {gen}: Best Fitness = {best_fitness[-1]:.4f}")

    pd.DataFrame(best_fitness, columns=["Fitness"]).to_csv("data/de_convergence.csv", index=False)

if __name__ == "__main__":
    de()