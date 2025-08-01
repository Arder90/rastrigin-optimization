import numpy as np
import pandas as pd
from rastrigin import rastrigin

np.random.seed(42)

# Parameters
POP_SIZE = 50
GENS = 100
w = 0.7       # inertia weight
c1 = 1.5      # cognitive component
c2 = 1.5      # social component
BOUNDS = [-5.12, 5.12]
VEL_LIMIT = 0.2 * (BOUNDS[1] - BOUNDS[0])

def initialize_population():
    pos = np.random.uniform(BOUNDS[0], BOUNDS[1], (POP_SIZE, 2))
    vel = np.random.uniform(-VEL_LIMIT, VEL_LIMIT, (POP_SIZE, 2))
    return pos, vel

def pso():
    pos, vel = initialize_population()
    pbest = np.copy(pos)
    pbest_val = np.array([rastrigin(ind) for ind in pos])
    gbest_idx = np.argmin(pbest_val)
    gbest = np.copy(pbest[gbest_idx])
    gbest_val = pbest_val[gbest_idx]
    best_fitness = []

    for gen in range(GENS):
        for i in range(POP_SIZE):
            r1, r2 = np.random.rand(2)
            vel[i] = (w * vel[i] +
                      c1 * r1 * (pbest[i] - pos[i]) +
                      c2 * r2 * (gbest - pos[i]))
            vel[i] = np.clip(vel[i], -VEL_LIMIT, VEL_LIMIT)
            pos[i] += vel[i]
            pos[i] = np.clip(pos[i], BOUNDS[0], BOUNDS[1])

            fit = rastrigin(pos[i])
            if fit < pbest_val[i]:
                pbest[i] = pos[i]
                pbest_val[i] = fit

        gbest_idx = np.argmin(pbest_val)
        gbest = pbest[gbest_idx]
        gbest_val = pbest_val[gbest_idx]

        best_fitness.append(gbest_val)
        print(f"Gen {gen}: Best Fitness = {gbest_val:.4f}")

    pd.DataFrame(best_fitness, columns=["Fitness"]).to_csv("data/pso_convergence.csv", index=False)

if __name__ == "__main__":
    pso()