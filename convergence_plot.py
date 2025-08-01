import pandas as pd
import matplotlib.pyplot as plt

ga = pd.read_csv("data/ga_convergence.csv")
de = pd.read_csv("data/de_convergence.csv")
pso = pd.read_csv("data/pso_convergence.csv")

plt.plot(ga["Fitness"], label="GA")
plt.plot(de["Fitness"], label="DE")
plt.plot(pso["Fitness"], label="PSO")
plt.xlabel("Generation")
plt.ylabel("Best Fitness")
plt.title("Convergence Comparison on Rastrigin Function")
plt.legend()
plt.grid(True)
plt.savefig("report/convergence_plot.png")
plt.show()
