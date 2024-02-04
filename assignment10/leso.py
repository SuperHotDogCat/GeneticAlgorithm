import numpy as np
import matplotlib.pyplot as plt

class LESO:
    def __init__(self, objective_function, initial_population, max_iterations, max_eval_count):
        self.objective_function = objective_function
        self.population = initial_population
        self.max_iterations = max_iterations
        self.max_eval_count = max_eval_count
        self.eval_count = 0

    def evaluate_fitness(self, x):
        return self.objective_function(x)

    def optimize(self):
        for iteration in range(self.max_iterations):
            selected_solutions = self.select_solutions()
            for solution in selected_solutions:
                fitness = self.evaluate_fitness(solution)
                self.eval_count += 1
                if self.eval_count >= self.max_eval_count:
                    return self.population[np.argmin([self.evaluate_fitness(sol) for sol in self.population])]
            self.update_population(selected_solutions)
        return self.population[np.argmin([self.evaluate_fitness(sol) for sol in self.population])]

    def select_solutions(self):
        # Placeholder for adaptive selection strategy
        return self.population[:len(self.population) // 2]  # Select half of the population for evaluation

    def update_population(self, selected_solutions):
        # Placeholder for adaptive swarm dynamics update
        for i in range(len(selected_solutions)):
            self.population[i] = self.explore(selected_solutions[i])

    def explore(self, solution):
        # Placeholder for exploration strategy
        return solution + np.random.normal(0, 0.1, size=solution.shape)  # Add random noise to the solution

# Example usage:
def objective_function(x):
    fx = x**6-6*x**4+9*x**2+x
    return np.sum(fx)

initial_population = np.random.uniform(-5, 5, size=(10, 2))
max_iterations = 1000
max_eval_count = 1000

optimizer = LESO(objective_function, initial_population, max_iterations, max_eval_count)
best_solution = optimizer.optimize()
print("Best solution found:", best_solution)
print("Fitness:", objective_function(best_solution))

n = 100
scores = [0] * n
solutions = [0] * n
for idx in range(n):
    optimizer = LESO(objective_function, initial_population, max_iterations, max_eval_count)
    best_solution = optimizer.optimize()
    scores[idx] = objective_function(best_solution)
    solutions[idx] = best_solution

scores.sort()
plt.hist(scores[:100])
plt.legend([f"min: {np.min(scores[:100])}, best_solution: {solutions[np.argmin(scores[:100])]}"])
plt.savefig("leso.png")