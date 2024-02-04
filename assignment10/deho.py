import numpy as np
import matplotlib.pyplot as plt
class DEHO:
    def __init__(self, objective_function, max_iterations, population_size, search_space_dimension, search_space_boundaries):
        self.objective_function = objective_function
        self.max_iterations = max_iterations
        self.population_size = population_size
        self.search_space_dimension = search_space_dimension
        self.search_space_boundaries = search_space_boundaries
        self.population = None
        self.fitness_values = None

    def initialize_population(self):
        self.population = np.random.uniform(low=self.search_space_boundaries[0], high=self.search_space_boundaries[1],
                                             size=(self.population_size, self.search_space_dimension))

    def evaluate_fitness(self):
        self.fitness_values = np.apply_along_axis(self.objective_function, 1, self.population)

    def waggle_dance_communication(self):
        # Calculate fitness rank of each solution
        fitness_rank = np.argsort(self.fitness_values)

        # Calculate waggle dance information based on fitness rank
        waggle_dance_info = 1 / (fitness_rank + 1)  # Inverse rank (higher fitness => lower rank => more information)

        # Share waggle dance information among solutions
        for i, solution in enumerate(self.population):
            self.population[i] += np.random.uniform(-0.1, 0.1, size=solution.shape) * waggle_dance_info[i]

    def update_velocities(self, inertia_weight=0.5, cognitive_weight=0.5, social_weight=0.5):
        personal_best = self.population[np.argmin(self.fitness_values)]
        global_best = personal_best

        for i, solution in enumerate(self.population):
            velocity = np.random.uniform(-0.1, 0.1, size=solution.shape)  # Initialize velocity randomly
            velocity += inertia_weight * velocity  # Inertia component
            velocity += cognitive_weight * np.random.uniform() * (personal_best - solution)  # Cognitive component
            velocity += social_weight * np.random.uniform() * (global_best - solution)  # Social component
            self.population[i] += velocity

    def perform_levy_flights(self, levy_flight_probability=0.1, levy_flight_scale=0.1):
        for i, solution in enumerate(self.population):
            if np.random.uniform() < levy_flight_probability:
                levy_flight_step = levy_flight_scale * np.random.standard_cauchy(size=solution.shape)
                self.population[i] += levy_flight_step

    def optional_local_search(self):
        # Optionally apply local search to improve solutions
        for i, solution in enumerate(self.population):
            improved_solution = self.hill_climbing(solution)
            if self.objective_function(improved_solution) < self.objective_function(solution):
                self.population[i] = improved_solution

    def hill_climbing(self, solution, step_size=0.1, max_iterations=10):
        current_solution = solution.copy()
        for _ in range(max_iterations):
            # Generate a random neighboring solution
            neighbor_solution = current_solution + np.random.uniform(-step_size, step_size, size=current_solution.shape)
            # If the neighbor has better fitness, move to it
            if self.objective_function(neighbor_solution) < self.objective_function(current_solution):
                current_solution = neighbor_solution
        return current_solution
        
    def run(self):
        self.initialize_population()
        self.evaluate_fitness()

        for iteration in range(self.max_iterations):
            self.waggle_dance_communication()
            self.update_velocities()
            self.perform_levy_flights()
            self.optional_local_search()
            self.evaluate_fitness()

        best_solution_index = np.argmin(self.fitness_values)
        best_solution = self.population[best_solution_index]
        best_fitness = self.fitness_values[best_solution_index]

        return best_solution, best_fitness

# Define the objective function
def objective_function(x):
    fx = x**6 - 6*x**4 + 9*x**2 + x
    return np.sum(fx)

def ackley_objective_function(x):
    t1 = 20
    t2 = -20 * np.exp(-0.2 * np.sqrt(1.0 / len(x) * np.sum(x ** 2, axis=0)))
    t3 = np.e
    t4 = -np.exp(1.0 / len(x) * np.sum(np.cos(2 * np.pi * x), axis=0))
    return t1 + t2 + t3 + t4

def griewank_objective_function(x):
    w = np.array([1.0 / np.sqrt(i + 1) for i in range(len(x))])
    t1 = 1
    t2 = 1.0 / 4000.0 * np.sum(x ** 2)
    t3 = - np.prod(np.cos(x * w))
    return t1 + t2 + t3

def schwefel_objective_function(x):
    return - np.sum(x * np.sin( np.sqrt( np.abs(x) ) ) )

def xinsheyang_objective_function(x):
    t1 = np.sum( np.abs(x) )
    e1 = - np.sum( np.sin(x ** 2) )
    t2 = np.exp(e1)
    return t1 * t2

# Example usage
max_iterations = 100
population_size = 50
search_space_dimension = 2
search_space_boundaries = (-10, 10)

deho = DEHO(objective_function, max_iterations, population_size, search_space_dimension, search_space_boundaries)
best_solution, best_fitness = deho.run()

print("Best solution:", best_solution)
print("Best fitness:", best_fitness)

deho = DEHO(ackley_objective_function, max_iterations, population_size, search_space_dimension, search_space_boundaries)
best_solution, best_fitness = deho.run()

print("Best solution:", best_solution)
print("Best fitness:", best_fitness)

deho = DEHO(griewank_objective_function, max_iterations, population_size, search_space_dimension, search_space_boundaries)
best_solution, best_fitness = deho.run()

print("Best solution:", best_solution)
print("Best fitness:", best_fitness)

deho = DEHO(schwefel_objective_function, max_iterations, population_size, search_space_dimension, search_space_boundaries)
best_solution, best_fitness = deho.run()

print("Best solution:", best_solution)
print("Best fitness:", best_fitness)

deho = DEHO(xinsheyang_objective_function, max_iterations, population_size, search_space_dimension, search_space_boundaries)
best_solution, best_fitness = deho.run()

print("Best solution:", best_solution)
print("Best fitness:", best_fitness)

objective_functions = [objective_function, ackley_objective_function, griewank_objective_function, schwefel_objective_function, xinsheyang_objective_function]
n = 30
fig, ax = plt.subplots(len(objective_functions), 1, figsize=(12, 8))
for i, objective_f in enumerate(objective_functions):
    scores = [0] * n
    solutions = [0] * n
    for idx in range(n):
        deho = DEHO(objective_f, max_iterations, population_size, search_space_dimension, search_space_boundaries)
        best_solution, best_fitness = deho.run()
        scores[idx] = best_fitness
        solutions[idx] = best_solution
    ax[i].hist(scores)
    ax[i].legend([f"min: {np.min(scores)}, best_solution: {solutions[np.argmin(scores)]}"])
plt.savefig("deho.png")