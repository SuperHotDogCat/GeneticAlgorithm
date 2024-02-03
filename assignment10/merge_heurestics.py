import numpy as np
import matplotlib.pyplot as plt

class Particle:
    def __init__(self, dimension, lower_bound, upper_bound):
        self.position = np.random.uniform(lower_bound, upper_bound, size=dimension)
        self.velocity = np.random.uniform(-1, 1, size=dimension)
        self.best_position = self.position
        self.best_fitness = float('inf')

class PSO:
    def __init__(self, objective_function, dimension, num_particles, lower_bound, upper_bound, max_iterations, inertia_weight, cognitive_weight, social_weight):
        self.objective_function = objective_function
        self.dimension = dimension
        self.num_particles = num_particles
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.max_iterations = max_iterations
        self.inertia_weight = inertia_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        self.particles = [Particle(dimension, lower_bound, upper_bound) for _ in range(num_particles)]

    def optimize(self):
        for iteration in range(self.max_iterations):
            for particle in self.particles:
                fitness = self.objective_function(particle.position)
                if fitness < particle.best_fitness:
                    particle.best_fitness = fitness
                    particle.best_position = particle.position.copy()
                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = particle.position.copy()
            for particle in self.particles:
                r1 = np.random.random(self.dimension)
                r2 = np.random.random(self.dimension)
                cognitive_component = self.cognitive_weight * r1 * (particle.best_position - particle.position)
                social_component = self.social_weight * r2 * (self.global_best_position - particle.position)
                particle.velocity = self.inertia_weight * particle.velocity + cognitive_component + social_component
                particle.position += particle.velocity
                particle.position = np.clip(particle.position, self.lower_bound, self.upper_bound)
        return self.global_best_position, self.global_best_fitness

dimension = 2
num_particles = 20
lower_bound = -5
upper_bound = 5
max_iterations = 100
inertia_weight = 0.7
cognitive_weight = 1.5
social_weight = 1.5


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

def schwefel_objective_function(x):
    return - np.sum(x * np.sin( np.sqrt( np.abs(x) ) ) )

# Example usage
max_iterations = 1000
population_size = 50
search_space_dimension = 2
search_space_boundaries = (-10, 10)


pso = PSO(schwefel_objective_function, dimension, num_particles, lower_bound, upper_bound, max_iterations, inertia_weight, cognitive_weight, social_weight)
best_solution, best_fitness = pso.optimize()
print("Best solution found:", best_solution)
print("Best fitness:", best_fitness)

#merge 
deho = DEHO(schwefel_objective_function, max_iterations, population_size, search_space_dimension, search_space_boundaries)
best_solution, best_fitness = deho.run()
print("Best solution:", best_solution)
print("Best fitness:", best_fitness)

pso = PSO(schwefel_objective_function, dimension, num_particles, np.min(best_solution), np.max(best_solution), max_iterations, inertia_weight, cognitive_weight, social_weight)
pso.global_best_fitness = best_fitness
pso.global_best_position = best_solution
best_solution, best_fitness = pso.optimize()
print("Best solution found:", best_solution)
print("Best fitness:", best_fitness)