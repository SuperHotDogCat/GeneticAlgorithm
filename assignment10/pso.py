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

# Example usage:
def objective_function(x):
    fx = x**6-6*x**4+9*x**2+x
    return np.sum(fx)

dimension = 2
num_particles = 20
lower_bound = -5
upper_bound = 5
max_iterations = 100
inertia_weight = 0.7
cognitive_weight = 1.5
social_weight = 1.5

pso = PSO(objective_function, dimension, num_particles, lower_bound, upper_bound, max_iterations, inertia_weight, cognitive_weight, social_weight)
best_solution, best_fitness = pso.optimize()
print("Best solution found:", best_solution)
print("Best fitness:", best_fitness)

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

pso = PSO(ackley_objective_function, dimension, num_particles, lower_bound, upper_bound, max_iterations, inertia_weight, cognitive_weight, social_weight)
best_solution, best_fitness = pso.optimize()
print("Best solution found:", best_solution)
print("Best fitness:", best_fitness)

pso = PSO(griewank_objective_function, dimension, num_particles, lower_bound, upper_bound, max_iterations, inertia_weight, cognitive_weight, social_weight)
best_solution, best_fitness = pso.optimize()
print("Best solution found:", best_solution)
print("Best fitness:", best_fitness)

pso = PSO(schwefel_objective_function, dimension, num_particles, lower_bound, upper_bound, max_iterations, inertia_weight, cognitive_weight, social_weight)
best_solution, best_fitness = pso.optimize()
print("Best solution found:", best_solution)
print("Best fitness:", best_fitness)

pso = PSO(xinsheyang_objective_function, dimension, num_particles, lower_bound, upper_bound, max_iterations, inertia_weight, cognitive_weight, social_weight)
best_solution, best_fitness = pso.optimize()
print("Best solution found:", best_solution)
print("Best fitness:", best_fitness)
"""
n = 100
scores = [0] * n
for idx in range(n):
    optimizer = PSO(objective_function, dimension, num_particles, lower_bound, upper_bound, max_iterations, inertia_weight, cognitive_weight, social_weight)
    best_solution, best_fitness = optimizer.optimize()
    scores[idx] = best_fitness
    
scores.sort()
plt.hist(scores[:100])
plt.legend([f"min: {np.min(scores[:100])}, best_solution: {best_solution}"])
plt.savefig("pso.png")
"""

objective_functions = [objective_function, ackley_objective_function, griewank_objective_function, schwefel_objective_function, xinsheyang_objective_function]
n = 30
fig, ax = plt.subplots(len(objective_functions), 1, figsize=(12, 8))
for i, objective_f in enumerate(objective_functions):
    scores = [0] * n
    solutions = [0] * n
    for idx in range(n):
        pso = PSO(objective_f, dimension, num_particles, lower_bound, upper_bound, max_iterations, inertia_weight, cognitive_weight, social_weight)
        best_solution, best_fitness = pso.optimize()
        scores[idx] = best_fitness
        solutions[idx] = best_solution
    ax[i].hist(scores)
    ax[i].legend([f"min: {np.min(scores)}, best_solution: {solutions[np.argmin(scores)]}"])
plt.savefig("pso.png")