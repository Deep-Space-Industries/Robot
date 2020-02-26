import numpy as np
import random
import math

mu = 0.0
std = 1.0
dim = 2
scale = 0.25

class Individual:
    def __init__(self):
        self.position = np.random.normal(loc=mu, scale=std, size=dim)
        self.fitness = rastrigin(self.position)

def pair(bestIndividuals, n_offsprings):
    random.shuffle(bestIndividuals)
    offsprings = []
    for i in range(len(bestIndividuals))[::2]:
        newOffspring = Individual()
        newOffspring.position = (bestIndividuals[i].position+bestIndividuals[i+1].position)/2
        newOffspring.position = mutate(newOffspring)
        offsprings.append(newOffspring)
    return offsprings

def mutate(offspring):
    return offspring.position + np.random.normal(loc=mu*scale, scale=std*scale, size=dim)

class Population:
    def __init__(self, n_individuals, n_bestIndividuals, n_offsprings, n_kill, n_epochs):
        self.n_individuals = n_individuals
        self.individuals = [Individual() for i in range(n_individuals)]

        for i in range(n_epochs):
            self.individuals.sort(key=lambda x: x.fitness, reverse=False)
            self.bestIndividuals = self.individuals[:n_bestIndividuals]
            self.offsprings = pair(self.bestIndividuals, n_offsprings)
            for o in self.offsprings:
                o.fitness = rastrigin(o.position)
            self.allIndividuals = self.individuals + self.offsprings
            self.allIndividuals.sort(key=lambda x: x.fitness, reverse=False)
            self.allIndividuals = self.allIndividuals[:-n_kill]
            self.individuals = self.allIndividuals

# Define Rosenbrock performance function
def rosenbrock(position):
  return np.square(1 - position[0]) + 100 * np.square((position[1] - position[0] * position[0]))

# Wu, Zhangyi
# Define Rastrigin performance function
def rastrigin(position):
  sigma = 0
  for i in position:
    sigma = sigma + (i * i - 10 * math.cos(2 * math.pi * i))
  return 20 + sigma

myPop = Population(10, 4, 1, 2, 1000)

for i in myPop.individuals:
    print(i.position)

print("Best Ones")

for i in myPop.bestIndividuals:
    print(i.fitness)

print("Offsprings")
for i in myPop.offsprings:
    print(i.position)

print("Kill")
for i in myPop.allIndividuals:
    print(i.position)