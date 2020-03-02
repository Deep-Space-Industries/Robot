import numpy as np
import random
import math
import matplotlib.pyplot as plt

#
mu = 0.0
std = 5.0
dim = 2
scale = 0.25
precision = 10000

# Create
class Individual:
    def __init__(self):
        self.genotype = encoder(np.random.normal(loc=mu, scale=std, size=dim))
        self.fitness = rosenbrock(decoder(self.genotype))

def pair(bestIndividuals, n_offsprings, method):
    if (method == "random"):
        random.shuffle(bestIndividuals)
    offsprings = []
    for i in range(len(bestIndividuals))[::2]:
        newOffspring = Individual()
        newOffspring.genotype = (decoder(bestIndividuals[i].genotype)+decoder(bestIndividuals[i+1].genotype))/2
        newOffspring.genotype = encoder(mutate(newOffspring))
        offsprings.append(newOffspring)
    return offsprings

def mutate(offspring):
    return offspring.genotype + np.random.normal(loc=mu*scale, scale=std*scale, size=dim)

class Population:
    def __init__(self, n_individuals, n_bestIndividuals, n_offsprings, n_kill, n_epochs, decreaseFactorMutation, scale=scale):
        self.n_individuals = n_individuals
        self.individuals = [Individual() for i in range(n_individuals)]
        self.decreaseFactorMutation = decreaseFactorMutation
        self.history=[]
        self.bhistory=[]
        for i in range(n_epochs):
            self.history.append([decoder(k.genotype) for k in self.individuals])
            self.individuals.sort(key=lambda x: x.fitness, reverse=False)
            self.bestIndividuals = self.individuals[:n_bestIndividuals]
            self.offsprings = pair(self.bestIndividuals, n_offsprings, "else")
            for o in self.offsprings:
                o.fitness = rosenbrock(decoder(o.genotype))
            self.allIndividuals = self.individuals + self.offsprings
            self.allIndividuals.sort(key=lambda x: x.fitness, reverse=False)
            self.allIndividuals = self.allIndividuals[:-n_kill]
            self.individuals = self.allIndividuals
            scale -= decreaseFactorMutation

def encoder(phenotype):
    return np.array((bin(int(phenotype[0]*precision)),bin(int(phenotype[1]*precision))))

def decoder(genotype):
    return np.array((int(genotype[0], 2)/precision, int(genotype[1], 2)/precision))

# Define Rosenbrock performance function
def rosenbrock(genotype):
  return np.square(1 - genotype[0]) + 100 * np.square((genotype[1] - genotype[0] * genotype[0]))

# Define Rastrigin performance function
def rastrigin(genotype):
  sigma = 0
  for i in genotype:
    sigma = sigma + (i * i - 10 * math.cos(2 * math.pi * i))
  return 20 + sigma

n_epochs = 5000
myPop = Population(100, 60, 1, 30, n_epochs, 0.001)

print("Survivors")
for i in myPop.allIndividuals:
    print(decoder(i.genotype))

print("Fitness")
for i in myPop.allIndividuals:
    print(i.fitness)

print("The End")

fig = plt.figure(dpi=320)

# Performance functions for plotting
markers = ["*"]  # Symbols for different particles
xxx = np.linspace(-10, 10, 1000)
yyy = np.linspace(-10, 10, 1000)
X, Y = np.meshgrid(xxx, yyy)
Z = 20 + (X * X - 10 * np.cos(2 * np.pi * X)) + (Y * Y - 10 * np.cos(2 * np.pi * Y))  # Rastrigin function
#Z = (1 - X) ** 2 + 100 * (Y - X * X) ** 2  # Rosenbrock function

for t in range(n_epochs):
    if(t%100==0):
        print(t + 1)

        # Start drawing
        plt.title("Number of total iterations: " + str(t + 1))

        pos = myPop.history[t]
        for j in pos:
            plt.scatter(j[0], j[1], marker=markers[0])
            #i.performance_history.append(perfunc(i.genotype))

        plt.xlim(-10, 10)
        plt.ylim(-10, 10)
        plt.show()