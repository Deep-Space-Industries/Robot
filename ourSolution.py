# This code was written in pair-programming style
# Members: Berat Cakir, Koushik Haridasyam, Zhangyi Wu

# Import packages
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from matplotlib import ticker

# Parameters for dimensionality and distribution of positions
mu = 0.0
std = 5.0
dim = 2
scale = 0.25

# Define class individual and its variables
class Individual:
    def __init__(self):
        self.genotype = encoder(np.random.normal(loc=mu, scale=std, size=dim))
        self.fitness = rosenbrock(decoder(self.genotype))

# Create n new offsprings by m number of individuals
def pair(bestIndividuals, n_offsprings, method):
    # Pair individuals randomly
    if (method == "random"):
        random.shuffle(bestIndividuals)
    # Else: pair best individuals with best individuals
    offsprings = []
    for i in range(len(bestIndividuals))[::2]:
        # Generates n new offsprings with mutation value added
        newOffspring = Individual()
        # Genes of parents being passed as average value
        newOffspring.genotype = (decoder(bestIndividuals[i].genotype)+decoder(bestIndividuals[i+1].genotype))/2
        # Mutation being added to the new gene
        newOffspring.genotype = encoder(mutate(newOffspring))
        offsprings.append(newOffspring)
    return offsprings

# Add mutation to genes
def mutate(offspring):
    return offspring.genotype + np.random.normal(loc=mu*scale, scale=std*scale, size=dim)

# Define class population and its variables
class Population:
    def __init__(self, n_individuals, n_bestIndividuals, n_offsprings, n_kill, n_epochs, decreaseFactorMutation, scale=scale):
        # Create initial population
        self.n_individuals = n_individuals
        self.individuals = [Individual() for i in range(n_individuals)]
        self.decreaseFactorMutation = decreaseFactorMutation
        # Saving positions into history
        self.history=[]
        self.bhistory=[]
        # Evolving for n number of epochs
        for i in range(n_epochs):
            self.history.append([decoder(k.genotype) for k in self.individuals])
            self.individuals.sort(key=lambda x: x.fitness, reverse=False)
            self.bestIndividuals = self.individuals[:n_bestIndividuals]
            # Pairing of n individuals
            self.offsprings = pair(self.bestIndividuals, n_offsprings, "else")
            # Calculate fitness of new offsprings
            for o in self.offsprings:
                o.fitness = rosenbrock(decoder(o.genotype))
            # Add new offsprings to the total population
            self.allIndividuals = self.individuals + self.offsprings
            self.allIndividuals.sort(key=lambda x: x.fitness, reverse=False)
            # Let n number of individuals die
            self.allIndividuals = self.allIndividuals[:-n_kill]
            self.individuals = self.allIndividuals
            scale -= decreaseFactorMutation

# Encodes position genes into binary format
def encoder(phenotype, precision=10000):
    return np.array((bin(int(phenotype[0]*precision)),bin(int(phenotype[1]*precision))))

# Decodes binary genes into position format
def decoder(genotype, precision=10000):
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
# Z = 20 + (X * X - 10 * np.cos(2 * np.pi * X)) + (Y * Y - 10 * np.cos(2 * np.pi * Y))  # Rastrigin function
Z = (1 - X) ** 2 + 100 * (Y - X * X) ** 2  # Rosenbrock function

for t in range(n_epochs):
    if(t%50==0):
        print(t + 1)
        # plt.contour(X, Y, Z, locator=ticker.LogLocator(), colors='k')
        contourf = plt.contourf(X, Y, Z, locator=ticker.LogLocator())
        # Calling environment genotype methods

        # Start drawing
        plt.title("Number of total iterations: " + str(t + 1))

        pos = myPop.history[t]
        for j in pos:
            plt.scatter(j[0], j[1], marker=markers[0])
            #i.performance_history.append(perfunc(i.genotype))

        plt.xlim(-10, 10)
        plt.ylim(-10, 10)
        plt.colorbar(contourf, orientation='horizontal')
        plt.show()