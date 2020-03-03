# This code was written in pair-programming style
# Members: Berat Cakir, Koushik Haridasyam, Zhangyi Wu

# Import packages
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from matplotlib import ticker
from scipy.spatial.distance import hamming

# Parameters for dimensionality and distribution of positions
mu = 0.0
std = 5.0
dim = 2
scale = 0.25

# Define class individual and its variables
class Individual:
    def __init__(self, benchmarkFunction):
        self.genotype = encoder(np.random.normal(loc=mu, scale=std, size=dim))
        self.benchmarkFunction = benchmarkFunction
        self.fitness = self.benchmarkFunction(decoder(self.genotype))

# Create n new offsprings by m number of individuals
def pair(bestIndividuals, n_offsprings, m_parents, method, benchmarkFunction):
    # Pair individuals randomly
    if (method == "random"):
        random.shuffle(bestIndividuals)
    # Else: pair best individuals with best individuals
    offsprings = []
    for i in range(len(bestIndividuals))[::2]:
        # Generates n new offsprings with mutation value added
        newOffspring = Individual(benchmarkFunction)
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
    def __init__(self, n_individuals, n_bestIndividuals, n_offsprings, m_parents, n_kill, n_epochs, decreaseFactorMutation, benchmarkFunction, scale=scale):
        # Create initial population
        self.n_individuals = n_individuals
        self.benchmarkFunction = benchmarkFunction
        self.individuals = [Individual(self.benchmarkFunction) for i in range(n_individuals)]
        self.decreaseFactorMutation = decreaseFactorMutation
        # Saving positions into history
        self.history=[]
        # Evolving for n number of epochs
        for i in range(n_epochs):
            self.history.append([decoder(k.genotype) for k in self.individuals])
            self.individuals.sort(key=lambda x: x.fitness, reverse=False)
            self.bestIndividuals = self.individuals[:n_bestIndividuals]
            # Pairing of n individuals
            self.offsprings = pair(self.bestIndividuals, n_offsprings, m_parents, "else", self.benchmarkFunction)
            # Calculate fitness of new offsprings
            for o in self.offsprings:
                o.fitness = self.benchmarkFunction(decoder(o.genotype))
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

# Create population
n_epochs = 2000
benchmarkFunction = rastrigin
myPop = Population(100, 60, 1, 3, 30, n_epochs, 0.00001, benchmarkFunction)

# Output
print("Final population (survivors):")
for i in myPop.allIndividuals:
    print("Individual - Genotype:",i.genotype[0],i.genotype[1], "Fitness:",i.fitness)

# Save fitness and diversity of individuals
maxFitness = []
avgFitness = []
maxDiversity = []
avgDiversity = []

# Compute the max fitness, average fitness, max diversity and average diversity through out all generations
for t in range(n_epochs):
    t_th_population = myPop.history[t]
    fitnessValues = [benchmarkFunction([ind[0], ind[1]]) for ind in t_th_population]
    # Get maximum and average fitness of each iteration
    maxFitness.append(min(fitnessValues))
    avgFitness.append(sum(fitnessValues) / len(fitnessValues))
    totalDiversities = 0
    bestDiversityOfThisGeneration = -1
    for genei in t_th_population:
        individualDiversity = 0
        for genej in t_th_population:
            # Calculate Euclidean distance
            distance = np.sqrt( (genei[0] - genej[0]) ** 2 + (genei[1] - genej[1]) ** 2)
            individualDiversity += distance
        if individualDiversity > bestDiversityOfThisGeneration:
            bestDiversityOfThisGeneration = individualDiversity
        totalDiversities += individualDiversity
    # Get maximum and average diversity of each iteration
    avgDiversity.append(totalDiversities / len(t_th_population))
    maxDiversity.append(bestDiversityOfThisGeneration)

# Show fitness plots
plt.subplots(figsize=(20, 10))
plt.plot(range(len(avgFitness)), avgFitness)
plt.plot(range(len(maxFitness)), maxFitness)
plt.title("Maximum and average fitness values of individuals for each generation")
plt.legend(["Maximum fitness", "Average fitness"])
plt.show()

# Show diversity plots
plt.subplots(figsize=(20, 10))
plt.subplot(211)
plt.plot(range(len(maxDiversity)), maxDiversity)
plt.title("Maximum diversity of population for each generation")
plt.subplot(212)
plt.plot(range(len(avgDiversity)), avgDiversity)
plt.title("Average diversity of population for each generation")
plt.show()

# Plotting individuals on fitness function with their scores
fig = plt.figure(dpi=320)

markers = ["*"]  # Symbols for different individuals
xxx = np.linspace(-10, 10, 1000)
yyy = np.linspace(-10, 10, 1000)
X, Y = np.meshgrid(xxx, yyy)

# Performance functions for plotting
if (benchmarkFunction == rastrigin):
    Z = 20 + (X * X - 10 * np.cos(2 * np.pi * X)) + (Y * Y - 10 * np.cos(2 * np.pi * Y))  # Rastrigin function
elif(benchmarkFunction == rosenbrock):
    Z = (1 - X) ** 2 + 100 * (Y - X * X) ** 2  # Rosenbrock function

for t in range(n_epochs):
    if(t%50==0):
        print("Epochs:", (t + 1))
        if (benchmarkFunction == rastrigin):
            contourf = plt.contourf(X, Y, Z)
        elif (benchmarkFunction == rosenbrock):
            contourf = plt.contourf(X, Y, Z, locator=ticker.LogLocator())

        # Start drawing
        plt.title("Number of total iterations: " + str(t + 1))

        pos = myPop.history[t]
        for j in pos:
            plt.scatter(j[0], j[1], marker=markers[0])

        plt.xlim(-10, 10)
        plt.ylim(-10, 10)
        plt.colorbar(contourf, orientation='horizontal')
        plt.show()