# Evolutionary Algorithm
import math
from NeuralNetwork import *
import pygame
n_epochs = 100
n_epoch_max_duration_ms = 5000

from robo import *
pygame.init()
font = pygame.font.SysFont("futura", 16)

class Individual:
    def __init__(self, benchmarkFunction):
        self.benchmarkFunction = benchmarkFunction
        self.nn = NeuralNetwork(14,[4,3],2, tanh, 0.1)
        self.robot = Robot(random.randint(200, 500), random.randint(200, 500), \
                           0, 0, 20, walls)
        self.robot.draw()
        self.robot.draw_sensors()
        sensors = self.robot.get_sensors()
        input = np.array([sensors])
        #input = np.array([sensors])
        # input = np.array([[200,180,7,0,10,175,50,190,7,6,13,50]])
        input = scalerr(input[0], 0, 200, -3, 3) # Scale values
        velocities = [self.robot.left_velocity, self.robot.right_velocity]
        velocities = scalerr(velocities, -30, 30, -3, 3)
        input[0] = np.append(input[0], velocities)
        self.position = self.nn.forwardPropagation(input[0])  # velocities
        self.position = scalerr(self.position[0], -3, 3, -30, 30)[0]
        self.robot.update_velocities(self.position[0], self.position[1])
        self.robot.update_icc()
        self.environment = Environment(density = 1)
        # self.fitness = self.benchmarkFunction(self.position)
        self.fitness = self.environment.cleared_dust

    def update_individual(self):
        sensors = self.robot.get_sensors()
        input = np.array([sensors])
        input = scalerr(input[0], 0, 200, -3, 3)  # Scale values
        velocities = [self.robot.left_velocity, self.robot.right_velocity]
        velocities = scalerr(velocities, -30, 30, -3, 3)
        input[0] = np.append(input[0], velocities)
        self.position = self.nn.forwardPropagation(input[0])  # velocities
        self.position = scalerr(self.position[0], -3, 3, -30, 30)[0]
        #print("before", [self.robot.left_velocity, self.robot.right_velocity])
        self.robot.update_velocities(self.position[0], self.position[1])
        self.robot.update_icc()
        #print("after", [self.robot.left_velocity, self.robot.right_velocity])
        velocity = [self.robot.left_velocity, self.robot.right_velocity]
        collision = self.robot.collision1
        self.fitness = self.fitnessFunction(velocity, sensors, self.robot.environment.cleared_dust, collision, 0.25, 0.5, 0.25)

    def fitnessFunction(self, velocity, sensor, dust, collision, w1, w2, w3):
        averageVelocity = (velocity[0] + velocity[1])/2
        deltaVelocity = abs(velocity[0] - velocity[1])
        maxSensor = max(sensor)
        return w1*(averageVelocity*(1-math.sqrt(deltaVelocity))*(1-maxSensor))+w2*dust-w3*collision*10

class Population:
    def __init__(self, n_individuals, n_bestIndividuals, n_offsprings, m_parents, n_kill, n_epochs, scale, decreaseFactorMutation, benchmarkFunction):
        self.n_individuals = n_individuals
        self.n_bestIndividuals = n_bestIndividuals
        self.n_offsprings = n_offsprings
        self.m_parents = m_parents
        self.n_kill = n_kill
        self.n_epochs = n_epochs
        self.scale = scale
        self.decreaseFactorMutation = decreaseFactorMutation
        self.benchmarkFunction = benchmarkFunction

        self.individuals = [Individual(self.benchmarkFunction) for i in range(n_individuals)]
        self.history = []
        self.bestIndividuals = None
        self.allIndividuals = None
        self.offsprings = None

# Create n new offsprings by m number of individuals
def pair(bestIndividuals, n_offsprings, m_parents, method, benchmarkFunction):
    if (method == "random"):
        random.shuffle(bestIndividuals)
    offsprings = []
    for i in range(len(bestIndividuals))[::2]:
        newOffspring = Individual(benchmarkFunction)
        # Genes of parents being passed as average value
        newOffspring.nn.weightsIH=(bestIndividuals[i].nn.weightsIH+bestIndividuals[i+1].nn.weightsIH)/2
        newOffspring.nn.biasIHH[0]=(bestIndividuals[i].nn.biasIHH[0]+bestIndividuals[i+1].nn.biasIHH[0])/2
        newOffspring.nn.weightsHH[0] = (bestIndividuals[i].nn.weightsHH[0] + bestIndividuals[i + 1].nn.weightsHH[0]) / 2
        newOffspring.nn.biasIHH[1] = (bestIndividuals[i].nn.biasIHH[1] + bestIndividuals[i + 1].nn.biasIHH[1]) / 2
        newOffspring.nn.weightsHO = (bestIndividuals[i].nn.weightsHO + bestIndividuals[i + 1].nn.weightsHO) / 2
        newOffspring.nn.biasHO = (bestIndividuals[i].nn.biasHO + bestIndividuals[i + 1].nn.biasHO) / 2
        newOffspring.nn = mutate(newOffspring)
        offsprings.append(newOffspring)
    return offsprings

def mutate(offspring):
    offspring.nn.weightsIH += randomWeights(np.zeros((offspring.nn.inputnodes, offspring.nn.hiddennodes[0]), dtype=float))*population.scale
    offspring.nn.biasIHH[0] += randomWeights(np.zeros((1, offspring.nn.hiddennodes[0]), dtype=float))*population.scale
    offspring.nn.weightsHH[0] += randomWeights(np.zeros((offspring.nn.hiddennodes[0], offspring.nn.hiddennodes[1]), dtype=float))*population.scale
    offspring.nn.biasIHH[1] += randomWeights(np.zeros((1, offspring.nn.hiddennodes[1]), dtype=float))*population.scale
    offspring.nn.weightsHO += randomWeights(np.zeros((offspring.nn.hiddennodes[1], offspring.nn.outputnodes), dtype=float))*population.scale
    offspring.nn.biasHO += randomWeights(np.zeros((1, offspring.nn.outputnodes), dtype=float))*population.scale
    return offspring.nn

def updateEpoch(population):
    population.history.append([k.fitness for k in population.individuals])
    population.individuals.sort(key=lambda x: x.fitness, reverse=True)
    population.bestIndividuals = population.individuals[:population.n_bestIndividuals]
    population.offsprings = pair(population.bestIndividuals, 0, 0, "else", population.benchmarkFunction)
    for o in range(len(population.offsprings)):
        population.offsprings[o].fitness = population.benchmarkFunction(population.offsprings[o].position)
    population.allIndividuals = population.individuals + population.offsprings
    population.allIndividuals.sort(key=lambda x: x.fitness, reverse=True)
    population.allIndividuals = population.allIndividuals[:-population.n_kill]
    population.individuals = population.allIndividuals
    population.scale -= population.decreaseFactorMutation

def rastrigin(genotype):
  sigma = 0
  for i in genotype:
    sigma = sigma + (i * i - 10 * math.cos(2 * math.pi * i))
  return 20 + sigma

def rosenbrock(genotype):
  return np.square(1 - genotype[0]) + 100 * np.square((genotype[1] - genotype[0] * genotype[0]))

benchmarkFunction = rastrigin
population = Population(10, 6, 1, 3, 3, n_epochs, 0.25, 0.00001, benchmarkFunction)

# Simulator
# from OneFileSolution import *

def condition():
    return False
def reset():
    return False

done = False
x = 30
y = 30

clock = pygame.time.Clock()
restartEvent = pygame.USEREVENT + 1
pygame.time.set_timer(restartEvent, n_epoch_max_duration_ms)
updateEpoch(population)

while not done:
    time_seconds = (pygame.time.get_ticks() / 1000)
    if (pygame.event.get(restartEvent) or condition()):
        updateEpoch(population)
        n_epochs -= 1
        reset()
    if (n_epochs <= 0):
        done = True
    print("Epoch:", n_epochs)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                done = True

    # print(block.theta)
    # player1.update(block.x, block.y, block.theta, block.radius, 20)
    # player2.update(block.x, block.y, block.theta, block.radius, 0)
    screen.fill((255, 128, 128))
    blit_text(f"Epoch:{n_epochs}", 100, 100, BLACK, None, 20)
    for w in walls:
        w.draw()

    for individual in population.individuals:
    # blit_text(f'L: {block.left_velocity}; R: {block.right_velocity}', 800, 300, SILVER, BLACK)

        individual.update_individual()
        individual.robot.move()
        individual.robot.draw()
        print("fitness: ", individual.fitness)
        # individual.robot.environment.draw_dusts(individual.robot)
        individual.robot.draw_direction()
        individual.robot.draw_icc()
        individual.robot.draw_sensors()
        # e.draw_dusts(block)

    # pygame.display.flip()
    clock.tick(120)
    pygame.display.update()
    #screen.fill((0, 0, 0))
    #x += 1
    #pygame.draw.rect(screen, (0, 128, 255), pygame.Rect(x, y, 60, 60))

    #pygame.display.flip()
    #clock.tick(60)


for i in range(len(population.history())):
    print("Epoch", (i+1), "Final fitness", population.history[i])

# Output
# n_epochs = 50
# import matplotlib.pyplot as plt
# from matplotlib import ticker
# from scipy.spatial.distance import hamming
#
# print("Final population (survivors):")
# for i in population.allIndividuals:
#     print("Individual - Genotype:",i.position[0],i.position[1], "Fitness:",i.fitness)
#
# # Save fitness and diversity of individuals
# maxFitness = []
# avgFitness = []
# maxDiversity = []
# avgDiversity = []
#
# # Compute the max fitness, average fitness, max diversity and average diversity through out all generations
# for t in range(n_epochs):
#     t_th_population = population.history[t]
#     fitnessValues = [benchmarkFunction([ind[0], ind[1]]) for ind in t_th_population]
#     # Get maximum and average fitness of each iteration
#     maxFitness.append(min(fitnessValues))
#     avgFitness.append(sum(fitnessValues) / len(fitnessValues))
#     totalDiversities = 0
#     bestDiversityOfThisGeneration = -1
#     for genei in t_th_population:
#         individualDiversity = 0
#         for genej in t_th_population:
#             # Calculate Euclidean distance
#             distance = np.sqrt( (genei[0] - genej[0]) ** 2 + (genei[1] - genej[1]) ** 2)
#             individualDiversity += distance
#         if individualDiversity > bestDiversityOfThisGeneration:
#             bestDiversityOfThisGeneration = individualDiversity
#         totalDiversities += individualDiversity
#     # Get maximum and average diversity of each iteration
#     avgDiversity.append(totalDiversities / len(t_th_population))
#     maxDiversity.append(bestDiversityOfThisGeneration)
#
# # Show fitness plots
# plt.subplots(figsize=(20, 10))
# plt.plot(range(len(maxFitness)), maxFitness)
# plt.plot(range(len(avgFitness)), avgFitness)
# plt.title("Maximum and average fitness values of individuals for each generation")
# plt.legend(["Maximum fitness", "Average fitness"])
# plt.show()
#
# # Show diversity plots
# plt.subplots(figsize=(20, 10))
# plt.subplot(211)
# plt.plot(range(len(maxDiversity)), maxDiversity)
# plt.title("Maximum diversity of population for each generation")
# plt.subplot(212)
# plt.plot(range(len(avgDiversity)), avgDiversity)
# plt.title("Average diversity of population for each generation")
# plt.show()
#
# # Plotting individuals on fitness function with their scores
# fig = plt.figure(dpi=320)
#
# markers = ["*"]  # Symbols for different individuals
# xxx = np.linspace(-10, 10, 1000)
# yyy = np.linspace(-10, 10, 1000)
# X, Y = np.meshgrid(xxx, yyy)
#
# # Performance functions for plotting
# if (benchmarkFunction == rastrigin):
#     Z = 20 + (X * X - 10 * np.cos(2 * np.pi * X)) + (Y * Y - 10 * np.cos(2 * np.pi * Y))  # Rastrigin function
# elif(benchmarkFunction == rosenbrock):
#     Z = (1 - X) ** 2 + 100 * (Y - X * X) ** 2  # Rosenbrock function
#
# for t in range(n_epochs):
#     if(t%5==0):
#         print("Epochs:", (t + 1))
#         if (benchmarkFunction == rastrigin):
#             contourf = plt.contourf(X, Y, Z)
#         elif (benchmarkFunction == rosenbrock):
#             contourf = plt.contourf(X, Y, Z, locator=ticker.LogLocator())
#
#         # Start drawing
#         plt.title("Number of total iterations: " + str(t + 1))
#
#         pos = population.history[t]
#         for j in pos:
#             plt.scatter(j[0], j[1], marker=markers[0])
#
#         plt.xlim(-10, 10)
#         plt.ylim(-10, 10)
#         plt.colorbar(contourf, orientation='horizontal')
#         plt.show()