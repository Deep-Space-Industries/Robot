# Evolutionary Algorithm
import math
from NeuralNetwork import *
import pygame
n_epochs = 400
n_epoch_max_duration_ms = 15000
#sensormax = 500

from Robot import *
pygame.init()
font = pygame.font.SysFont("futura", 16)
COLOR_FOR_BEST = [(255,0,10), (0,255,128), (0,0,255), (238,114,114), (255,185,185)]

class Individual:
    def __init__(self, id = None):
        # self.nn = NeuralNetwork(14,[4,3],2, tanh, 0.1)
        self.nn = NeuralNetwork(22,[10],2, tanh, 0.1)
        self.robot = Robot(random.randint(50, 200), random.randint(50, 200), \
                           0, 0, 20, walls)
        self.robot.draw()
        self.id = id
        self.robot.draw_sensors()
        self.input = None
        self.sensors = None
        self.update_individual()

    def update_individual(self):
        sensors = self.robot.get_sensors()
        # sensors = list(map(lambda x: (np.e ** ((-(2 * x - 200)) / 37)) / 10, sensors))
        self.sensors = sensors
        isensors = [200 - s for s in sensors]
        input = scalerr(isensors, 0, 200, -2, 2) # Scale values
        # input = [np.array([sensors])]
        velocities = [self.robot.left_velocity, self.robot.right_velocity]
        velocities = scalerr(velocities, -10, 10, -1, 1)
        if len(self.nn.hiddenOutputs) == 0:
            self.nn.hiddenOutputs = randomWeights([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])[0]
        input[0] = np.append(input[0], self.nn.hiddenOutputs)
        self.input = input[0]
        self.position = self.nn.forwardPropagation(input[0])  # velocities
        # print(self.position)
        #self.position = scalerr(self.position[0], -1, 1, -10, 10)[0]
        self.robot.update_velocities(round(self.position[0][0]*10, 4), round(self.position[0][1] *10, 4))
        self.robot.update_icc()
        velocity = [self.robot.left_velocity, self.robot.right_velocity]
        collision = self.robot.collision1
        self.fitness = self.fitnessFunction(velocity, input[0], self.robot.environment.cleared_dust, collision, 0.4, 0.6, 0.25)
        print("Fitness before", self.fitness)
        if (min(sensors) < 40):
            self.fitness -= 10
        print("Fitness after", self.fitness)

    def fitnessFunction(self, velocity, sensor, dust, collision, w1, w2, w3):
        vels = scalerr([velocity[0], velocity[1]], -10, 10, 0, 1)[0]
        averageVelocity = (velocity[0] + velocity[1]) / 2
        deltaVelocity = abs(velocity[0] - velocity[1])
        maxSensor = max(sensor) / 200
        minSensor = min(sensor)

        velocityScore = w1 * ( averageVelocity * (1 - math.sqrt(deltaVelocity) ))
        dustScore = w2 * dust
        print("VelScore", velocityScore)
        print("dustScore", dustScore)

        fitness = velocityScore + dustScore

        #penalty = 0.2
        #ipt = abs(minSensor - 200) * 2
        #f1 = w1 * averageVelocity * (1 - math.sqrt(deltaVelocity)) * (1 - maxSensor)
        #f2 = w2 * dust
        #f3 = collision * (1-w2)
        # print(f"speed: {f1}")
        # print(f"dust: {f2}")
        #fitness = -f3 + f2 / 10

        # fitness = w1 * (averageVelocity * (1 - math.sqrt(deltaVelocity)) * (1 - maxSensor)) + \
        #           w2 * dust

        # fitness = w1 * (averageVelocity * (1 - math.sqrt(deltaVelocity)) * (1 - maxSensor)) + \
        #           w2 * dust - w3 * collision * 5 - minSensor * penalty - deltaVelocity * ipt

        # print(f"speed: {w1 * ( averageVelocity * (1 - math.sqrt(deltaVelocity) ) * ( 1 - maxSensor )) }")
        # print(f"dust: {w2 * dust}")
        # print(f"collision: {w3 * collision * 5}")
        # print(f"sensor: {minSensor*penalty}")
        # print(f"ipt: {deltaVelocity*ipt}")
        # print(f"total: {fitness}")

        return fitness

class Population:
    def __init__(self, n_individuals, n_bestIndividuals, n_offsprings, m_parents, n_kill, n_epochs, scale, decreaseFactorMutation):
        self.n_individuals = n_individuals
        self.n_bestIndividuals = n_bestIndividuals
        self.n_offsprings = n_offsprings
        self.m_parents = m_parents
        self.n_kill = n_kill
        self.n_epochs = n_epochs
        self.scale = scale
        self.decreaseFactorMutation = decreaseFactorMutation

        self.individuals = [Individual() for i in range(n_individuals)]
        idi = random.choice(self.individuals)
        idi.id = "!@#"
        self.history = []

        self.historyWeightsIH = []
        self.historyBiasIHH1 = []
        self.historyWeightsHH = []
        self.historyBiasIHHn = []
        self.historyWeightsHO = []
        self.historyBiasHO = []

        self.bestIndividuals = None
        self.allIndividuals = None
        self.offsprings = None

# Create n new offsprings by m number of individuals
def pair(bestIndividuals, n_offsprings, m_parents, method):
    if (method == "random"):
        random.shuffle(bestIndividuals)
    offsprings = []
    for i in range(len(bestIndividuals))[::2]:
        newOffspring = Individual()
        # Genes of parents being passed as average value
        newOffspring.nn.weightsIH=(bestIndividuals[i].nn.weightsIH+bestIndividuals[i+1].nn.weightsIH)/2
        newOffspring.nn.biasIHH[0]=(bestIndividuals[i].nn.biasIHH[0]+bestIndividuals[i+1].nn.biasIHH[0])/2
        #for l in range(len(newOffspring.nn.weightsHH)):
        #    newOffspring.nn.weightsHH[l] = (bestIndividuals[i].nn.weightsHH[l] + bestIndividuals[i + 1].nn.weightsHH[l]) / 2
        #    newOffspring.nn.biasIHH[l+1] = (bestIndividuals[i].nn.biasIHH[l+1] + bestIndividuals[i + 1].nn.biasIHH[l+1]) / 2
        newOffspring.nn.weightsHO = (bestIndividuals[i].nn.weightsHO + bestIndividuals[i + 1].nn.weightsHO) / 2
        newOffspring.nn.biasHO = (bestIndividuals[i].nn.biasHO + bestIndividuals[i + 1].nn.biasHO) / 2
        newOffspring.nn = mutate(newOffspring)
        offsprings.append(newOffspring)
    return offsprings

def mutate(offspring):
    offspring.nn.weightsIH += randomWeights(np.zeros((offspring.nn.inputnodes, offspring.nn.hiddennodes[0]), dtype=float))*population.scale
    offspring.nn.biasIHH[0] += randomWeights(np.zeros((1, offspring.nn.hiddennodes[0]), dtype=float))*population.scale
    #for l in range(len(offspring.nn.weightsHH)-1):
    #    offspring.nn.weightsHH[l] += randomWeights(np.zeros((offspring.nn.hiddennodes[l], offspring.nn.hiddennodes[l+1]), dtype=float))*population.scale
    #    offspring.nn.biasIHH[l+1] += randomWeights(np.zeros((1, offspring.nn.hiddennodes[l+1]), dtype=float))*population.scale
    offspring.nn.weightsHO += randomWeights(np.zeros((offspring.nn.hiddennodes[0], offspring.nn.outputnodes), dtype=float))*population.scale
    offspring.nn.biasHO += randomWeights(np.zeros((1, offspring.nn.outputnodes), dtype=float))*population.scale
    return offspring.nn

def updateEpoch(population, killbest):
    population.individuals.sort(key=lambda x: x.fitness, reverse=True)

    population.history.append([k.fitness for k in population.individuals])
    population.historyWeightsIH.append([k.nn.weightsIH for k in population.individuals])
    population.historyBiasIHH1.append([k.nn.biasIHH[0] for k in population.individuals])
    #population.historyWeightsHH.append([k.nn.weightsHH[0] for k in population.individuals])
    #population.historyBiasIHHn.append([k.nn.biasIHH[1] for k in population.individuals])
    population.historyWeightsHO.append([k.nn.weightsHO for k in population.individuals])
    population.historyBiasHO.append([k.nn.biasHO for k in population.individuals])
    #
    np.save("/FT", population.history, allow_pickle=True, fix_imports=True)
    np.save("/IH", population.historyWeightsIH, allow_pickle=True, fix_imports=True)
    np.save("/BIHH", population.historyBiasIHH1, allow_pickle=True, fix_imports=True)
    #np.save("n/HH", population.historyWeightsHH, allow_pickle=True, fix_imports=True)
    #np.save("n/BIHHn", population.historyBiasIHHn, allow_pickle=True, fix_imports=True)
    np.save("/HO", population.historyWeightsHO, allow_pickle=True, fix_imports=True)
    np.save("/BHO", population.historyBiasHO, allow_pickle=True, fix_imports=True)

    population.bestIndividuals = population.individuals[:population.n_bestIndividuals]
    population.offsprings = pair(population.bestIndividuals, 0, 0, "else")
    for o in range(len(population.offsprings)):
        po = population.offsprings[o]
        sensors = po.robot.get_sensors()
        #sensors = list(map(lambda x: (np.e ** ((-(2 * x - 200)) / 37)) / 10, po.robot.get_sensors()))
        population.offsprings[o].fitness = population.offsprings[o].fitnessFunction(\
            [po.robot.left_velocity, po.robot.right_velocity], sensors, po.robot.environment.cleared_dust, po.robot.collision1, 0.4, 0.6, 0.25)
    population.allIndividuals = population.individuals + population.offsprings
    population.allIndividuals.sort(key=lambda x: x.fitness, reverse=True)
    population.allIndividuals = population.allIndividuals[:-population.n_kill]
    population.individuals = population.allIndividuals
    if killbest:
        population.bestIndividuals = population.individuals[:population.n_bestIndividuals]
        population.offsprings = pair(population.bestIndividuals, 0, 0, "else")
        for o in range(len(population.offsprings)):
            po = population.offsprings[o]
            sensors = list(map(lambda x: (np.e ** ((-(2 * x - 200)) / 37)) / 10, po.robot.get_sensors()))
            population.offsprings[o].fitness = population.offsprings[o].fitnessFunction( \
                [po.robot.left_velocity, po.robot.right_velocity], sensors, po.robot.environment.cleared_dust,
                po.robot.collision1, 0.25, 0.5, 0.25)
        population.individuals = population.individuals[population.n_kill:]
        population.allIndividuals = population.individuals + population.offsprings
        population.individuals = population.allIndividuals

    #population.scale -= population.decreaseFactorMutation

def rastrigin(genotype):
  sigma = 0
  for i in genotype:
    sigma = sigma + (i * i - 10 * math.cos(2 * math.pi * i))
  return 20 + sigma

def rosenbrock(genotype):
  return np.square(1 - genotype[0]) + 100 * np.square((genotype[1] - genotype[0] * genotype[0]))

benchmarkFunction = rastrigin
population = Population(30, 18, 1, 3, 9, n_epochs, 0.1, 0.00001)

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
updateEpoch(population, False)
show_objects = True
n_original_epoch = n_epochs

from pprint import pprint

if __name__ == "__main__":

    while not done:
        time_seconds = (pygame.time.get_ticks() / 1000)
        if (pygame.event.get(restartEvent) or condition()):
            updateEpoch(population, False)
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
                elif event.key == pygame.K_s:
                    show_objects = not show_objects

        # print(block.theta)
        # player1.update(block.x, block.y, block.theta, block.radius, 20)
        # player2.update(block.x, block.y, block.theta, block.radius, 0)
        screen.fill((255, 128, 128))
        grid.draw_grid()
        for w in walls:
            w.draw()
        bestcount = 0
        population.individuals.sort(key=lambda x: x.fitness, reverse=True)
        for individual in population.individuals:
        # blit_text(f'L: {block.left_velocity}; R: {block.right_velocity}', 800, 300, SILVER, BLACK)

            individual.update_individual()
            bestcount += 1
            if (bestcount <= 5):
                individual.robot.move(COLOR_FOR_BEST[bestcount - 1], text = str(bestcount))

            else:
                if individual.id == "!@#":
                    pprint(f"{bestcount}. fitness: {individual.fitness}. LV: {individual.robot.left_velocity}, RV: {individual.robot.right_velocity}, Po: {individual.position}")
                    pprint(f"SS: {individual.input} ")
                    pprint(f"IP: {individual.sensors}")
                    individual.robot.move(BLACK, text="!@#")
                else:
                    individual.robot.move()

            if bestcount <= 5:
                pprint(
                    f"{bestcount}. fitness: {individual.fitness}. LV: {individual.robot.left_velocity}, RV: {individual.robot.right_velocity}, Po: {individual.position}")
                pprint(f"SS: {individual.input} ")
                pprint(f"IP: {individual.sensors}")
                individual.robot.draw(display=True)
                individual.robot.draw_direction(display=True)
                # individual.robot.draw_icc(display=True)
                individual.robot.environment.draw_dusts(individual.robot, disaplay=False)
                individual.robot.draw_sensors(display=False)
            elif bestcount > 5:
                individual.robot.draw(display=show_objects)
                individual.robot.draw_direction(display=show_objects)
                # individual.robot.draw_icc(display=show_objects)
                individual.robot.environment.draw_dusts(individual.robot, disaplay=False)
                individual.robot.draw_sensors(display=False)

            blit_text(f"Epoch:{n_epochs}", 100, 100, WHITE, BLACK, 36)

        pygame.display.flip()
        clock.tick(120)
            # pygame.display.update()
for i in range(len(population.history)):
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