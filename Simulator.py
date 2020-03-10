import math
import pygame
from NeuralNetwork import *
from Robot import *

# Initialize Simulator
pygame.init()
font = pygame.font.SysFont("futura", 16)
# Colors for coloring the individuals with highest fitness scores
COLOR_FOR_BEST = [(255, 0, 10), (0, 255, 128), (0, 0, 255), (238, 114, 114),
                  (255, 185, 185)]

# Define individual class and its variables
class Individual:
    def __init__(self):
        # Parameters of 13 inputs, 4 hidden nodes, 2 output nodes and tanh activation funcion
        self.nn = NeuralNetwork(16, [4], 2, tanh)
        # Each individual is assigned a simulated robot
        self.robot = Robot(random.randint(100, 1000), random.randint(100, 1000), \
                           0, 0, 20, walls)
        self.robot.draw()
        self.robot.draw_sensors()
        sensors = self.robot.get_sensors() # Get sensor values from the robot
        input = [np.array([sensors])]
        input = valueScaler(input[0], 0, 200, -3, 3)  # Scale sensor values
        velocities = [self.robot.left_velocity, self.robot.right_velocity]
        velocities = valueScaler(velocities, -15, 15, -1, 1)  # Scale velocities
        input[0] = np.append(input[0], randomWeights([[0,0,0,0]])[0]) # Init: passing random values
        self.position = self.nn.forwardPropagation(input[0])  # Position contains the actual velocities
        self.position = valueScaler(self.position[0], -1, 1, -15, 15)[0]  # Scale velocities
        self.robot.update_velocities(self.position[0], self.position[1])  # Update robot's velocities
        self.robot.update_icc()  # Update ICC after updating velocities
        self.environment = Environment(density=1)
        self.fitness = self.environment.cleared_dust

    # Update values of individual
    def update_individual(self):
        sensors = self.robot.get_sensors()
        input = [np.array([sensors])]
        input = valueScaler(input[0], 0, 200, -3, 3) # Scale sensor values
        velocities = [self.robot.left_velocity, self.robot.right_velocity]
        velocities = valueScaler(velocities, -15, 15, -1, 1)
        # Use hidden nodes results from previous time-step as the input for current time-step
        input[0] = np.append(input[0], self.nn.hiddenOutputs)
        self.position = self.nn.forwardPropagation(input[0]) # Position contains the actual velocities
        self.position = valueScaler(self.position[0], -1, 1, -15, 15)[0] # Scale velocities
        self.robot.update_velocities(round(self.position[0], 4),
                                     round(self.position[1], 4))  # Update robot's velocities
        self.robot.update_icc()  # Update ICC after updating velocities
        velocity = [self.robot.left_velocity, self.robot.right_velocity]
        collision = self.robot.collision1
        self.fitness = self.fitnessFunction(velocity, sensors, self.robot.environment.cleared_dust,
                                            collision, 0.15, 0.35, 0.15, 0.1, 0.2, 0.05) # Update fitness

    def fitnessFunction(self, velocity, sensor, dust, collision, w1, w2, w3, w4, w5, w6):
        scaledVelocities = valueScaler([velocity[0], velocity[1]], -15, 15, 0, 1)[0]
        averageVelocity = (scaledVelocities[0] + scaledVelocities[1]) / 2
        deltaVelocity = abs(scaledVelocities[0] - scaledVelocities[1])
        maxSensor = max(sensor) / 200
        minSensor = min(sensor)
        importance = abs(minSensor - 200)

        velocityScore = w1 * ((averageVelocity * (1 - math.sqrt(deltaVelocity))) ** 2)
        dustScore = w2 * dust
        rotationScore = w3 * (deltaVelocity * importance)

        if (minSensor > 5):
            distanceScore = w4 * math.sqrt((30 + dustScore))
        else:
            distanceScore = 0

        if (deltaVelocity >= 0.5 * random.uniform(0, 1)):
            explorationScore = w5 * (10 + dustScore)
        else:
            explorationScore = 0

        errorRate = w6 * math.sqrt((math.sqrt(importance) + (dustScore)))

        print("velocityScore", velocityScore)
        print("dustScore", dustScore)
        print("rotationScore", rotationScore)
        print("distanceScore", distanceScore)
        print("explorationScore", explorationScore)
        print("errorRate", errorRate)

        fitness = velocityScore + dustScore + rotationScore + distanceScore + explorationScore - errorRate
        print("fitness", fitness)

        return fitness

# Define population class and its variables
class Population:
    def __init__(self, n_individuals, n_bestIndividuals, n_offsprings, m_parents, n_kill, n_epochs, scale,
                 decreaseFactorMutation):
        # Properties of population
        self.n_individuals = n_individuals
        self.n_bestIndividuals = n_bestIndividuals
        self.n_offsprings = n_offsprings
        self.m_parents = m_parents
        self.n_kill = n_kill
        self.n_epochs = n_epochs
        self.scale = scale
        self.decreaseFactorMutation = decreaseFactorMutation

        # Initialize n_individuals individuals
        self.individuals = [Individual() for i in range(n_individuals)]

        # History for fitness function
        self.history = []

        # History of ANN
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
    # Pair individuals randomly
    if (method == "random"):
        random.shuffle(bestIndividuals)  # If method is "random", then shuffle the parents before pairing
    # Else: pair best individuals with best individuals
    offsprings = []
    for i in range(len(bestIndividuals))[::2]:
        # Initialize new offspring
        newOffspring = Individual()
        # Perform arithmetic crossover. Perform mean to parents in order to generate offspring.
        newOffspring.nn.weightsIH = (bestIndividuals[i].nn.weightsIH + bestIndividuals[i + 1].nn.weightsIH) / 2
        newOffspring.nn.biasIHH[0] = (bestIndividuals[i].nn.biasIHH[0] + bestIndividuals[i + 1].nn.biasIHH[0]) / 2
        newOffspring.nn.weightsHO = (bestIndividuals[i].nn.weightsHO + bestIndividuals[i + 1].nn.weightsHO) / 2
        newOffspring.nn.biasHO = (bestIndividuals[i].nn.biasHO + bestIndividuals[i + 1].nn.biasHO) / 2
        # Mutate offspring
        newOffspring.nn = mutate(newOffspring)
        offsprings.append(newOffspring)
    return offsprings

# Perform mutation, with mutation rate
def mutate(offspring):
    # Add a weight matrix with random value to the current weight matrix and bias
    offspring.nn.weightsIH += randomWeights(
        np.zeros((offspring.nn.inputnodes, offspring.nn.hiddennodes[0]), dtype=float)) * population.scale
    offspring.nn.biasIHH[0] += randomWeights(np.zeros((1, offspring.nn.hiddennodes[0]), dtype=float)) * population.scale
    offspring.nn.weightsHO += randomWeights(
        np.zeros((offspring.nn.hiddennodes[0], offspring.nn.outputnodes), dtype=float)) * population.scale
    offspring.nn.biasHO += randomWeights(np.zeros((1, offspring.nn.outputnodes), dtype=float)) * population.scale
    return offspring.nn

# Update epoch
def updateEpoch(population):
    population.individuals.sort(key=lambda x: x.fitness, reverse=True)  # Sorting individuals in the increasing order

    population.history.append([k.fitness for k in population.individuals])
    population.historyWeightsIH.append([k.nn.weightsIH for k in population.individuals])
    population.historyBiasIHH1.append([k.nn.biasIHH[0] for k in population.individuals])
    population.historyWeightsHO.append([k.nn.weightsHO for k in population.individuals])
    population.historyBiasHO.append([k.nn.biasHO for k in population.individuals])

    # Saving real-time ANN weights to local file
    np.save("/FT", population.history, allow_pickle=True, fix_imports=True)
    np.save("/IH", population.historyWeightsIH, allow_pickle=True, fix_imports=True)
    np.save("/BIHH", population.historyBiasIHH1, allow_pickle=True, fix_imports=True)
    np.save("/HO", population.historyWeightsHO, allow_pickle=True, fix_imports=True)
    np.save("/BHO", population.historyBiasHO, allow_pickle=True, fix_imports=True)

    population.bestIndividuals = population.individuals[:population.n_bestIndividuals]
    population.offsprings = pair(population.bestIndividuals, 0, 0, "Else")

    for o in range(len(population.offsprings)):
        # Initialize offspring and its fitness
        po = population.offsprings[o]
        sensors = po.robot.get_sensors()
        population.offsprings[o].fitness = population.offsprings[o].fitnessFunction( \
            [po.robot.left_velocity, po.robot.right_velocity], sensors, po.robot.environment.cleared_dust,
            po.robot.collision1, 0.15, 0.35, 0.15, 0.1, 0.2, 0.05)
    population.individuals = population.individuals[:-population.n_kill] # Kill poor performers
    population.allIndividuals = population.individuals + population.offsprings # Add new offspring to population
    population.individuals = population.allIndividuals

# 500 epochs, each lasting for 60 seconds
n_epochs = 500
n_epoch_max_duration_ms = 60000

# Initialize population
population = Population(15, 8, 1, 2, 4, n_epochs, 0.05, 0.00001)

# Start simulator
done = False
clock = pygame.time.Clock()
restartEvent = pygame.USEREVENT + 1
pygame.time.set_timer(restartEvent,
                      n_epoch_max_duration_ms)  # Each epoch lasts for n_epoch_max_duration_ms milliseconds
updateEpoch(population)
show_objects = True
show_every_one = True
show_best_ones = True
show_best = True
n_original_epoch = n_epochs

# Pygame start running
while not done:
    time_seconds = (pygame.time.get_ticks() / 1000)
    # Restart simulation after n time passes
    if (pygame.event.get(restartEvent)):
        updateEpoch(population)
        n_epochs -= 1
    if (n_epochs <= 0):
        done = True

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                done = True
            elif event.key == pygame.K_c:
                show_every_one = not show_every_one
                show_best_ones = not show_best_ones
            elif event.key == pygame.K_v:
                show_best_ones = not show_best_ones

    screen.fill((255, 128, 128))
    grid.draw_grid()
    for w in walls:
        w.draw()
    bestcount = 0
    population.individuals.sort(key=lambda x: x.fitness, reverse=True)

    for individual in population.individuals:
        individual.update_individual()
        bestcount += 1
        if (bestcount <= 5):
            individual.robot.move(COLOR_FOR_BEST[bestcount - 1], text=str(bestcount))
        else:
            individual.robot.move()

        if bestcount == 1:
            individual.robot.draw(display=True)
            individual.robot.draw_direction(display=True)
            # individual.robot.draw_icc(display=True)
            individual.robot.environment.draw_dusts(individual.robot, disaplay=False)
            individual.robot.draw_sensors(display=False)
        elif 1 < bestcount <= 3:
            to_show = bool(show_best_ones or show_every_one)
            individual.robot.draw(display=to_show)
            individual.robot.draw_direction(display=to_show)
            # individual.robot.draw_icc(display=True)
            individual.robot.environment.draw_dusts(individual.robot, disaplay=False)
            individual.robot.draw_sensors(display=False)
        elif bestcount > 5:
            to_show1 = bool(show_every_one and show_best_ones)
            individual.robot.draw(display=to_show1)
            individual.robot.draw_direction(display=to_show1)
            # individual.robot.draw_icc(display=show_objects)
            individual.robot.environment.draw_dusts(individual.robot, disaplay=False)
            individual.robot.draw_sensors(display=False)
    blit_text(f"Epoch:{n_original_epoch-n_epochs+1}", 100, 900, WHITE, BLACK, 36)

    pygame.display.flip()
    clock.tick(120)

# Print final fitness scores
for i in range(len(population.history)):
    print("Epoch", (i + 1), "Final fitness", population.history[i])