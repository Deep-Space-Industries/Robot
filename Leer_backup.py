# Evolutionary Algorithm
import math
import pygame
from pprint import pprint
from NeuralNetwork import *

n_epochs = 400
n_epoch_max_duration_ms = 15000
#sensormax = 500


from robo import *
pygame.init()
font = pygame.font.SysFont("futura", 16)
COLOR_FOR_BEST = [(255,0,10), (0,255,128), (0,0,255), (238,114,114), (255,185,185)]

# Define individual, assign each individual a simulateed robot
class Individual:

    # Initialize with a simulated robot, a neural networks
    def __init__(self, id = None):
        # self.nn = NeuralNetwork(14,[4,3],2, tanh, 0.1)
        self.robot = Robot(random.randint(50, 200), random.randint(50, 200), \
                           random.randint(1,5), random.randint(1,5), 20, walls)
        self.robot.draw()
        self.id = id
        self.robot.draw_sensors()
        self.nn = NeuralNetwork(18,[6],2, tanh, 0.1)
        self.input = None # for testing
        self.sensors = None # for testing
        self.update_individual()

    # Update individual
    def update_individual(self):
        sensors = self.robot.get_sensors()
        # sensors = list(map(lambda x: (np.e ** ((-(2 * x - 200)) / 37)) / 10, sensors))
        self.sensors = sensors
        isensors = [200 - s for s in sensors] # Attach minor values with higher importance
        input = scalerr(isensors, 0, 200, -1, 1) # Scale values
        # input = [np.array([sensors])]
        if len(self.nn.hiddenOutputs) != 0:
            input[0] = np.append(input[0], self.nn.hiddenOutputs) # Use the hidden nodes from previous moment as the input for this moment.
        else:
            input[0] = np.append(input[0], [0, 0, 0, 0, 0, 0])

        self.input = input[0]
        self.position = self.nn.forwardPropagation(input[0])  # "self.Position" is the velocities
        # self.position = scalerr(self.position[0], -1, 1, -10, 10)[0]
        self.robot.update_velocities(round(self.position[0][0]*30, 4), round(self.position[0][1]*30, 4))
        self.robot.update_icc() # Update ICC after updating velocities
        velocity = [self.robot.left_velocity, self.robot.right_velocity]
        collision = self.robot.collision1
        self.fitness = self.fitnessFunction(velocity, input[0], self.robot.environment.cleared_dust, collision, 0.4, 0.6, 0.25) # Update fitness

    def fitnessFunction(self, velocity, sensor, dust, collision, w1, w2, w3):
        averageVelocity = (velocity[0] + velocity[1]) / 2
        deltaVelocity = abs(velocity[0] - velocity[1])
        maxSensor = max(sensor)
        minSensor = min(self.sensors)
        penalty = 0.2
        ipt = abs(minSensor - 200) * 2
        # f1 = 10 * w1 * averageVelocity * (1 - math.sqrt(deltaVelocity)) * (1 - maxSensor)
        f1 = averageVelocity * (1 - math.sqrt(deltaVelocity)) * (1 - maxSensor) /4
        f2 =  np.sqrt(dust)
        f3 = -collision /5
        fitness = f2 + f1 + f3
        #
        # if minSensor < 35:
        #     fitness = fitness -  (35 - minSensor)
        print(f"speed: {f1}, dust: {f2}, collision: {f3}, mins: {minSensor}. combine: {fitness}")
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
        random.shuffle(bestIndividuals) # If method is "random", then shuffle the parents before pairing
    offsprings = []
    for i in range(len(bestIndividuals))[::2]:
        newOffspring = Individual()
        # Genes of parents being passed as average value
        newOffspring.nn.weightsIH=(bestIndividuals[i].nn.weightsIH+bestIndividuals[i+1].nn.weightsIH)/2
        newOffspring.nn.biasIHH[0]=(bestIndividuals[i].nn.biasIHH[0]+bestIndividuals[i+1].nn.biasIHH[0])/2
        newOffspring.nn.biasHO = (bestIndividuals[i].nn.biasHO + bestIndividuals[i + 1].nn.biasHO) / 2
        newOffspring.nn = mutate(newOffspring)
        offsprings.append(newOffspring)
    return offsprings

# Mutation, with mutate rate population.scale.
def mutate(offspring):
    # Add a weight matrix with random value to the current weight matrix and bias
    offspring.nn.weightsIH += randomWeights(np.zeros((offspring.nn.inputnodes, offspring.nn.hiddennodes[0]), dtype=float))*population.scale
    offspring.nn.biasIHH[0] += randomWeights(np.zeros((1, offspring.nn.hiddennodes[0]), dtype=float))*population.scale
    # for l in range(len(offspring.nn.weightsHH)-1):
    #     offspring.nn.weightsHH[l] += randomWeights(np.zeros((offspring.nn.hiddennodes[l], offspring.nn.hiddennodes[l+1]), dtype=float))*population.scale
    #     offspring.nn.biasIHH[l+1] += randomWeights(np.zeros((1, offspring.nn.hiddennodes[l+1]), dtype=float))*population.scale
    # offspring.nn.weightsHO += randomWeights(np.zeros((offspring.nn.hiddennodes[1], offspring.nn.outputnodes), dtype=float))*population.scale
    offspring.nn.biasHO += randomWeights(np.zeros((1, offspring.nn.outputnodes), dtype=float))*population.scale
    return offspring.nn

# Update epoch
def updateEpoch(population, killbest):
    population.individuals.sort(key=lambda x: x.fitness, reverse=True)
    population.history.append([k.fitness for k in population.individuals])
    population.historyWeightsIH.append([k.nn.weightsIH for k in population.individuals])
    population.historyBiasIHH1.append([k.nn.biasIHH[0] for k in population.individuals])
    # population.historyWeightsHH.append([k.nn.weightsHH[0] for k in population.individuals])
    # population.historyBiasIHHn.append([k.nn.biasIHH[1] for k in population.individuals])
    population.historyWeightsHO.append([k.nn.weightsHO for k in population.individuals])
    population.historyBiasHO.append([k.nn.biasHO for k in population.individuals])

    # Saving ANN's real-time weights to local file
    np.save("n/FT", population.history, allow_pickle=True, fix_imports=True)
    np.save("n/IH", population.historyWeightsIH, allow_pickle=True, fix_imports=True)
    np.save("n/BIHH", population.historyBiasIHH1, allow_pickle=True, fix_imports=True)
    # np.save("n/HH", population.historyWeightsHH, allow_pickle=True, fix_imports=True)
    # np.save("n/BIHHn", population.historyBiasIHHn, allow_pickle=True, fix_imports=True)
    np.save("n/HO", population.historyWeightsHO, allow_pickle=True, fix_imports=True)
    np.save("n/BHO", population.historyBiasHO, allow_pickle=True, fix_imports=True)

    population.bestIndividuals = population.individuals[:population.n_bestIndividuals]
    population.offsprings = pair(population.bestIndividuals, 0, 0, "else")
    for o in range(len(population.offsprings)):
        # Initialize offspring's fitness
        po = population.offsprings[o]
        sensors = list(map(lambda x: (np.e ** ((-(2 * x - 200)) / 37)) / 10, po.robot.get_sensors()))
        population.offsprings[o].fitness = population.offsprings[o].fitnessFunction(\
            [po.robot.left_velocity, po.robot.right_velocity], sensors, po.robot.environment.cleared_dust, po.robot.collision1, 0.25, 0.5, 0.25)
    population.individuals = population.individuals[:-population.n_kill]
    population.allIndividuals = population.individuals + population.offsprings
    population.allIndividuals.sort(key=lambda x: x.fitness, reverse=True)
    population.individuals = population.allIndividuals

# Initiazlied population
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
pygame.time.set_timer(restartEvent, n_epoch_max_duration_ms) # Each epoch lasts for n_epoch_max_duration_ms milliseconds
updateEpoch(population, False)
show_objects = True
n_original_epoch = n_epochs

show_best_ones = False
show_best = False

if __name__ == "__main__":
    # pygame part
    while not done:
        time_seconds = (pygame.time.get_ticks() / 1000)
        if (pygame.event.get(restartEvent) or condition()):
            updateEpoch(population, False)
            n_epochs -= 1
            reset()
        if (n_epochs <= 0):
            done = True

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    done = True
                elif event.key == pygame.K_s:
                    show_objects = not show_objects
                elif event.key == pygame.K_b:
                    show_best = not show_best

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
                pprint(
                    f"{bestcount}. fitness: {individual.fitness}. LV: {individual.robot.left_velocity}, RV: {individual.robot.right_velocity}, Po: {individual.position}")
                # pprint(f"SS: {individual.input} ")
                # pprint(f"IP: {individual.sensors}")
            else:
                individual.robot.move()


            if bestcount <= 5:
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

            blit_text(f"Epoch:{n_epochs}", 900, 100, WHITE, BLACK, 36)

        pygame.display.flip()
        clock.tick(120)
            # pygame.display.update()
for i in range(len(population.history)):
    print("Epoch", (i+1), "Final fitness", population.history[i])

