# This code was written in pair-programming style
# Members: Berat Cakir, Koushik Haridasyam, Zhangyi Wu

# Import packages
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from matplotlib import ticker
from NeuralNetwork import *
#from robo import *
from scipy.spatial.distance import hamming

# Parameters for dimensionality and distribution of positions
mu = 0.0
std = 5.0
dim = 2
scale = 0.25

# Define class individual and its variables
class Individual:
    def __init__(self, benchmarkFunction,walls,screen):
        self.walls = walls
        self.screen = screen
        self.nn=NeuralNetwork(12,[4,3],2, tanh, 0.1)
        self.robot=Robot(random.randint(220, 290) , random.randint(220, 290) , 2 , 3 , 20 , self.walls, self.screen)
        # self.genotype = encoder(np.random.normal(loc=mu, scale=std, size=dim))
        # self.benchmarkFunction = benchmarkFunction
        # self.fitness = self.benchmarkFunction(decoder(self.genotype))

# Create n new offsprings by m number of individuals
def pair(bestIndividuals, n_offsprings, m_parents, method, benchmarkFunction,walls,screen):
    # Pair individuals randomly
    if (method == "random"):
        random.shuffle(bestIndividuals)
    # Else: pair best individuals with best individuals
    offsprings = []
    for i in range(len(bestIndividuals))[::2]:
        # Generates n new offsprings with mutation value added
        newOffspring = Individual(benchmarkFunction,walls,screen)
        # Genes of parents being passed as average value
        newOffspring.nn.weightsIH=(bestIndividuals[i].nn.weightsIH+bestIndividuals[i+1].nn.weightsIH)/2
        newOffspring.nn.biasIHH[0]=(bestIndividuals[i].nn.biasIHH[0]+bestIndividuals[i+1].nn.biasIHH[0])/2
        newOffspring.nn.weightsHH[0] = (bestIndividuals[i].nn.weightsHH[0] + bestIndividuals[i + 1].nn.weightsHH[0]) / 2
        newOffspring.nn.biasIHH[1] = (bestIndividuals[i].nn.biasIHH[1] + bestIndividuals[i + 1].nn.biasIHH[1]) / 2
        newOffspring.nn.weightsHO = (bestIndividuals[i].nn.weightsHO + bestIndividuals[i + 1].nn.weightsHO) / 2
        newOffspring.nn.biasHO = (bestIndividuals[i].nn.biasHO + bestIndividuals[i + 1].nn.biasHO) / 2
        # newOffspring.genotype = (decoder(bestIndividuals[i].genotype)+decoder(bestIndividuals[i+1].genotype))/2
        # Mutation being added to the new gene
        newOffspring.nn = mutate(newOffspring)
        offsprings.append(newOffspring)
    return offsprings

# Add mutation to genes
def mutate(offspring):
    offspring.nn.weightsIH += randomWeights(np.zeros((offspring.nn.inputnodes, offspring.nn.hiddennodes[0]), dtype=float))
    return offspring.nn

# Define class population and its variables
class Population:
    def __init__(self, n_individuals, n_bestIndividuals, n_offsprings, m_parents, n_kill, n_epochs, decreaseFactorMutation, benchmarkFunction, scale=scale):
        # Create initial population
        self.n_individuals = n_individuals
        self.benchmarkFunction = benchmarkFunction
        self.individuals = [Individual(self.benchmarkFunction,None,None) for i in range(n_individuals)]
        self.decreaseFactorMutation = decreaseFactorMutation
        # Saving positions into history
        # self.history=[]
        # Evolving for n number of epochs
        for i in range(n_epochs):
            print("Epoch",i)
            # self.history.append([decoder(k.genotype) for k in self.individuals])
            # self.individuals.sort(key=lambda x: x.fitness, reverse=False)
            #for j in range(len(self.individuals)):
                #print("Individual",j)
            pygame.init()
            font = pygame.font.SysFont("futura", 16)
            width = 1000
            height = 1000
            clock = pygame.time.Clock()
            screen = pygame.display.set_mode((width, height),
                                             pygame.DOUBLEBUF | pygame.HWSURFACE | pygame.FULLSCREEN)
            walls = []
            east_border = Wall((width - 5, 0), (width - 5, height - 5), LIGHTBLUE, screen)
            west_border = Wall((5, 5), (5, height - 5), LIGHTBLUE, screen)
            south_border = Wall((5, height - 5), (width - 5, height - 5), LIGHTBLUE, screen)
            north_border = Wall((5, 5), (width - 5, 5), LIGHTBLUE, screen)
            walls.append(Wall((250, 250), (750, 250), LIGHTBLUE, screen))
            walls.append(Wall((750, 250), (750, 750), LIGHTBLUE, screen))
            walls.append(Wall((750, 750), (250, 750), LIGHTBLUE, screen))
            walls.append(Wall((250, 750), (250, 250), LIGHTBLUE, screen))
            # walls.append(Wall((100, 200), (400, 300), LIGHTBLUE))
            # walls.append(Wall((600, 500), (800, 900), LIGHTBLUE))
            # walls.append(Wall((300, 500), (300, 750), LIGHTBLUE))
            # walls.append(Wall((600, 400), (600, 805), LIGHTBLUE))
            walls.append(east_border)
            walls.append(west_border)
            walls.append(south_border)
            walls.append(north_border)
            e = Environment(screen, width,height,1)
            loopExit = True
            crash = False
            screen.blit(pygame.transform.scale(screen, (1000, 1000)), (0, 0))
            try:
                while loopExit:
                    for r in self.individuals:
                        r.robot.walls = walls
                        r.robot.screen = screen

                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            loopExit = False
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_w:
                                #self.individuals[j].robot.speedup_left()
                                #self.individuals[j].robot.update_icc()
                                crash = True
                            elif event.key == pygame.K_s:
                                self.individuals[j].robot.slowdown_left()
                                self.individuals[j].robot.update_icc()
                            elif event.key == pygame.K_o:
                                self.individuals[j].robot.speedup_right()
                                self.individuals[j].robot.update_icc()
                            elif event.key == pygame.K_l:
                                self.individuals[j].robot.slowdown_right()  # decrement of left wheel
                                self.individuals[j].robot.update_icc()
                            elif event.key == pygame.K_x:
                                self.individuals[j].robot.stop_both()  # zero both wheel speed
                                self.individuals[j].robot.update_icc()
                            elif event.key == pygame.K_t:
                                self.individuals[j].robot.speedup_both()  # increment both wheel speed
                                self.individuals[j].robot.update_icc()
                            elif event.key == pygame.K_g:
                                self.individuals[j].robot.slowdown_both()  # decrement both wheel speed
                                self.individuals[j].robot.update_icc()
                            elif event.key == pygame.K_ESCAPE:
                                loopExit = False
                    if(crash):
                        loopExit = False
                    screen.fill(BLACK)
                    # print(block.theta)
                    # player1.update(block.x, block.y, block.theta, block.radius, 20)
                    # player2.update(block.x, block.y, block.theta, block.radius, 0)
                    screen.fill((255, 128, 128))

                    #blit_text(f'L: {self.individuals[j].robot.left_velocity}; R: {self.individuals[j].robot.right_velocity}', 800, 300, SILVER, BLACK)
                    # screen.blit(player2.image, player2.rect)
                    for r in self.individuals:
                        r.robot.move()
                        r.robot.draw_direction()
                    for w in walls:
                        w.draw()
                    for r in self.individuals:
                        r.robot.draw_icc()
                        r.robot.draw_sensors()
                        e.draw_dusts(r.robot)

                    # pygame.display.flip()
                    clock.tick(120)
                    pygame.display.update()
                pygame.quit()
            except SystemExit:
                pygame.quit()
                #print("Vel:",self.individuals[j].robot.left_velocity, self.individuals[j].robot.right_velocity)
            print("hierr")
            self.bestIndividuals = self.individuals[:n_bestIndividuals]
            # print(self.bestIndividuals)
            # Pairing of n individuals
            self.offsprings = pair(self.bestIndividuals, n_offsprings, m_parents, "else", self.benchmarkFunction,walls,screen)
            # Calculate fitness of new offsprings
            # for o in self.offsprings:
                # o.fitness = self.benchmarkFunction(decoder(o.genotype))
            # Add new offsprings to the total population
            self.allIndividuals = self.individuals + self.offsprings
            # self.allIndividuals.sort(key=lambda x: x.fitness, reverse=False)
            # Let n number of individuals die
            self.allIndividuals = self.allIndividuals[:-n_kill]
            self.individuals = self.allIndividuals
            scale -= decreaseFactorMutation
            print("Ende")
            # for i in self.individuals:
            #     i.nn.print()

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
n_epochs = 2
benchmarkFunction = rastrigin
myPop = Population(10, 6, 1, 3, 3, n_epochs, 0.00001, benchmarkFunction)
