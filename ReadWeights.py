import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from NeuralNetwork import *
from Simulator import *
from Robot import *

fitness = np.load("NN/FT.npy")
historyWeightsIH = np.load("NN/IH.npy") # input nodes to hidden one
historyBiasIHH1 = np.load("NN/BIHH.npy") # biased to hidden one
historyWeightsHO = np.load("NN/HO.npy") # hidden two to output
historyBiasHO = np.load("NN/BHO.npy") # bias to output

# Save fitness and diversity of individuals
maxFitness = []
avgFitness = []
maxDiversity = []
avgDiversity = []
n_epochs = 500

IH = historyWeightsIH[-1][0]
IHH1 = historyBiasIHH1[-1][0] # biased to hidden one
HO = historyWeightsHO[-1][0] # hidden two to output
BiasHO = historyBiasHO[-1][0] # bias to output

nn = NeuralNetwork(16, [4], 2, tanh)
nn.weightsIH = IH
nn.biasIHH[0] = IHH1
nn.weightsHO = HO
nn.biasHO = BiasHO

# print(fitness.shape)
individual = Individual()
individual.nn = nn

# Compute the max fitness, average fitness, max diversity and average diversity through out all generations

def draw_fitness(fitness_history, display = False):
    for t in range(n_epochs):
        t_th_population = fitness_history[t]
        # Get maximum and average fitness of each iteration
        maxFitness.append(max(t_th_population))
        avgFitness.append(sum(t_th_population) / len(t_th_population))

    if display:
        # Show fitness plots
        plt.subplots(figsize=(20, 10))
        plt.plot(range(len(maxFitness)), maxFitness)
        plt.plot(range(len(avgFitness)), avgFitness)
        plt.title("Maximum and average fitness values of individuals for each generation", fontsize = 20)
        plt.legend(["Maximum fitness", "Average fitness"])
        plt.savefig("fitness.png", dpi=150)
        plt.show()

print(historyWeightsIH[0].shape)
for i in historyWeightsIH[0]:
    print(i)

def draw_diversity(population = 15, display = False):
    for t in range(n_epochs):
        print(t)
        best_generation_diversity = -1
        total_diversity = 0
        t1 = historyWeightsIH[t]
        t2 = historyBiasIHH1[t]
        t5 = historyWeightsHO[t]
        t6 = historyBiasHO[t]
        for i in range(population):
            individual_diversity = 0
            for j in range(population):
                individual_diversity += matrix_distance(t1[i], t1[j])
                individual_diversity += matrix_distance(t2[i], t2[j])
                individual_diversity += matrix_distance(t5[i], t5[j])
                individual_diversity += matrix_distance(t6[i], t6[j])
                if individual_diversity > best_generation_diversity:
                        best_generation_diversity = individual_diversity
            total_diversity += individual_diversity
        avgDiversity.append(total_diversity / population)
        maxDiversity.append(best_generation_diversity)

    if display:
        # Show diversity plots
        plt.subplots(figsize=(20, 10))
        plt.subplot(211)
        plt.plot(range(len(maxDiversity)), maxDiversity)
        plt.title("Maximum diversity of population for each generation", fontsize = 20)
        plt.subplot(212)
        plt.plot(range(len(avgDiversity)), avgDiversity)
        plt.title("Average diversity of population for each generation", fontsize =20)
        plt.savefig("diversity.png", dpi=150)
        plt.show()
        print("printed")

def matrix_distance(matrix1, matrix2):
    return np.sum(np.abs(matrix1 - matrix2))

if _name_ == "_main_":
    draw_fitness(fitness, display=True)
    draw_diversity(display=True)
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    done = True
                elif event.key == pygame.K_s:
                    individual = Individual()
                    individual.nn = nn

        screen.fill((255, 128, 128))
        grid.draw_grid()
        for w in walls:
            w.draw()
        bestcount = 0

        individual.update_individual()
        bestcount += 1
        if (bestcount <= 5):
            individual.robot.move(COLOR_FOR_BEST[bestcount - 1], text = str(bestcount))

        else:
            individual.robot.move()
        if show_objects:
            individual.robot.draw()
            individual.robot.draw_direction()
            individual.robot.draw_icc()
            individual.robot.environment.draw_dusts(individual.robot, disaplay=True)
            individual.robot.draw_sensors(display=False)
            print(f"fitness: {individual.fitness}. LV: {individual.robot.left_velocity}, RV: {individual.robot.right_velocity}, Po: {individual.position}")
        # blit_text(f"Epoch:{n_epochs}", 100, 100, WHITE, BLACK, 36)

        pygame.display.flip()
        clock.tick(120)