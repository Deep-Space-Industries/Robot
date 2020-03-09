import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from NeuralNetwork import *
from Leer import *
from robo import *
fitness = np.load("n/FT.npy")
historyWeightsIH = np.load("n/IH.npy") # input nodes to hidden one
historyBiasIHH1 = np.load("n/BIHH.npy") # biased to hidden one
historyWeightsHH = np.load("n/HH.npy") # hidden one to hidden two
historyBiasIHHn = np.load("n/BIHHn.npy") # biased to hidden two
historyWeightsHO = np.load("n/HO.npy") # hidden two to output
historyBiasHO = np.load("n/BHO.npy") # bias to output
historyWeights = [historyWeightsIH, historyBiasHO, historyBiasIHH1, historyBiasIHHn, historyWeightsHH, historyWeightsHO]

# Save fitness and diversity of individuals
maxFitness = []
avgFitness = []
maxDiversity = []
avgDiversity = []
n_epochs = 40

table = pd.DataFrame(data=(zip(fitness[-1], historyWeightsIH[-1])))
print(table)

[w[-1] for w in historyWeights]
IH = historyWeightsIH[-1][0]
IHH1 = historyBiasIHH1[-1][0] # biased to hidden one
HH = historyWeightsHH[-1][0] # hidden one to hidden two
BiasIHHn = historyBiasIHHn[-1][0] # biased to hidden two
HO = historyWeightsHO[-1][0] # hidden two to output
BiasHO = historyBiasHO[-1][0] # bias to output

nn = NeuralNetwork(14, [3, 2], 2, tanh, 0.1)
nn.weightsIH = IH
nn.biasIHH[0] = IHH1
nn.weightsHH[0] = HH
nn.biasIHH[1] = BiasIHHn
nn.weightsHO = HO
nn.biasHO = BiasHO

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
        plt.title("Maximum and average fitness values of individuals for each generation")
        plt.legend(["Maximum fitness", "Average fitness"])
        plt.show()

def draw_diversity(population = 30, display = False):
    for t in range(n_epochs):
        best_generation_diversity = -1
        total_diversity = 0
        t1 = historyWeightsIH[t]
        t2 = historyBiasIHH1[t]
        t3 = historyWeightsHH[t]
        t4 = historyBiasIHHn[t]
        t5 = historyWeightsHO[t]
        t6 = historyBiasHO[t]
        for i in range(population):
            individual_diversity = 0
            for j in range(population):
                individual_diversity += matrix_distance(t1[i], t1[j])
                individual_diversity += matrix_distance(t2[i], t2[j])
                individual_diversity += matrix_distance(t3[i], t3[j])
                individual_diversity += matrix_distance(t4[i], t4[j])
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
        plt.title("Maximum diversity of population for each generation")
        plt.subplot(212)
        plt.plot(range(len(avgDiversity)), avgDiversity)
        plt.title("Average diversity of population for each generation")
        plt.show()
        print("printed")

def matrix_distance(matrix1, matrix2):
    return np.sum(np.abs(matrix1 - matrix2))

if __name__ == "__main__":
    # draw_fitness(fitness, display=False)
    # draw_diversity(display=False)
    while not done:
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
            individual.robot.environment.draw_dusts(individual.robot, disaplay=False)
            individual.robot.draw_sensors(display=False)
            print(f"fitness: {individual.fitness}. LV: {individual.robot.left_velocity}, RV: {individual.robot.right_velocity}, Po: {individual.position}")
        blit_text(f"Epoch:{n_epochs}", 100, 100, WHITE, BLACK, 36)

        pygame.display.flip()
        clock.tick(120)