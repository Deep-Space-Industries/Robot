import random
import math
import pygame
import pygame.gfxdraw as gfxdraw
import numpy as np
from shapely.geometry import *

WHITE = (255, 255, 255)
GREEN = (20, 255, 140)
GREY = (210, 210, 210)
RED = (255, 0, 0)
PURPLE = (255, 0, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
GOLD = (255,215,0)
SILVER = (192, 192, 192)
LIGHTBLUE = (135,206,250)

scaler = 10
PI = np.pi
cos = np.cos
sin = np.sin
tan = np.tan
arctan = np.arctan
decrease_factor = 1
increase_factor = 1

class Robot:
    def __init__(self, x, y, left_velocity, right_velocity, radius, walls, screen):
        self.x = x
        self.y = y
        self.radius = radius
        self.history = []
        self.color = GREEN
        self.left_velocity = left_velocity
        self.right_velocity = right_velocity
        self.velocity = (self.left_velocity + self.right_velocity) / 2
        self.theta = 0
        self.omega = None
        self.icc_radius = None
        self.icc_centre_x = None
        self.icc_centre_y = None
        self.update_icc()
        self.time = 0
        self.decrease_factor = 1
        self.increase_factor = 1
        self.walls = walls
        self.collision = None
        self.collided_wall = None
        self.to_collide = []
        self.time_step = 1
        self.sensors = []
        self.screen = screen

    def draw(self):
        if len(self.history) > 10:
            for i, h in enumerate(self.history[2:-2]):
                pygame.draw.line(self.screen, BLACK, self.history[i], self.history[i+1], 3)

        gfxdraw.aacircle(self.screen, int(round(self.x)), int(round(self.y)), self.radius, self.color)
        gfxdraw.filled_circle(self.screen, int(round(self.x)), int(round(self.y)), self.radius, self.color)
        self.history.append([int(round(self.x)), int(round(self.y))])
        angle = 0
        self.sensors = []
        while (angle <= 360):
            self.sensors.append(Sensor([self.x + self.radius * cos(self.theta + angle * PI / 180), \
                                       self.y + self.radius * sin(self.theta + angle * PI / 180)]))
            angle = angle + 30

    def draw_sensors(self):
        circle_centre = Point((self.x, self.y))
        for sensor in self.sensors:
            sensor_point = Point((sensor.x, sensor.y))
            sensor.to = None
            pygame.draw.circle(self.screen, BLUE, (int(round(sensor.x)),int(round(sensor.y))), 2)
            sensor_distance = np.inf
            for wall in self.walls:
                px, py = sensor.line_line_intersection(self.x, self.y, wall)
                sensor_reach = Point((px, py))
                point_in_the_wall = wall.linestring.distance(sensor_reach)
                sensor_to_wall = sensor_point.distance(sensor_reach)
                if point_in_the_wall >= 1 or sensor_to_wall >= 200:
                    continue
                if sensor_to_wall < sensor_distance:
                    sensor_distance = sensor_to_wall
                    sensor.observed_wall = wall
                    sensor.to = sensor_reach
                    sensor.dist_to_wall = round(sensor_to_wall, 2)
            if (sensor.to is None):
                blit_text("> 200", sensor.x, sensor.y, self.screen, BLACK, None, 12)
                sensor.value = 200
                continue
            line_from_sensor_to_wall = LineString((sensor_point, sensor.to))
            centre_on_the_line = line_from_sensor_to_wall.distance(circle_centre)
            if centre_on_the_line <= 2:
                blit_text("> 200", sensor.x, sensor.y, self.screen, BLACK, None, 12)
                sensor.value = 200
                continue
            pygame.draw.line(self.screen, SILVER, \
                             (int(round(sensor.x)), int(round(sensor.y))), \
                             (int(round(sensor.to.xy[0][0])), int(round(sensor.to.xy[1][0]))), 2)
            blit_text(f"{sensor.dist_to_wall}", sensor.to.xy[0][0], sensor.to.xy[1][0], self.screen, BLACK, SILVER, 12)
            sensor.value = sensor.dist_to_wall
        # print([x.value for x in self.sensors])
        return

    def draw_icc(self):
        if (self.left_velocity == self.right_velocity):
            return
        pygame.draw.circle(self.screen, PURPLE, [int(round(self.icc_centre_x)), int(round(self.icc_centre_y))], 2)

    def draw_direction(self):
        p1 = (int(round(self.x + .25 * self.radius * cos(self.theta))), \
              int(round(self.y + .25 * self.radius * sin(self.theta))))
        p4 = (int(round(self.x + self.radius * cos(self.theta))), \
            int(round(self.y + self.radius * sin(self.theta))))
        p6 = (int(round(self.x + .25 * self.radius * cos(self.theta - PI / 6))), \
              int(round(self.y + .25 * self.radius * sin(self.theta - PI / 6)))),
        p5 = (int(round(self.x + self.radius * cos(self.theta - PI / 6))), \
              int(round(self.y + self.radius * sin(self.theta - PI / 6))))
        p2 = (int(round(self.x + .25 * self.radius * cos(self.theta + PI / 6))), \
              int(round(self.y + .25 * self.radius * sin(self.theta + PI / 6))))
        p3 = (int(round(self.x + self.radius * cos(self.theta + PI / 6))), \
              int(round(self.y + self.radius * sin(self.theta + PI / 6))))
        pygame.draw.polygon(self.screen, SILVER, [p1, p2, p3, p4, p5, p6])

    def speedup_left(self):
        self.left_velocity += self.increase_factor

    def slowdown_left(self):
        self.left_velocity -= self.decrease_factor

    def speedup_right(self):
        self.right_velocity += self.increase_factor

    def slowdown_right(self):
        self.right_velocity -= self.decrease_factor

    def speedup_both(self):
        self.left_velocity += self.increase_factor
        self.right_velocity += self.increase_factor

    def slowdown_both(self):
        self.left_velocity -= self.decrease_factor
        self.right_velocity -= self.decrease_factor

    def stop_both(self):
        self.right_velocity = self.left_velocity = 0

    def update_icc(self):
        self.velocity = (self.left_velocity + self.right_velocity) / 2
        self.omega = (self.right_velocity - self.left_velocity) / (2 * self.radius)
        self.icc_radius = self.radius * ((self.left_velocity) + (self.right_velocity)) / (
                (self.right_velocity - self.left_velocity) + 0.0001)
        self.icc_centre_x = self.x - self.icc_radius * sin(self.theta)
        self.icc_centre_y = self.y + self.icc_radius * cos(self.theta)
        max_v = max(abs(self.left_velocity), abs(self.right_velocity))
        if (max_v // 5 != 0):
            self.time_step = round(1 / (max_v // 5), 2)
        else:
            self.time_step = 1
        self.time_step = 1/1

    def update_pos(self, x = None, y = None, theta = None, timestep = 1):
        if x is None:
            x = self.x
            y = self.y
            theta = self.theta
            # timestep = self.time_step

        if (self.left_velocity == self.right_velocity):
            new_x = x + timestep * (self.velocity * cos(theta))
            new_y = y + timestep * (self.velocity * sin(theta))
            new_theta = theta + self.omega * timestep
        else:
            p = np.dot(np.array([[cos(self.omega * timestep), -sin(self.omega * timestep), 0],
                                 [sin(self.omega * timestep), cos(self.omega * timestep), 0],
                                 [0, 0, 1]]), \
                       np.array(
                           [self.icc_radius * sin(theta),
                            -self.icc_radius * cos(theta),
                            theta]))
            new_x = p[0] + self.icc_centre_x
            new_y = p[1] + self.icc_centre_y
            new_theta = p[2] + self.omega * timestep
        return [new_x, new_y, new_theta]

    def move(self):
        prev_x, prev_y, prev_theta = self.x, self.y, self.theta
        self.collision = False
        self.collided_wall = None
        self.collision_num = 0
        self.color = GREEN

        self.ppdistance_to_each_wall()

        rColliding = self.get_positions_new(self.x, self.y, self.theta)
        sColliding = self.get_positions_new(self.x, self.y, self.theta, slide = True)

        #
        if (not sColliding and not rColliding) or (sColliding and not rColliding):
            self.x, self.y, self.theta = self.update_pos()
            self.draw()
            return
        elif (rColliding and not sColliding):
            self.x, self.y, self.theta = self.slide()
            self.color = GREY
            self.update_icc()
            self.draw()
            return
        elif (sColliding and rColliding):
            self.theta += self.omega
            self.color = BLACK
            self.update_icc()
            self.draw()
            return
        return


    def slide(self, x = None, y = None, theta = None, time_step = 1):
        along_wall = None
        d = np.inf
        if len(self.to_collide) == 1:
            #print("1")
            along_wall = self.to_collide[0]
        else:
            #print("2")
            for w in self.to_collide:
                print(f"{w.dist}D")
                if w.dist < d:
                    print(f"{w.dist} < {d}")
                    d = w.dist
                    along_wall = w
        if x is None:
            x = self.x
            y = self.y
            theta = self.theta
        theta1 = along_wall.angle
        # theta1 = self.collided_wall.angle
        direction = cos(theta1 - theta)

        if -0.005 <= direction <= 0.005 or direction >= 0.995 or direction <= -0.995:
            next_x, next_y, next_theta = self.update_pos(x, y, theta, time_step)
            return next_x, next_y, next_theta

        if direction <= 0.0:
            next_x = x + time_step * (self.velocity * - cos(theta1))
            next_y = y + time_step * (self.velocity * - sin(theta1))
        if direction >= 0.0:
            next_x = x + time_step * (self.velocity * cos(theta1))
            next_y = y + time_step * (self.velocity * sin(theta1))
        next_theta = theta + self.omega * time_step
        return next_x, next_y, next_theta

    def ppdistance_to_each_wall(self):
        point = Point(self.x, self.y)
        closest_dist = np.inf
        closest_wall = None
        self.to_collide = []
        for wall in self.walls:
            project = wall.linestring.project(point)
            nearest_point = wall.linestring.interpolate(project).coords
            point2 = Point([nearest_point[0][0], nearest_point[0][1]])
            if point2.distance(wall.linestring) > 2:
                continue
            pygame.draw.circle(self.screen, BLACK, (int(round(point2.xy[0][0])), int(round(point2.xy[1][0]))), 6)
            dist = point.distance(point2)
            if dist < closest_dist:
                closest_dist = dist
                closest_wall = wall
                if dist < self.radius:
                    wall.color = GOLD
                    wall.dist = dist
                    self.to_collide.append(wall)
        if not self.to_collide:
            self.to_collide.append(closest_wall)

    def get_positions_new(self, startx, starty, start_theta, slide = False, steps = 2):
        time_step = 1 / steps
        new_x, new_y, new_theta = startx, starty, start_theta
        start_point = Point(startx, starty)
        e_points = []
        colliding_walls = []
        dist2s = []
        for i in range(steps):
            if not slide:
                next_step = self.update_pos(new_x, new_y, new_theta, time_step)
            else:
                next_step = self.slide(new_x, new_y, new_theta, time_step)
            new_x, new_y, new_theta = next_step[0], next_step[1], next_step[2]
            e_points.append(Point([new_x, new_y]))

        line1 = LineString([start_point, e_points[0]])
        line2 = LineString([e_points[0], e_points[1]])
        line3 = LineString([start_point, e_points[1]])
        for wall in self.walls:
            wall.color = LIGHTBLUE
            wline = wall.linestring
            dist1 = start_point.distance(wline)
            dist2 = e_points[1].distance(wline)
            dist3 = e_points[0].distance(wline)
            C1 = wline.intersection(line1).coords
            C2 = wline.intersection(line2).coords
            C3 = wline.intersection(line3).coords
            if (C1 or C2 or C3 or dist2 < self.radius or dist3 < self.radius):
                return True
                # colliding_walls.append(wall)
                # self.collision_num += 1
                # dist2s.append(dist2)
        # if colliding_walls:
        #     print(len(colliding_walls))
        #     j = dist2s.index(min(dist2s))
        #     self.collided_wall = colliding_walls[j]
        #     self.collided_wall.color = GOLD
        #     return True
        return False

class Wall:
    def __init__(self, start_point, end_point, color, screen):
        self.start_point = start_point
        self.end_point = end_point
        self.color = color
        self.m = None
        self.angle = self.get_angle()
        self.dist = None
        self.hit = None
        self.linestring = LineString([(start_point[0], start_point[1]), (end_point[0], end_point[1])])
        self.screen = screen

    def draw(self):
        pygame.draw.line(self.screen, self.color, self.start_point, self.end_point, 3)

    def get_angle(self):
        if (self.end_point[0] != self.start_point[0]):
            self.m = (self.end_point[1] - self.start_point[1]) / (self.end_point[0] - self.start_point[0])
            return np.arctan(self.m)
        else:
            return PI / 2

class Sensor():
    def __init__(self, position):
        self.x = position[0]
        self.y = position[1]
        self.value = None

    def line_line_intersection(self, cx, cy, wall):
        x1, y1 = self.x, self.y
        x2, y2 = cx, cy
        x3, y3 = wall.start_point[0], wall.start_point[1]
        x4, y4 = wall.end_point[0], wall.end_point[1]

        self.Px = ( ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / \
               ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)) )
        self.Py = ( ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / \
               ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)) )
        return self.Px, self.Py

class Dust:
    def __init__(self, x, y, screen):
        self.x = x
        self.y = y
        self.collected = False
        self.screen = screen

    def draw(self):
        if self.collected:
            return
        if not self.collected:
            gfxdraw.aacircle(self.screen, int(round(self.x)), int(round(self.y)), 1, GREY)

    def clear(self):
        if not self.collected:
            self.collected = True
        else:
            return

class Environment:
    def __init__(self, screen, width,height,density = 1):
        self.all_dusts = []
        self.cleared_dust = 0
        self.screen = screen
        self.width = width
        self.height = height
        if density == 3:
            interval = 12
            for w in range(8, self.width - 8, interval):
                for h in range(8, self.height - 8, interval):
                    self.all_dusts.append(Dust(w, h, self.screen))
        if density == 2:
            interval = 18
            for w in range(8, self.width - 8, interval):
                for h in range(8, self.height - 8, interval):
                    self.all_dusts.append(Dust(w, h, self.screen))
        if density == 1:
            interval = 24
            for w in range(8, self.width - 8, interval):
                for h in range(8, self.height - 8, interval):
                    self.all_dusts.append(Dust(w, h,self.screen))

    def draw_dusts(self, robot):
        for d in self.all_dusts:
            if d.collected: continue
            dist = np.sqrt((d.x - robot.x) ** 2 + (d.y - robot.y) ** 2)
            if dist <= robot.radius:
                self.cleared_dust += 1
                d.clear()
            d.draw()


class Player(pygame.sprite.Sprite):
    def __init__(self, pos, size=(200, 200)):
        super(Player, self).__init__()
        self.original_image = load_image('img/tireForward_0.png')
        # pygame.draw.line(self.original_image, (255, 0, 255), (size[0] / 2, 0), (size[0] / 2, size[1]), 3)
        # pygame.draw.line(self.original_image, (0, 255, 255), (size[1], 0), (0, size[1]), 3)
        self.image = self.original_image
        self.rect = self.image.get_rect()
        self.rect.center = pos
        self.angle = 0

    def update(self, x1, y1, theta, radius, shift):

        x2 = (x1 + radius * cos(theta))
        y2 = (y1 + radius * sin(theta))
        x2 += shift
        y2 += shift

        # self.image = pygame.transform.rotate(self.original_image, self.angle)
        self.angle = (-np.arctan2((y2 - y1), (x2 - x1)) * 180 / np.pi) - 90
        # x, y = self.rect.center  # Save its current center.
        self.rect = self.image.get_rect()  # Replace old rect with new rect.
        self.rect.center = (x2, y2)  # Put the new rect's center at old center.


def load_image(name):
    image = pygame.image.load(name)
    return image

def blit_text(text, x, y, screen, text_color = SILVER, bkg_color = None, font_size = 16):
    font1 = pygame.font.SysFont("futura", font_size)
    text = font1.render(text, True, text_color, bkg_color)
    textRect = text.get_rect()
    textRect.center = (x, y)
    screen.blit(text, textRect)

#pygame.init()


#if __name__ == '__main__':

    #block = Robot(220, 290 , 2 , 3 , 20 , walls)
    #main()


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

# Output
# print("Final population (survivors):")
# for i in myPop.allIndividuals:
#     print("Individual - Genotype:",i.genotype[0],i.genotype[1], "Fitness:",i.fitness)
#
# nn = NeuralNetwork(12,[4],2, tanh, 0.1)
# input = np.array([[200,180,7,0,10,175,50,190,7,6,13,50]])
# input = scaler(input[0], 0, 200, -3, 3) # Scale values
# output = nn.forwardPropagation(input[0])
# nn.print()

# Save fitness and diversity of individuals
# maxFitness = []
# avgFitness = []
# maxDiversity = []
# avgDiversity = []
#
# # Compute the max fitness, average fitness, max diversity and average diversity through out all generations
# for t in range(n_epochs):
#     t_th_population = myPop.history[t]
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
# plt.plot(range(len(avgFitness)), avgFitness)
# plt.plot(range(len(maxFitness)), maxFitness)
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
#     if(t%50==0):
#         print("Epochs:", (t + 1))
#         if (benchmarkFunction == rastrigin):
#             contourf = plt.contourf(X, Y, Z)
#         elif (benchmarkFunction == rosenbrock):
#             contourf = plt.contourf(X, Y, Z, locator=ticker.LogLocator())
#
#         # Start drawing
#         plt.title("Number of total iterations: " + str(t + 1))
#
#         pos = myPop.history[t]
#         for j in pos:
#             plt.scatter(j[0], j[1], marker=markers[0])
#
#         plt.xlim(-10, 10)
#         plt.ylim(-10, 10)
#         plt.colorbar(contourf, orientation='horizontal')
#         plt.show()
