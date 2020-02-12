import pygame
import random
import numpy as np

WHITE  = (255, 255, 255)
GREEN  = (20, 255, 140)
GREY   = (210, 210 ,210)
RED    = (255, 0, 0)
PURPLE = (255, 0, 255)
BLACK  = (0,0,0)

background_colour = RED
(width, height) = (1000, 1000)
PI = np.pi
cos = np.cos
sin = np.sin

screen = pygame.display.set_mode((width, height))
pygame.display.set_caption('Assignment 2')
# screen.fill(background_colour)
clock = pygame.time.Clock()
import jedi
jedi.preload_module("pygame", "numpy")

class Wall:
  def __init__(self, start_point  ,end_point):
    self.start_point=start_point
    self.end_point = end_point
  def draw(self):
    pygame.draw.line(screen,RED,self.start_point,self.end_point,1)

class Robot:
    def __init__(self, x, y, left_velocity, right_velocity,radius):
        self.x = x
        self.y = y
        self.radius=radius
        self.left_velocity = left_velocity
        self.right_velocity = right_velocity
        self.angle = 0

    def draw(self):
        # self.rect = pygame.rect.Rect((self.x, self.y, self.width, self.height))
        # pygame.draw.rect(screen, GREEN, self.rect)
        # self.rect = pygame.circ.Rect((self.x, self.y, self.width, self.height))
        pygame.draw.circle(screen, GREEN, [self.x,self.y], self.radius)

    def draw_direction(self):
        pygame.draw.line(screen, RED, (self.x, self.y), \
                         (self.x + self.radius * cos(self.angle),
                         self.y + self.radius * sin(self.angle)), 2)

    def get_velocity(self):
        self.velocity = (self.left_velocity + self.right_velocity) / 2
    def rotate(self, clockwise = True):
        if clockwise:
            self.angle += PI / 20
        else:
            self.angle -= PI / 20

    def move(self):
        self.get_velocity()
        v = self.velocity
        self.x = int(self.x + v * cos(self.angle))
        self.y = int(self.y + v * sin(self.angle))
        # rect = pygame.rect.Rect((self.x, self.y, self.width, self.height))
        # pygame.draw.rect(screen, RED, rect)
        self.draw()


block = Robot(100, 100, 5, 5, 30)
wall=Wall((100,200),(200,900))
def gameloop():
  t = 0
  loopExit = True
  while loopExit == True:

      for event in pygame.event.get():
          if event.type == pygame.QUIT:
              loopExit = False
          if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
              t = t+1
            elif event.key == pygame.K_w:
              block.rotate()
            elif event.key == pygame.K_s:
              block.rotate(False)
            elif event.key == pygame.K_ESCAPE:
              loopExit = False

      block.move()


      screen.fill(BLACK)
      wall.draw()
      block.draw()
      block.draw_direction()

      clock.tick(60)

      pygame.display.flip()

gameloop()

