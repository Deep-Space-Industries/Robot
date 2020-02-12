import pygame
import random
WHITE  = (255, 255, 255)
GREEN  = (20, 255, 140)
GREY   = (210, 210 ,210)
RED    = (255, 0, 0)
PURPLE = (255, 0, 255)
BLACK  = (0,0,0)

background_colour = RED
(width, height) = (1000, 1000)

screen = pygame.display.set_mode((width, height))
pygame.display.set_caption('Assignment 2')
# screen.fill(background_colour)
clock = pygame.time.Clock()

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

    def draw(self):
        # self.rect = pygame.rect.Rect((self.x, self.y, self.width, self.height))
        # pygame.draw.rect(screen, GREEN, self.rect)
        # self.rect = pygame.circ.Rect((self.x, self.y, self.width, self.height))
        pygame.draw.circle(screen, GREEN, (self.x,self.y),self.radius)

        # self.x = self.rect.left
        # self.y = self.rect.top

    def move(self, t):
        self.x = self.x + t
        self.y = self.y + t
        # rect = pygame.rect.Rect((self.x, self.y, self.width, self.height))
        # pygame.draw.rect(screen, RED, rect)
        self.draw()


block = Robot(100, 100, 50, 50, 30)
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


      block.move(t)


      screen.fill(BLACK)
      wall.draw()
      block.draw()

      clock.tick(60)

      pygame.display.flip()

gameloop()

