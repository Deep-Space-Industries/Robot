import pygame
import random
import numpy as np
import pygame.gfxdraw as gfxdraw

WHITE  = (255, 255, 255)
GREEN  = (20, 255, 140)
GREY   = (210, 210 ,210)
RED    = (255, 0, 0)
PURPLE = (255, 0, 255)
BLACK  = (0,0,0)

pygame.init()
background_colour = RED
(width, height) = (1000, 1000)
PI = np.pi
cos = np.cos
sin = np.sin
decrease_factor = 1
increase_factor = 1
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption('Assignment 2')
font = pygame.font.SysFont("Helvetica", 20)
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
        self.angle = 0

    def draw(self):
        # self.rect = pygame.rect.Rect((self.x, self.y, self.width, self.height))
        # pygame.draw.rect(screen, GREEN, self.rect)
        # self.rect = pygame.circ.Rect((self.x, self.y, self.width, self.height))
        # pygame.draw.circle(screen, GREEN, [self.x,self.y], self.radius)
        gfxdraw.aacircle(screen, self.x, self.y, self.radius, GREEN)
        gfxdraw.filled_circle(screen, self.x, self.y, self.radius, GREEN)

    def draw_direction(self):
        pygame.draw.line(screen, RED, (self.x, self.y), \
                         (self.x + self.radius * cos(self.angle),
                         self.y + self.radius * sin(self.angle)), 2)

    def speedup_left(self):
        self.left_velocity += increase_factor

    def slowdown_left(self):
        self.left_velocity -= decrease_factor

    def speedup_right(self):
        self.right_velocity += increase_factor

    def slowdown_right(self):
        self.right_velocity -= decrease_factor

    def speedup_both(self):
        self.left_velocity += increase_factor
        self.right_velocity += increase_factor

    def slowdown_both(self):
        self.right_velocity -= decrease_factor
        self.right_velocity -= decrease_factor

    def stop_both(self):
        self.right_velocity = self.left_velocity = 0

    def get_velocity(self):
        self.velocity = (self.left_velocity + self.right_velocity) / 2

    def distance_from_ICC(self):
        return self.radius * ( (self.right_velocity + self.left_velocity) / (self.right_velocity - self.left_velocity) )

    def omega(self):
        omega = (self.right_velocity - self.left_velocity) / (2 * self.radius)
        return omega

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

    def message_display(self):
        messages = [f"l velocity: {self.left_velocity}", \
                    f"r velocity: {self.right_velocity}"]
        message = "\n".join(messages)
        text = font.render(message, True ,(0, 128, 0))
        return text

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
              elif event.key == pygame.K_o:
                  block.speedup_left()
              elif event.key == pygame.K_l:
                  block.slowdown_left() # decrement of left wheel
              elif event.key == pygame.K_x:
                  block.stop_both() # zero both wheel speed
              elif event.key == pygame.K_t:
                  block.speedup_both() # increment both wheel speed
              elif event.key == pygame.K_t:
                  block.slowdown_both() # decrement both wheel speed
              elif event.key == pygame.K_ESCAPE:
                loopExit = False

      block.move()


      screen.fill(BLACK)
      wall.draw()
      block.draw()
      block.draw_direction()
      # text = block.message_display()
      # screen.blit(text, (320, 240))
      clock.tick(60)

      pygame.display.flip()

gameloop()

