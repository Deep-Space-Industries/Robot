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
FPS = 120

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
  def __init__(self, start_point, end_point, color):
    self.start_point = start_point
    self.end_point = end_point
    self.color = color

  def draw(self):
      pygame.draw.line(screen, self.color, self.start_point, self.end_point, 3)

class Robot:
    def __init__(self, x, y, left_velocity, right_velocity, radius):
        self.x = x
        self.y = y
        self.radius = radius
        self.history = []
        self.left_velocity = left_velocity
        self.right_velocity = right_velocity
        self.theta = 0
        self.thetarecord = []
        # self.omega=((self.right_velocity-self.left_velocity))/(2*self.radius)
        # self.icc_radius = self.radius * (self.left_velocity + self.right_velocity) / (self.right_velocity-self.left_velocity + 0.00001)
        # self.icc_centre_x=self.x-self.icc_radius*sin(self.theta)
        # self.icc_centre_y=self.y+self.icc_radius*cos(self.theta)
        self.omega = None
        self.icc_radius = None
        self.icc_centre_x = None
        self.icc_centre_y = None
        self.t = 0
        self.update_icc()
        self.decrease_factor = 1
        self.increase_factor = 1

    def draw(self):
        # self.rect = pygame.rect.Rect((self.x, self.y, self.width, self.height))
        # pygame.draw.rect(screen, GREEN, self.rect)
        # self.rect = pygame.circ.Rect((self.x, self.y, self.width, self.height))
        # pygame.draw.circle(screen, GREEN, [self.x,self.y], self.radius)
        gfxdraw.aacircle(screen, self.x, self.y, self.radius, GREEN)
        gfxdraw.filled_circle(screen, self.x, self.y, self.radius, GREEN)
        self.history.append([int(self.x), int(self.y)])
        self.thetarecord.append(self.theta)
        print(f"x, y: {[self.x, self.y]}")
        if len(self.history) > 10:
            # color1 = 10
            # for i, h in enumerate(self.history[:-2]):
            #     color1 = int((color1 + 3) & 255)
            #     pygame.draw.aaline(screen, (color1, color1, 100), self.history[i], self.history[i + 1])
            pygame.draw.aalines(screen, PURPLE, False, self.history)


    def draw_icc(self):
        if (self.left_velocity == self.right_velocity):
            return
        pygame.draw.circle(screen, PURPLE, [int(self.icc_centre_x),int(self.icc_centre_y)],2)
        print(f"icc x y: {[int(self.icc_centre_x), int(self.icc_centre_y)]}")


    def draw_direction(self):
        if (self.right_velocity  < 0):
            pygame.draw.line(screen, RED, (self.x, self.y), \
                             (self.x + self.radius * -cos(self.theta),
                              self.y + self.radius * -sin(self.theta)), 3)
        else:
            pygame.draw.line(screen, RED, (self.x, self.y), \
                         (self.x + self.radius * cos(self.theta),
                         self.y + self.radius * sin(self.theta)), 3)

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
        self.omega = (self.right_velocity - self.left_velocity) / (2 * self.radius)
        self.icc_radius = self.radius * ((self.left_velocity) + (self.right_velocity)) / (
            (self.right_velocity - self.left_velocity) + 0.0001)
        self.icc_centre_x = self.x - self.icc_radius * sin(self.theta)
        self.icc_centre_y = self.y + self.icc_radius * cos(self.theta)

    def check_boundary(self):
        max_pos_x = width - 5
        max_pos_y = height - 5
        min_pos_x = min_pos_y = 5
        if (self.x + self.radius >= max_pos_x):
            self.x = max_pos_x - self.radius
        if (self.x - self.radius <= min_pos_x):
            self.x = min_pos_x + self.radius
        if (self.y + self.radius >= max_pos_y):
            self.y = max_pos_y - self.radius
        if (self.y - self.radius <= min_pos_y):
            self.y = min_pos_y + self.radius
        # self.theta += self.omega

    def move(self):
        if (self.left_velocity == self.right_velocity):
            print("same speed")
            self.x += int(round(self.left_velocity * cos(self.theta)))
            self.y += int(round(self.right_velocity * sin(self.theta)))
            self.theta += self.omega
        else:
            p= np.dot(np.array([[cos(self.omega), -sin(self.omega), 0],
                                [sin(self.omega), cos(self.omega), 0],
                                [0, 0, 1]]), \
                                np.array(
                                    [
                                    self.icc_radius * sin(self.theta),# self.x - self.icc_centre_x,
                                    -self.icc_radius * cos(self.theta),# self.y - self.icc_centre_y,
                                    self.theta]))
            self.x = int(np.round(p[0] + self.icc_centre_x))
            self.y = int(np.round(p[1] + self.icc_centre_y))
            print(f"ps : {p}")
            self.theta = p[2] + self.omega
            self.t += 1
        self.check_boundary()
        # rect = pygame.rect.Rect((self.x, self.y, self.width, self.height))
        # pygame.draw.rect(screen, RED, rect)
        print(f"time: {self.t}")
        print(f"theta: {self.theta}")
        print(f"omega: {self.omega}")
        print(f"icc radius: {self.icc_radius}")
        self.draw()

    def message_display(self):
        messages = [f"l velocity: {self.left_velocity}", \
                    f"r velocity: {self.right_velocity}"]
        message = "\n".join(messages)
        text = font.render(message, True ,(0, 128, 0))
        return text

block = Robot(500, 520, 1, 2, 30)

walls = []
wall=Wall((100,200),(200,900), WHITE)
east_border = Wall((width - 5, 0), (width - 5, height - 5), WHITE)
west_border = Wall((5, 5), (5, height - 5), WHITE)
south_border = Wall((5, height - 5), (width - 5, height - 5), WHITE)
north_border = Wall((5, 5), (width - 5, 5), WHITE)
walls.append(wall)
walls.append(east_border)
walls.append(west_border)
walls.append(south_border)
walls.append(north_border)

def gameloop():
  t = 0
  loopExit = True
  # clock.tick(200)
  while loopExit == True:
      for event in pygame.event.get():
          if event.type == pygame.QUIT:
              loopExit = False
          if event.type == pygame.KEYDOWN:
              if event.key == pygame.K_UP:
                  t = t+1
              elif event.key == pygame.K_w:
                  block.speedup_left()
                  block.update_icc()
              elif event.key == pygame.K_s:
                  block.slowdown_left()
                  block.update_icc()
              elif event.key == pygame.K_o:
                  block.speedup_right()
                  block.update_icc()
              elif event.key == pygame.K_l:
                  block.slowdown_right() # decrement of left wheel
                  block.update_icc()
              elif event.key == pygame.K_x:
                  block.stop_both() # zero both wheel speed
                  block.update_icc()
              elif event.key == pygame.K_t:
                  block.speedup_both() # increment both wheel speed
                  block.update_icc()
              elif event.key == pygame.K_g:
                  block.slowdown_both() # decrement both wheel speed
                  block.update_icc()
              elif event.key == pygame.K_ESCAPE:
                  loopExit = False

      screen.fill(BLACK)
      for w in walls:
          w.draw()
      # block.draw()
      block.move()
      block.draw_direction()
      block.draw_icc()
      print(f"left_V: {block.left_velocity}, right_V: {block.right_velocity}")
      print(f"fps: {pygame.time.get_ticks()}")
      print(len(block.history))
      print(len(block.thetarecord))
      # block.update_icc()
      # text = block.message_display()
      # screen.blit(text, (320, 240))

      clock.tick(180)
      pygame.display.update()


gameloop()