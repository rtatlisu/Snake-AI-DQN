import pygame
import random
from helperClass import get_dangers, get_fruit_position, get_direction

pygame.init()
font = pygame.font.SysFont(None, 25)

SCREEN_SIZE = 640
WHITE = (128,128,128)
RED = (255,0,0)
BLACK = (0,0,0)
CELLSIZE = 20

# used to easily store positions of snake segments and fruit
class Vector2:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

class SnakeAI:
    def __init__(self, width, height, ticks):
        self.width = width
        self.height = height
        self.ticks = ticks
        self.tick_modifier = 1
        self.fruit = None
        self.lastPressedKey = 'd'
        self.moveDir = 'right'
        self.score = 0

        self.head = Vector2(CELLSIZE*4,SCREEN_SIZE//2)
        self.tail = Vector2(CELLSIZE*3, SCREEN_SIZE//2)
        self.snake_segments = [self.head, self.tail]

        self.fruit = self.spawn_fruit()

        self.screen = pygame.display.set_mode((SCREEN_SIZE,SCREEN_SIZE))
        self.clock = pygame.time.Clock()
    
    def gameLoop(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            if event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_w, pygame.K_a, pygame.K_s, pygame.K_d]:
                    self.pressed = pygame.key.name(event.key)
                    # makes sure that snake can't do a 180
                    if (self.moveDir == 'up' and self.pressed != 's' or 
                        self.moveDir == 'down' and self.pressed != 'w' or
                        self.moveDir == 'left' and self.pressed != 'd' or 
                        self.moveDir == 'right' and self.pressed != 'a'):
                        self.lastPressedKey = self.pressed
                # speeds up game
                elif event.key == pygame.K_UP:
                    self.tick_modifier += 0.2
                # slows down game
                elif event.key == pygame.K_DOWN:
                    if self.ticks * (self.tick_modifier - 0.2) > 1:
                        self.tick_modifier -= 0.2


        self.move(self.lastPressedKey)
        # moving the snake consists of adding a 'cube' in front of the head and removing the tail
        self.snake_segments.insert(0, self.head)

        # if death by either, reset game and exit current loop iteration
        if self.is_tail_collision() or self.is_wall_collision():
            self.death()
            return
        # if fruit eaten, dont remove last tail part 
        if self.is_fruit_eaten():
            self.fruit = self.spawn_fruit()
        else:
            self.snake_segments.pop()

        
        self.update_ui()
        # tick_modifier used to increase game speed while in-game
        self.clock.tick(self.ticks * self.tick_modifier)




    def update_ui(self):
        self.screen.fill(BLACK)

        # draw snake
        for el in self.snake_segments:
            rect = pygame.Rect(el.x, el.y, CELLSIZE-2,CELLSIZE-2)
            pygame.draw.rect(self.screen, WHITE, rect, 10)

        # draw fruit
        rect = pygame.Rect(self.fruit.x, self.fruit.y, CELLSIZE-2,CELLSIZE-2)
        pygame.draw.rect(self.screen, RED, rect, 10)

        # other
        text_surface = font.render(f'Score: {self.score}', True, (255, 255, 255))
        self.screen.blit(text_surface, (10, 10))

        pygame.display.update()

    # for visualization purposes regarding cells
    def drawGrid(self):
        for i in range(0, SCREEN_SIZE, CELLSIZE):
            for j in range(0, SCREEN_SIZE, CELLSIZE):
                rect = pygame.Rect(i, j, CELLSIZE, CELLSIZE)
                pygame.draw.rect(self.screen, WHITE, rect, 1)
                pygame.draw.circle(self.screen,RED,[i+CELLSIZE/2,j+CELLSIZE/2],1)


    def move(self, direction):
        xPos = self.head.x
        yPos = self.head.y

        if direction == 'w':
            yPos -= CELLSIZE
            self.moveDir = 'up'
        elif direction == 'a':
            xPos -= CELLSIZE
            self.moveDir = 'left'
        elif direction == 's':
            yPos += CELLSIZE
            self.moveDir = 'down'
        elif direction == 'd':
            xPos += CELLSIZE
            self.moveDir = 'right'
        
        # create new instance for self.head to insert at the top of list and after pop the last element in list
        self.head = Vector2(xPos,yPos)

  
    # spawns the fruit and makes sure that it's not spawning on top of the snake
    def spawn_fruit(self) -> Vector2:
        # random numbers between 0 - 31 and then multiplying by CELLSIZE to assign fruit to a cell
        # -2 comes from design choice, since the snake shouldn't be a solid moving mass, but visually separable and
        # therefore the fruit should have the same size
        #while True:
            while True:
                xPos = random.randint(0,31)
                yPos = random.randint(0,31)
                xPos *= CELLSIZE 
                yPos *= CELLSIZE
                valid = False
                for el in self.snake_segments:
                    if el.x == xPos and el.y == yPos:
                        valid = False
                        continue
                    else:
                        valid = True
                if valid:
                    break

            return Vector2(xPos,yPos)

    # check if snake runs into wall
    def is_wall_collision(self):
        if (self.head.x + CELLSIZE > SCREEN_SIZE or self.head.x < 0 or 
            self.head.y + CELLSIZE > SCREEN_SIZE  or self.head.y < 0):
            return True
        return False

    # check if fruit was eaten
    def is_fruit_eaten(self):
        if self.head.x == self.fruit.x and self.head.y == self.fruit.y:
            self.score += 1
            return True
        return False
      


    def is_tail_collision(self):
        for el in self.snake_segments[1:]:
            if self.head.x == el.x and self.head.y == el.y:
                return True
        return False



    # resets runtime variables to initial state for restart
    def death(self):
        self.snake_segments = []
        self.fruit = None
        self.lastPressedKey = 'd'
        self.moveDir = 'right'
        self.score = 0

        self.head = Vector2(CELLSIZE*4,SCREEN_SIZE//2)
        self.tail = Vector2(CELLSIZE*3, SCREEN_SIZE//2)
        self.snake_segments = [self.head, self.tail]

        self.fruit = self.spawn_fruit()



        


if __name__ == '__main__':
    instance = SnakeAI(640,640,5)

    while True:
        instance.gameLoop()
