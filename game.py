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
WIN_REWARD = 1
NO_REWARD = 0
LOSE_REWARD = -1

# used to easily store positions of snake segments and fruit
class Vector2:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

class SnakeAI:
    rounds_played = 0
    max_score = 0
    def __init__(self, width, height, ticks, agent):
        self.agent = agent
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
        self.moves_left = self.set_lifespan()

        self.fruit = self.spawn_fruit()

        self.screen = pygame.display.set_mode((SCREEN_SIZE,SCREEN_SIZE))
        self.clock = pygame.time.Clock()
    
    def gameLoop(self, direction):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                if self.agent.trainer.ENABLE_SAVING:
                    self.agent.trainer.save_models()
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

        self.move(direction)
        # moving the snake consists of adding a 'cube' in front of the head and removing the tail
        self.snake_segments.insert(0, self.head)

        # if death by either, reset game and exit current loop iteration
        if self.is_tail_collision() or self.is_wall_collision():
            score = self.score
            self.death()
            return LOSE_REWARD, True, score
        # if fruit eaten, dont remove last tail part 
        fruit_eaten = False
        if self.is_fruit_eaten():
            self.fruit = self.spawn_fruit()
            self.moves_left = self.set_lifespan()
            fruit_eaten = True
        else:
            self.snake_segments.pop()

        
        self.update_ui()
        self.moves_left -= 1
        # tick_modifier used to increase game speed while in-game
        self.clock.tick(self.ticks * self.tick_modifier)

        if fruit_eaten:
            return WIN_REWARD, False, self.score
        if self.moves_left == 0:
            score = self.score
            self.death()
            return LOSE_REWARD, True, score
        
        return NO_REWARD, False, self.score
        






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
        # score
        text_surface = font.render(f'Score: {self.score}', True, (255, 255, 255))
        self.screen.blit(text_surface, (10, 10))
        # moves left
        text_surface = font.render(f'Moves left: {self.moves_left}', True, (255, 255, 255))
        self.screen.blit(text_surface, (10, 40))

        pygame.display.update()

    # for visualization purposes regarding cells
    def drawGrid(self):
        for i in range(0, SCREEN_SIZE, CELLSIZE):
            for j in range(0, SCREEN_SIZE, CELLSIZE):
                rect = pygame.Rect(i, j, CELLSIZE, CELLSIZE)
                pygame.draw.rect(self.screen, WHITE, rect, 1)
                pygame.draw.circle(self.screen,RED,[i+CELLSIZE/2,j+CELLSIZE/2],1)


    def move(self, direction):
        # direction input comes as 3-element binary array with order straight, left, right
        xPos = self.head.x
        yPos = self.head.y
        directions = ['up','right','down','left']
        # for left, multiply 1 by -1 to shift index to the left, else to the right
        # since its always either left or right, we can combine this in one call
        self.moveDir = directions[(directions.index(self.moveDir) + (direction[1] * -1) + direction[2]) % 4]

        if self.moveDir == 'up':
            yPos -= CELLSIZE
        elif self.moveDir == 'left':
            xPos -= CELLSIZE
        elif self.moveDir == 'down':
            yPos += CELLSIZE
        elif self.moveDir == 'right':
            xPos += CELLSIZE
        
        # create new instance for self.head to insert at the top of list and after pop the last element in list
        self.head = Vector2(xPos,yPos)

  
    # spawns the fruit and makes sure that it's not spawning on top of the snake
    def spawn_fruit(self) -> Vector2:
        # random numbers between 0 - 31 and then multiplying by CELLSIZE to assign fruit to a cell
            while True:
                xPos = random.randint(0,31)
                yPos = random.randint(0,31)
                xPos *= CELLSIZE 
                yPos *= CELLSIZE
                valid = True
                for el in self.snake_segments:
                    if el.x == xPos and el.y == yPos:
                        valid = False
                        break
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

    def set_lifespan(self):
        return min(100 + 2 * len(self.snake_segments), 200)



    # resets runtime variables to initial state for restart
    def death(self):
        SnakeAI.rounds_played += 1
        if self.score > SnakeAI.max_score:
            SnakeAI.max_score = self.score
        self.snake_segments = []
        self.fruit = None
        self.lastPressedKey = 'd'
        self.moveDir = 'right'
        self.score = 0

        self.head = Vector2(CELLSIZE*4,SCREEN_SIZE//2)
        self.tail = Vector2(CELLSIZE*3, SCREEN_SIZE//2)
        self.snake_segments = [self.head, self.tail]
        self.moves_left = self.set_lifespan()

        self.fruit = self.spawn_fruit()
