import pygame
import random
from helperClass import getDangers, getFruitPosition, getDirection


SCREEN_SIZE = 640
WHITE = (128,128,128)
RED = (255,0,0)
BLACK = (0,0,0)
CELLSIZE = 20
ticks = 5
tick_modifier = 1
bodyPositions = []
# initially moves to the right
lastPressedKey = 'd'
moveDir = 'right'
fruitPos = None
pause = False


def main():
    global screen, lastPressedKey, ticks, tick_modifier, pause
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_SIZE,SCREEN_SIZE))
    clock = pygame.time.Clock()
    running = True


    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_w, pygame.K_a, pygame.K_s, pygame.K_d]:
                    pressed = pygame.key.name(event.key)
                    # makes sure that snake can't do a 180
                    if (moveDir == 'up' and pressed != 's' or 
                        moveDir == 'down' and pressed != 'w' or
                        moveDir == 'left' and pressed != 'd' or 
                        moveDir == 'right' and pressed != 'a'):
                        lastPressedKey = pressed
                # speeds up game
                elif event.key == pygame.K_UP:
                    tick_modifier += 0.2
                # slows down game
                elif event.key == pygame.K_DOWN:
                    if ticks * (tick_modifier - 0.2) > 1:
                        tick_modifier -= 0.2
                # pauses game
                elif event.key == pygame.K_SPACE:
                    pause = not pause
                    


        #drawGrid()
        if(len(bodyPositions) > 0 and not pause):
            move(lastPressedKey)
            if fruitCollision():
                pass
            elif wallCollision():
                continue
            elif tailCollision():
                continue
        
        screen.fill(BLACK)
        drawSnake()
        drawFruit()
        if(len(bodyPositions) > 0 ):
            #print(getDangers(bodyPositions, moveDir, CELLSIZE, SCREEN_SIZE))
            #print(getFruitPosition((bodyPositions[0].x,bodyPositions[0].y), (fruitPos.x,fruitPos.y)))
            print(getDirection(moveDir))

        pygame.display.update()
        clock.tick(ticks * tick_modifier)



    pygame.quit()


# for visualization purposes regarding cells
def drawGrid():
    for i in range(0, SCREEN_SIZE, CELLSIZE):
        for j in range(0, SCREEN_SIZE, CELLSIZE):
            rect = pygame.Rect(i, j, CELLSIZE, CELLSIZE)
            pygame.draw.rect(screen, WHITE, rect, 1)
            pygame.draw.circle(screen,RED,[i+CELLSIZE/2,j+CELLSIZE/2],1)


def drawSnake():
    global bodyPositions
    # initial drawing/spawning of the snake
    if len(bodyPositions) == 0:
        head = pygame.Rect(CELLSIZE*4,SCREEN_SIZE/2, CELLSIZE-2, CELLSIZE-2)
        pygame.draw.rect(screen, WHITE, head, 10)
        tail = pygame.Rect(head.left-CELLSIZE,head.top, CELLSIZE-2, CELLSIZE-2)
        pygame.draw.rect(screen, WHITE, tail, 10)
        bodyPositions = [head,tail]
    else:
        # draw tails
        for i in range(1, len(bodyPositions)):
            rect = pygame.Rect(bodyPositions[i].left,bodyPositions[i].top, CELLSIZE-2,CELLSIZE-2)
            pygame.draw.rect(screen, WHITE, rect, 10)
        # draw head
        rect = pygame.Rect(bodyPositions[0].left,bodyPositions[0].top, CELLSIZE-2,CELLSIZE-2)
        pygame.draw.rect(screen, WHITE, rect, 10)



def move(key):
    global moveDir
    # starting from the last part of the snake's tail, assign the predecessor's position
    # done for all parts but the head
    for i in range(-1, -len(bodyPositions),-1):
        bodyPositions[i].x = bodyPositions[i-1].x
        bodyPositions[i].y = bodyPositions[i-1].y
        

    # separately updates the head's position from the rest of its body
    if key == 'w':
        bodyPositions[0].y -= CELLSIZE
        moveDir = 'up'
    elif key == 'a':
        bodyPositions[0].x -= CELLSIZE
        moveDir = 'left'
    elif key == 's':
        bodyPositions[0].y += CELLSIZE
        moveDir = 'down'
    elif key == 'd':
        bodyPositions[0].x += CELLSIZE
        moveDir = 'right'


# spawns the fruit and makes sure that it's not spawning on top of the snake
def drawFruit():
    global fruitPos
    # random numbers between 0 - 31 and then multiplying by CELLSIZE to assign fruit to a cell
    # -2 comes from design choice, since the snake shouldn't be a solid moving mass, but visually separable and
    # therefore the fruit should have the same size
    if fruitPos == None:     
        xPos = random.randint(0,31)
        yPos = random.randint(0,31)
        xPos *= CELLSIZE 
        yPos *= CELLSIZE
    else:
        xPos = fruitPos.x
        yPos = fruitPos.y

    rect = pygame.Rect(xPos,yPos, CELLSIZE-2,CELLSIZE-2)
    pygame.draw.rect(screen, RED, rect, 10)
    fruitPos = rect


def wallCollision():
    # check if snake runs into wall
    head = bodyPositions[0]
    if head.x + CELLSIZE > SCREEN_SIZE or head.x < 0 or head.y + CELLSIZE > SCREEN_SIZE  or head.y < 0:
        death()
        return True
    return False


def fruitCollision():
    global fruitPos
    # check if fruit was eaten
    try:
        if bodyPositions[0].colliderect(fruitPos):
            # add tail to end of  snake
            rect = pygame.Rect(bodyPositions[-1].left,bodyPositions[-1].top, CELLSIZE-2,CELLSIZE-2)
            pygame.draw.rect(screen, WHITE, rect, 10)
            bodyPositions.append(rect)
            fruitPos = None
            return True
        return False

    except Exception as e:
        print('couldn\'t determine if fruit was eaten')
        print(e)


def tailCollision():
    head = bodyPositions[0]
    for tail in bodyPositions[1:]:
        if head.colliderect(tail):
            death()
            return True
    return False


# resets runtime variables to initial state for restart
def death():
    global bodyPositions, fruitPos, lastPressedKey, moveDir
    bodyPositions = []
    fruitPos = None
    lastPressedKey = 'd'
    moveDir = 'right'



    


    

if __name__ == '__main__':
    main()