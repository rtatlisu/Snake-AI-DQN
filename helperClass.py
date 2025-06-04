import pygame

# returns a 3-element binary list indicating if there is a danger to the left, straight or right relative to
# the snake's orentation
def getDangers(
        bodyPositions: list[pygame.Rect], 
        direction: str,
        CELLSIZE: int, 
        SCREEN_SIZE: int
        ):
    head = bodyPositions[0]
    dangers = []

    head_next_up = pygame.Rect(head)
    head_next_down = pygame.Rect(head)
    head_next_left = pygame.Rect(head)
    head_next_right = pygame.Rect(head)
    head_next_up.y -= CELLSIZE
    head_next_down.y += CELLSIZE
    head_next_left.x -= CELLSIZE
    head_next_right.x += CELLSIZE
    northTail = False
    eastTail = False
    southTail = False
    westTail = False

    for tail in bodyPositions[1:]:
        if head_next_up.colliderect(tail):
            northTail = True
        elif head_next_right.colliderect(tail):
            eastTail = True
        elif head_next_down.colliderect(tail):
            southTail = True
        elif head_next_left.colliderect(tail):
            westTail = True
    
    northWall = True if head.y - CELLSIZE < 0 else False
    eastWall = True if head.x + CELLSIZE >= SCREEN_SIZE else False
    southWall = True if head.y + CELLSIZE >= SCREEN_SIZE else False
    westWall = True if head.x - CELLSIZE < 0 else False

    # order: straight, left, right
    if direction == 'up':
        dangers.append(1 if northWall or northTail else 0)
        dangers.append(1 if westWall or westTail else 0)
        dangers.append(1 if eastWall or eastTail else 0)
    elif direction == 'down':
        dangers.append(1 if southWall or southTail else 0)
        dangers.append(1 if eastWall or eastTail else 0)
        dangers.append(1 if westWall or westTail else 0)
    elif direction == 'left':
        dangers.append(1 if westWall or westTail else 0)
        dangers.append(1 if southWall or southTail else 0)
        dangers.append(1 if northWall or northTail else 0)
    else:
        dangers.append(1 if eastWall or eastTail else 0)
        dangers.append(1 if northWall or northTail else 0)
        dangers.append(1 if southWall or southTail else 0)
    

    return dangers


# returns a 4-element binary list giving info on the fruit's position, e.g. up right
def getFruitPosition(headPos: tuple[int,int], fruitPos: tuple[int,int]):
    head_x, head_y = headPos
    fruit_x, fruit_y = fruitPos

    # order: up, left, right, down
    fruitDirection = []
    fruitDirection.append(1 if head_y - fruit_y > 0 else 0)
    fruitDirection.append(1 if head_x - fruit_x > 0 else 0)
    fruitDirection.append(1 if head_x - fruit_x < 0 else 0)
    fruitDirection.append(1 if head_y - fruit_y < 0 else 0)

    return fruitDirection


# returns 4-element binary list with direction of snake
def getDirection(direction: str):
    # order: up, left, right, dwon
    return [
        1 if direction == 'up' else 0,
        1 if direction == 'left' else 0,
        1 if direction == 'right' else 0,
        1 if direction == 'down' else 0
        ]

    




    


