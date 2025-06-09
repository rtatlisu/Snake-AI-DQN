import pygame

# returns a 3-element binary list indicating if there is a danger to the left, straight or right relative to
# the snake's orentation
def get_dangers(
        snake_segments: list[pygame.Rect], 
        direction: str,
        CELLSIZE: int, 
        SCREEN_SIZE: int
        ):
    head = snake_segments[0]
    dangers = []

    head_next_up = pygame.Rect(head.x,head.y, CELLSIZE, CELLSIZE)
    head_next_down = pygame.Rect(head.x,head.y, CELLSIZE, CELLSIZE)
    head_next_left = pygame.Rect(head.x,head.y, CELLSIZE, CELLSIZE)
    head_next_right = pygame.Rect(head.x,head.y, CELLSIZE, CELLSIZE)
    head_next_up.y -= CELLSIZE
    head_next_down.y += CELLSIZE
    head_next_left.x -= CELLSIZE
    head_next_right.x += CELLSIZE
    north_tail = False
    east_tail = False
    south_tail = False
    west_tail = False

    for tail in snake_segments[1:]:
        tail = pygame.Rect(tail.x,tail.y, CELLSIZE, CELLSIZE)
        if head_next_up.colliderect(tail):
            north_tail = True
        elif head_next_right.colliderect(tail):
            east_tail = True
        elif head_next_down.colliderect(tail):
            south_tail = True
        elif head_next_left.colliderect(tail):
            west_tail = True
    
    north_wall = True if head.y - CELLSIZE < 0 else False
    east_wall = True if head.x + CELLSIZE >= SCREEN_SIZE else False
    south_wall = True if head.y + CELLSIZE >= SCREEN_SIZE else False
    west_wall = True if head.x - CELLSIZE < 0 else False

    # order: straight, left, right
    if direction == 'up':
        dangers.append(north_wall or north_tail)
        dangers.append(west_wall or west_tail)
        dangers.append(east_wall or east_tail)
    elif direction == 'down':
        dangers.append(south_wall or south_tail)
        dangers.append(east_wall or east_tail)
        dangers.append(west_wall or west_tail)
    elif direction == 'left':
        dangers.append(west_wall or west_tail)
        dangers.append(south_wall or south_tail)
        dangers.append(north_wall or north_tail)
    else:
        dangers.append(east_wall or east_tail)
        dangers.append(north_wall or north_tail)
        dangers.append(south_wall or south_tail)
    

    return dangers


# returns a 4-element binary list giving info on the fruit's position, e.g. up right
def get_fruit_position(head_pos: tuple[int,int], fruit_pos: tuple[int,int]):
    # order: up, right, down, left
    return [
        head_pos.y - fruit_pos.y > 0,
        head_pos.x - fruit_pos.x < 0,
        head_pos.y - fruit_pos.y < 0,
        head_pos.x - fruit_pos.x > 0
    ]



# returns 4-element binary list with direction of snake
def get_direction(direction: str):
    # order: up, right, down, left
    return [
        direction == 'up',
        direction == 'right',
        direction == 'down',
        direction == 'left'
    ]

    




    


