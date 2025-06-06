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

    head_next_up = pygame.Rect(head)
    head_next_down = pygame.Rect(head)
    head_next_left = pygame.Rect(head)
    head_next_right = pygame.Rect(head)
    head_next_up.y -= CELLSIZE
    head_next_down.y += CELLSIZE
    head_next_left.x -= CELLSIZE
    head_next_right.x += CELLSIZE
    north_tail = False
    east_tail = False
    south_tail = False
    west_tail = False

    for tail in snake_segments[1:]:
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
        dangers.append(1 if north_wall or north_tail else 0)
        dangers.append(1 if west_wall or west_tail else 0)
        dangers.append(1 if east_wall or east_tail else 0)
    elif direction == 'down':
        dangers.append(1 if south_wall or south_tail else 0)
        dangers.append(1 if east_wall or east_tail else 0)
        dangers.append(1 if west_wall or west_tail else 0)
    elif direction == 'left':
        dangers.append(1 if west_wall or west_tail else 0)
        dangers.append(1 if south_wall or south_tail else 0)
        dangers.append(1 if north_wall or north_tail else 0)
    else:
        dangers.append(1 if east_wall or east_tail else 0)
        dangers.append(1 if north_wall or north_tail else 0)
        dangers.append(1 if south_wall or south_tail else 0)
    

    return dangers


# returns a 4-element binary list giving info on the fruit's position, e.g. up right
def get_fruit_position(head_pos: tuple[int,int], fruit_pos: tuple[int,int]):
    head_x, head_y = head_pos
    fruit_x, fruit_y = fruit_pos

    # order: up, left, right, down
    fruit_direction = []
    fruit_direction.append(1 if head_y - fruit_y > 0 else 0)
    fruit_direction.append(1 if head_x - fruit_x > 0 else 0)
    fruit_direction.append(1 if head_x - fruit_x < 0 else 0)
    fruit_direction.append(1 if head_y - fruit_y < 0 else 0)

    return fruit_direction


# returns 4-element binary list with direction of snake
def get_direction(direction: str):
    # order: up, left, right, dwon
    return [
        1 if direction == 'up' else 0,
        1 if direction == 'left' else 0,
        1 if direction == 'right' else 0,
        1 if direction == 'down' else 0
        ]

    




    


