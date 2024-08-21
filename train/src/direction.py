import sys


def direction_get_value(dir) -> int:
    """ Returns the integer value of the action given as a string. """
    if dir == 'north':
        return 1
    elif dir == 'east':
        return 2
    elif dir == 'south':
        return 3
    elif dir == 'west':
        return 4
    else:
        sys.exit("Error: Direction '" + str(dir) + "' does not exist!")

def direction_turn_left(dir) -> int:
    """ Returns the new direction when turning left. """
    if dir == 0:
        return 0
    elif dir not in {1,2,3,4}:
        sys.exit("Error: Direction '" + str(dir) +"' does not exist!")
    return ((dir - 2) % 4) + 1   

def direction_turn_right(dir) -> int:
    """ Returns the new direction when turning right. """
    if dir == 0:
        return 0
    elif dir not in {1,2,3,4}:
        sys.exit("Error: Direction '" + str(dir) +"' does not exist!")
    return (dir % 4) + 1

def direction_move(dir) -> list:
    """ Returns the movement offset (y,x) when moving in the given direction. """
    if dir == 1: return [-1,0]
    elif dir == 2: return [0, 1]
    elif dir == 3: return [1, 0]
    elif dir == 4: return [0, -1]
    else: sys.exit("Error: Direction '" + str(dir) + "' does not exist!")
