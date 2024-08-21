import sys

from actions import Action


class Sequence:
    """ Contains information about a sequence. """
    def __init__(self, data) -> None:
        sequence_string_list = data['sequence']
        self.sequence = list()
        for s in sequence_string_list:
            self.sequence.append(self.__get_action(s))

    def pretty_print(self) -> None:
        """ Pretty prints the sequence to the console. """
        for s in self.sequence:
            print(s, end=' --> ')
        print()

    def get_next(self):
        """ Returns the next action to perform without removing. """
        return self.sequence[0]

    def pop_action(self):
        """ Returns the next action to perform in this sequence. """
        return self.sequence.pop(0)

    def __get_action(self, a_string) -> Action:
        """ Maps the action string to the action enum value. """
        if a_string == 'move': return Action.MOVE
        elif a_string == 'turnLeft': return Action.TURN_LEFT
        elif a_string == 'turnRight': return Action.TURN_RIGHT
        elif a_string == 'pickMarker': return Action.PICK_MARKER
        elif a_string == 'putMarker': return Action.PUT_MARKER
        elif a_string == 'finish': return Action.FINISH
        else: sys.exit("Error: Action '" + a_string + "' is not a valid action.")

    def __len__(self):
        """ Returns the length of the sequence. """
        return len(self.sequence)

   