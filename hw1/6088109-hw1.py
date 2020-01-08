"""A module for homework 1."""
import random
import copy
from functools import reduce


def shuffled(board):
    if len(board) == 0:
        return []
    randPosition = random.randrange(len(board))
    targetNumber = board[randPosition]
    newBoard = list(filter(lambda x: x != targetNumber, board))

    return shuffled(newBoard) + [targetNumber]


def chunk(board):
    return [board[i::3] for i in range(3)]


class EightPuzzleState:
    """A class for a state of an 8-puzzle game."""

    def __init__(self, board):
        """Create an 8-puzzle state."""
        self.action_space = {'u', 'd', 'l', 'r'}
        self.board = board
        for i, row in enumerate(self.board):
            for j, v in enumerate(row):
                if v == 0:
                    self.y = i
                    self.x = j

    def __repr__(self):
        """Return a string representation of a board."""
        output = []
        row_string = ''
        for row in self.board:
            row_string = ' | '.join([str(e) for e in row])
            output.append(row_string)

        return ('\n' + '-' * len(row_string) + '\n').join(output)

    def __str__(self):
        """Return a string representation of a board."""
        return self.__repr__()

    @staticmethod
    def initializeState():
        """
        Create an 8-puzzle state with a SHUFFLED tiles.

        Return
        ----------
        EightPuzzleState
            A state that contain an 8-puzzle board with a type of List[List[int]]:
            a nested list containing integers representing numbers on a board
            e.g., [[0, 1, 2], [3, 4, 5], [6, 7, 8]] where 0 is a blank tile.
        """
        initialList = list(range(0, 9))
        shuffledBoard = chunk(shuffled(initialList))
        return EightPuzzleState(shuffledBoard)

    def move(self, new_board, action, blank_position):
        x = blank_position[0]
        y = blank_position[1]
        move_position_x = x
        move_position_y = y

        if action == 'u':
            move_position_y -= 1
        elif action == 'd':
            move_position_y += 1
        elif action == 'l':
            move_position_x -= 1
        elif action == 'r':
            move_position_x += 1

        if move_position_x < 0 or move_position_x > 2 or move_position_y < 0 or move_position_y > 2:
            return None

        move_tile = new_board[move_position_y][move_position_x]
        new_board[move_position_y][move_position_x] = new_board[y][x]
        new_board[y][x] = move_tile

        return EightPuzzleState(new_board)

    def successor(self, action):
        """
        Move a blank tile in the current state, and return a new state.

        Parameters
        ----------
        action:  string
            Either 'u', 'd', 'l', or 'r'.

        Return
        ----------
        EightPuzzleState or None
            A resulting 8-puzzle state after performing `action`.
            If the action is not possible, this method will return None.

        Raises
        ----------
        ValueError
            if the `action` is not in the action space

        """
        if action not in self.action_space:
            raise ValueError(f'`action`: {action} is not valid.')

        # TODO: 2
        # YOU NEED TO COPY A BOARD BEFORE MODIFYING IT
        new_board = copy.deepcopy(self.board)
        x = self.x
        y = self.y
        blank_position = [x, y]

        return self.move(new_board, action, blank_position)

    def is_goal(self, goal_board=[[1, 2, 3], [4, 5, 6], [7, 8, 0]]):
        """
        Return True if the current state is a goal state.

        Parameters
        ----------
        goal_board (optional)
            The desired state of 8-puzzle.

        Return
        ----------
        Boolean
            True if the current state is a goal.

        """

        # TODO: 3
        flattern_board = sum(self.board, [])
        flattern_goal_board = sum(goal_board, [])

        return reduce(lambda x, y: x and y, map(lambda i, j: i ==
                                                j, flattern_board, flattern_goal_board), True)


class EightPuzzleNode:
    """A class for a node in a search tree of 8-puzzle state."""

    def __init__(
            self, state, parent=None, action=None, cost=1):
        """Create a node with a state."""
        self.state = state
        self.parent = parent
        self.action = action
        self.cost = cost
        if parent is not None:
            self.path_cost = parent.path_cost + self.cost
        else:
            self.path_cost = 0

    def trace(self):
        """
        Return a path from the root to this node.

        Return
        ----------
        List[EightPuzzleNode]
            A list of nodes stating from the root node to the current node.

        """
        trace = []
        cur_node = self

        while cur_node.parent is not None:
            trace.append(cur_node)
            cur_node = cur_node.parent
        return trace


def test_by_hand():
    """Run a CLI 8-puzzle game."""
    state = EightPuzzleState.initializeState()
    root_node = EightPuzzleNode(state, action='INIT')
    cur_node = root_node
    print(state)
    action = input('Please enter the next move (q to quit): ')
    while action != 'q':
        new_state = cur_node.state.successor(action)
        cur_node = EightPuzzleNode(new_state, cur_node, action)
        print(new_state)
        if new_state.is_goal():
            print('Congratuations!')
            break
        action = input('Please enter the next move (q to quit): ')

    print('Your actions are: ')
    for node in cur_node.trace():
        print(f'  - {node.action}')
    print(f'The total path cost is {cur_node.path_cost}')


if __name__ == '__main__':
    test_by_hand()
