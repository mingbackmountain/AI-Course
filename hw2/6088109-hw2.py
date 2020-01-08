"""A module for homework 2. Version 3."""

import abc
import heapq
import itertools
import copy
from collections import defaultdict

from hw1 import EightPuzzleState, EightPuzzleNode


def eightPuzzleH1(state, goal_state):
    """
    Return the number of misplaced tiles including blank.

    Parameters
    ----------
    state : EightPuzzleState
        an 8-puzzle state containing a board (List[List[int]])
    goal_state : EightPuzzleState
        a desired 8-puzzle state.

    Returns
    ----------
    int

    """
    # TODO 1:
    misplacedTiles = 0
    goalBoard = goal_state.board
    for i in range(len(goalBoard)):
        for j in range(len(goalBoard[0])):
            if goalBoard[i][j] != state.board[i][j]:
                misplacedTiles += 1
    return misplacedTiles


def eightPuzzleH2(state: EightPuzzleState, goal_state: EightPuzzleState):
    """
    Return the total Manhattan distance from goal position of all tiles.

    Parameters
    ----------
    state : EightPuzzleState
        an 8-puzzle state containing a board (List[List[int]])
    goal_state : EightPuzzleState
        a desired 8-puzzle state.

    Returns
    ----------
    int

    """
    # TODO 2:
    boardFlattern = sum(state.board, [])
    manhattanDis = 0
    for i, value in enumerate(boardFlattern):
        previous_row, previous_column = int(i / 3), i % 3
        goal_row, goal_column = int(value / 3), value % 3
        manhattanDis += (abs(previous_row - goal_row) +
                         abs(previous_column - goal_column))
    return manhattanDis


class Frontier(abc.ABC):
    """An abstract class of a frontier."""

    def __init__(self):
        """Create a frontier."""
        raise NotImplementedError()

    @abc.abstractmethod
    def is_empty(self):
        """Return True if empty."""
        raise NotImplementedError()

    @abc.abstractmethod
    def add(self, node):
        """
        Add a node into the frontier.

        Parameters
        ----------
        node : EightPuzzleNode

        Returns
        ----------
        None

        """
        raise NotImplementedError()

    @abc.abstractmethod
    def next(self):
        """
        Return a node from a frontier and remove it.

        Returns
        ----------
        EightPuzzleNode

        Raises
        ----------
        IndexError
            if the frontier is empty.

        """
        raise NotImplementedError()


class DFSFrontier(Frontier):
    """An example of how to implement a depth-first frontier (stack)."""

    def __init__(self):
        """Create a frontier."""
        self.stack = []

    def is_empty(self):
        """Return True if empty."""
        return len(self.stack) == 0

    def add(self, node):
        """
        Add a node into the frontier.

        Parameters
        ----------
        node : EightPuzzleNode

        Returns
        ----------
        None

        """
        for n in self.stack:
            # HERE we assume that state implements __eq__() function.
            # This could be improve further by using a set() datastructure,
            # by implementing __hash__() function.
            if n.state == node.state:
                return None
        self.stack.append(node)

    def next(self):
        """
        Return a node from a frontier and remove it.

        Returns
        ----------
        EightPuzzleNode

        Raises
        ----------
        IndexError
            if the frontier is empty.

        """
        return self.stack.pop()


REMOVED = -100


class GreedyFrontier(Frontier):
    """A frontier for greedy search."""

    def __init__(self, h_func, goal_state):
        """
        Create a frontier.

        Parameters
        ----------
        h_func : callable h(state, goal_state)
            a heuristic function to score a state.
        goal_state : EightPuzzleState
            a goal state used to compute h()

        """
        self.h = h_func
        self.goal = goal_state
        # TODO: 3
        # Note that you have to create a data structure here and
        # implement the rest of the abstract methods.
        self.queue = []
        self.entryFinder = {}
        self.counter = itertools.count()

    def is_empty(self):
        return len(self.queue) == 0

    def add(self, node: EightPuzzleNode):
        if node in self.entryFinder:
            self.remove(node)
        count = next(self.counter)

        priority = self.h(node.state, self.goal)
        entry = [priority, count, node]
        self.entryFinder[node] = entry
        heapq.heappush(self.queue, entry)

    def remove(self, node: EightPuzzleNode):
        entry = self.entryFinder.pop(node)
        entry[-1] = REMOVED

    def next(self):
        while self.queue:
            priority, count, node = heapq.heappop(self.queue)
            if node is not REMOVED:
                del self.entryFinder[node]
                return node
        raise KeyError('pop from an empty priority queue')


class AStarFrontier(Frontier):
    """A frontier for greedy search."""

    def __init__(self, h_func, goal_state):
        """
        Create a frontier.

        Parameters
        ----------
        h_func : callable h(state)
            a heuristic function to score a state.
        goal_state : EightPuzzleState
            a goal state used to compute h()


        """
        self.h = h_func
        self.goal = goal_state
        # TODO: 4
        # Note that you have to create a data structure here and
        # implement the rest of the abstract methods.
        self.queue = []
        self.entryFinder = {}
        self.counter = itertools.count()

    def is_empty(self):
        return len(self.queue) == 0

    def add(self, node: EightPuzzleNode):
        if node in self.entryFinder:
            self.remove(node)
        count = next(self.counter)

        priority = node.path_cost + self.h(node.state, self.goal)
        entry = [priority, count, node]
        self.entryFinder[node] = entry
        heapq.heappush(self.queue, entry)

    def remove(self, node: EightPuzzleNode):
        entry = self.entryFinder.pop(node)
        entry[-1] = REMOVED

    def next(self):
        while self.queue:
            priority, count, node = heapq.heappop(self.queue)
            if node is not REMOVED:
                del self.entryFinder[node]
                return node
        raise KeyError('pop from an empty priority queue')


def _parity(board):
    """Return parity of a square matrix."""
    inversions = 0
    nums = []
    for row in board:
        for value in row:
            nums.append(value)
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] != 0 and nums[j] != 0 and nums[i] > nums[j]:
                inversions += 1
    return inversions % 2


def _is_reachable(board1, board2):
    """Return True if two N-Puzzle state are reachable to each other."""
    return _parity(board1) == _parity(board2)


def getActionSpace(state: EightPuzzleState, y: int, x: int):

    actionSpace = copy.deepcopy(state.action_space)

    rowTop = y - 1
    if rowTop < 0:
        actionSpace.remove('u')

    rowBottom = y + 1
    if rowBottom > len(state.board) - 1:
        actionSpace.remove('d')

    colLeft = x - 1
    if colLeft < 0:
        actionSpace.remove('l')

    colRight = x + 1
    if colRight > len(state.board[0]) - 1:
        actionSpace.remove('r')

    return actionSpace


def getNeighborNodes(currentNode: EightPuzzleNode):
    neighborNodes = []
    actionSpace = getActionSpace(
        currentNode.state, currentNode.state.y, currentNode.state.x)

    for action in actionSpace:
        successorBoard: EightPuzzleState = currentNode.state.successor(action)
        neighborNodes.append(EightPuzzleNode(
            successorBoard, currentNode, action))

    return neighborNodes


def graph_search(init_state, goal_state, frontier):
    """
    Search for a plan to solve problem.

    Parameters
    ----------
    init_state : EightPuzzleState
        an initial state
    goal_state : EightPuzzleState
        a goal state
    frontier : Frontier
        an implementation of a frontier which dictates the order of exploreation.

    Returns
    ----------
    plan : List[string] or None
        A list of actions to reach the goal, None if the search fails.
        Your plan should NOT include 'INIT'.
    num_nodes: int
        A number of nodes generated in the search.

    """
    if not _is_reachable(init_state.board, goal_state.board):
        return None, 0
    if init_state.is_goal(goal_state.board):
        return [], 0
    num_nodes = 0
    solution = []
    # Perform graph search
    root_node = EightPuzzleNode(init_state, action='INIT')
    frontier.add(root_node)
    exploreNodes = set()
    num_nodes += 1

    # TODO: 5
    while not frontier.is_empty():
        currentNode = frontier.next()

        if currentNode.state.is_goal():
            solutionNode = currentNode
            break

        if currentNode.state not in exploreNodes:
            exploreNodes.add(currentNode.state)
            num_nodes += 1

            for node in getNeighborNodes(currentNode):
                frontier.add(node)

    paths = solutionNode.trace()
    paths.pop(0)

    for path in paths:
        solution.append(path.action)
    return solution, num_nodes


def test_by_hand(verbose=True):
    """Run a graph-search."""
    goal_state = EightPuzzleState([[1, 2, 3], [4, 5, 6], [7, 8, 0]])
    init_state = EightPuzzleState.initializeState()
    while not _is_reachable(goal_state.board, init_state.board):
        init_state = EightPuzzleState.initializeState()

    # Change this to your own implementation.
    # frontier = DFSFrontier()
    # frontier = GreedyFrontier(eightPuzzleH1, goal_state)
    frontier = AStarFrontier(eightPuzzleH2, goal_state)
    if verbose:
        print(init_state)
    plan, num_nodes = graph_search(init_state, goal_state, frontier)
    if verbose:
        print(f'A solution is found after generating {num_nodes} nodes.')
    if verbose:
        for action in plan:
            print(f'- {action}')
    return len(plan), num_nodes


def experiment(n=10000):
    """Run experiments and report number of nodes generated."""
    result = defaultdict(list)
    for __ in range(n):
        d, n = test_by_hand(False)
        result[d].append(n)
    max_d = max(result.keys())
    for i in range(max_d + 1):
        n = result[d]
        if len(n) == 0:
            continue
        print(f'{d}, {len(n)}, {sum(n)/len(n)}')


if __name__ == '__main__':
    __, __ = test_by_hand()
    # experiment()  #  run graph search 10000 times and report result.
