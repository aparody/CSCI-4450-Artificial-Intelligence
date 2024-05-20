# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions

    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"   
    class SearchNode:

        def __init__(self, state, action=None, parent=None):
            self.state = state
            self.action = action
            self.parent = parent

        def plan(self):
            path = []
            search = self
            while search:
                if search.action:
                    path.append(search.action)
                search = search.parent
            return list(reversed(path))
            
    fringe = util.Stack()
    closed = set()
    fringe.push(SearchNode(problem.getStartState()))

    while not fringe.isEmpty():
        node = fringe.pop()  
        if problem.isGoalState(node.state):
            return node.plan()
        for successor in problem.getSuccessors(node.state):
            child = SearchNode(successor[0], successor[1], node)
            if child.state not in closed:
                closed.add(node.state)
                fringe.push(child)

    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    class SearchNode:

        def __init__(self, state, action=None, parent=None):
            self.state = state
            self.action = action
            self.parent = parent

        def plan(self):
            path = []
            search = self
            while search:
                if search.action:
                    path.append(search.action)
                search = search.parent
            return list(reversed(path)) 
         
    fringe = util.Queue()
    closed = set()
    fringe.push(SearchNode(problem.getStartState()))

    while not fringe.isEmpty():
        node = fringe.pop()
        closed.add(node.state)
        if problem.isGoalState(node.state):
            return node.plan()
        for successor in problem.getSuccessors(node.state):
            child = SearchNode(successor[0], successor[1], node)
            if child.state not in closed:
                closed.add(child.state)
                fringe.push(child)

    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    class SearchNode:

        def __init__(self, state, action=None, parent=None, pathCost = 0):
            self.state = state
            self.action = action
            self.parent = parent
            if parent:
                self.pathCost = pathCost + parent.pathCost
            else:
                self.pathCost = pathCost

        def plan(self):
            path = []
            search = self
            while search:
                if search.action:
                    path.append(search.action)
                search = search.parent
            return list(reversed(path))
    
    frontier = util.PriorityQueue()
    explored = set()
    frontier.push(SearchNode(problem.getStartState()), SearchNode(problem.getStartState()).pathCost)

    while not frontier.isEmpty():
        node = frontier.pop()
        
        if problem.isGoalState(node.state):
            return node.plan()
        if node.state not in explored:
            explored.add(node.state)
            for successor in problem.getSuccessors(node.state):
                child = SearchNode(successor[0], successor[1], node, successor[2])
                frontier.update(child, child.pathCost)

    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    class SearchNode:

        def __init__(self, state, action=None, parent=None, h = None, g = None):
            self.state = state
            self.action = action
            self.parent = parent
            self.h = h
            if parent:
                self.g = g + parent.g
            else:
                self.g = 0
            self.f = self.g + self.h

        def plan(self):
            path = []
            search = self
            while search:
                if search.action:
                    path.append(search.action)
                search = search.parent
            return list(reversed(path))
        
    def createNode(state, action=None, parent=None, cost=None):
        if hasattr(problem, "heuristicInfo"):
            if parent:
                if parent == problem.heuristicInfo["parent"]:
                    problem.heuristicInfo["sameParent"] = True
                else:
                    problem.heuristicInfo["sameParent"] = False
            problem.heuristicInfo["parent"] = parent
        hValue = heuristic(state, problem)
        return SearchNode(state, action, parent, hValue, cost)

    frontier = util.PriorityQueue()
    explored = set()
    frontier.push(createNode(problem.getStartState()), createNode(problem.getStartState()).f)
    gCost = {}

    while not frontier.isEmpty():
        node = frontier.pop()       
        if node.state not in explored or gCost[node.state] > node.g:
            explored.add(node.state)
            gCost[node.state] = node.g
            if problem.isGoalState(node.state):
                return node.plan()
            for successor in problem.getSuccessors(node.state):
                child = createNode(successor[0], successor[1], node, successor[2])
                if child.h < float("inf"):
                    frontier.push(child, child.f)

    util.raiseNotDefined()

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
