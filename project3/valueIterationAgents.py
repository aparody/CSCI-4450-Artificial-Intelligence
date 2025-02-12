# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import collections

import mdp
import util
from learningAgents import ValueEstimationAgent


class ValueIterationAgent(ValueEstimationAgent):
    """
    * Please read learningAgents.py before reading this.*

    A ValueIterationAgent takes a Markov decision process
    (see mdp.py) on initialization and runs value iteration
    for a given number of iterations using the supplied
    discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
        Your value iteration agent should take an mdp on
        construction, run the indicated number of iterations
        and then act according to the resulting policy.

        Some useful mdp methods you will use:
            mdp.getStates()
            mdp.getPossibleActions(state)
            mdp.getTransitionStatesAndProbs(state, action)
            mdp.getReward(state, action, nextState)
            mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for i in range (self.iterations):
            values = self.values.copy()
            for state in self.mdp.getStates():
                self.values[state] = -float('inf')
                for action in self.mdp.getPossibleActions(state):
                    temp = 0
                    for nextState, probability in self.mdp.getTransitionStatesAndProbs(state, action):
                        temp += probability * (self.mdp.getReward(state, action, nextState) + self.discount * values[nextState])
                    self.values[state] = max(self.values[state], temp)
                if self.values[state] == -float('inf'):
                    self.values[state] = 0.0

    def getValue(self, state):
        """
        Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
        Compute the Q-value of action in state from the
        value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        value = 0
        for nextState, probability in self.mdp.getTransitionStatesAndProbs(state, action):
            value += probability * (self.mdp.getReward(state, action, nextState) + self.discount * self.values[nextState])
        
        return value

    def computeActionFromValues(self, state):
        """
        The policy is the best action in the given state
        according to the values currently stored in self.values.

        You may break ties any way you see fit.  Note that if
        there are no legal actions, which is the case at the
        terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
            return None
        max, action = -float('inf'), None
        for a in self.mdp.getPossibleActions(state):
            temp = self.computeQValueFromValues(state, a)
            if temp > max:
                max, action = temp, a

        return action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
    * Please read learningAgents.py before reading this.*

    An AsynchronousValueIterationAgent takes a Markov decision process
    (see mdp.py) on initialization and runs cyclic value iteration
    for a given number of iterations using the supplied
    discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=1000):
        """
        Your cyclic value iteration agent should take an mdp on
        construction, run the indicated number of iterations,
        and then act according to the resulting policy. Each iteration
        updates the value of only one state, which cycles through
        the states list. If the chosen state is terminal, nothing
        happens in that iteration.

        Some useful mdp methods you will use:
            mdp.getStates()
            mdp.getPossibleActions(state)
            mdp.getTransitionStatesAndProbs(state, action)
            mdp.getReward(state)
            mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        for i in range(self.iterations):
            state = self.mdp.getStates()[i %  len(self.mdp.getStates())]
            best = self.computeActionFromValues(state)
            if best == None:
                value = 0
            else:
                value = self.computeQValueFromValues(state, best)
            self.values[state] = value

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
    * Please read learningAgents.py before reading this.*

    A PrioritizedSweepingValueIterationAgent takes a Markov decision process
    (see mdp.py) on initialization and runs prioritized sweeping value iteration
    for a given number of iterations using the supplied parameters.
    """

    def __init__(self, mdp, discount=0.9, iterations=100, theta=1e-5):
        """
        Your prioritized sweeping value iteration agent should take an mdp on
        construction, run the indicated number of iterations,
        and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def computeQValues(self, state):
        actions = self.mdp.getPossibleActions(state)
        qValues = util.Counter()
        for action in actions:
            qValues[action] = self.computeQValueFromValues(state, action)

        return qValues
    
    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        predecessors = dict()
        queue = util.PriorityQueue()
        
        for state in states:
            predecessors[state]=set()
        for state in states:
            actions = self.mdp.getPossibleActions(state)
            for action in actions:
                possibleNextStates = self.mdp.getTransitionStatesAndProbs(state, action)
                for nextState, predecessor in possibleNextStates:
                    if predecessor > 0:
                        predecessors[nextState].add(state)

        for state in states:
            qValues = self.computeQValues(state)
            if len(qValues) > 0:
                maxQValue = qValues[qValues.argMax()]
                diff = abs(self.values[state] - maxQValue)
                queue.push(state, -diff)

        for i in range(self.iterations):
            if queue.isEmpty():
                return
            state = queue.pop()
            qValues = self.computeQValues(state)
            maxQValue = qValues[qValues.argMax()]
            self.values[state] = maxQValue

            for p in predecessors[state]:
                pQValues = self.computeQValues(p)
                maxQValue = pQValues[pQValues.argMax()]
                diff = abs(self.values[p] - maxQValue)
                if diff > self.theta:
                    queue.update(p, -diff)