import random

from pacai.agents.base import BaseAgent
from pacai.agents.search.multiagent import MultiAgentSearchAgent
import pacai.core.distance as dist
from pacai.core.directions import Directions

class ReflexAgent(BaseAgent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.
    You are welcome to change it in any way you see fit,
    so long as you don't touch the method headers.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        `ReflexAgent.getAction` chooses among the best options according to the evaluation function.

        Just like in the previous project, this method takes a
        `pacai.core.gamestate.AbstractGameState` and returns some value from
        `pacai.core.directions.Directions`.
        """

        # Collect legal moves.
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions.
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best.

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current `pacai.bin.pacman.PacmanGameState`
        and an action, and returns a number, where higher numbers are better.
        Make sure to understand the range of different values before you combine them
        in your evaluation function.
        """

        successorGameState = currentGameState.generatePacmanSuccessor(action)

        # Useful information you can extract.
        # newPosition = successorGameState.getPacmanPosition()
        # oldFood = currentGameState.getFood()
        # newGhostStates = successorGameState.getGhostStates()
        # newScaredTimes = [ghostState.getScaredTimer() for ghostState in newGhostStates]

        # *** Your Code Here ***
        newPosition = successorGameState.getPacmanPosition()
        oldFood = currentGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()

        # avg dist to ghosts using manhattan
        distGhost = [dist.manhattan(newPosition, ghostState.getPosition()) for ghostState in
                     newGhostStates]
        ghostSum = sum(distGhost)
        # average the dist to ghosts (use number of agents minus pacman)
        ghostAvg = ghostSum / (currentGameState.getNumAgents() - 1)
        # reciprocal
        # error: division by 0
        # add something to the denominator so it never divides by 0
        # ghosts should lower the score so subtract later
        ghostScore = 1.0 / (ghostAvg + 0.001)

        # min dist to food
        distFood = [dist.manhattan(newPosition, food) for food in oldFood.asList()]
        # no food left, make it a large value
        if not distFood:
            # large number to lower food score but not be 0
            distFood = [1000]
        closestFood = min(distFood)
        # reciprocal
        # error: division by 0
        # add something to the denominator so it never divides by 0
        foodScore = 1.0 / (closestFood + 0.001)

        # the current score.
        score = successorGameState.getScore()

        f = score + foodScore - ghostScore

        return f

class MinimaxAgent(MultiAgentSearchAgent):
    """
    A minimax agent.

    Here are some method calls that might be useful when implementing minimax.

    `pacai.core.gamestate.AbstractGameState.getNumAgents()`:
    Get the total number of agents in the game

    `pacai.core.gamestate.AbstractGameState.getLegalActions`:
    Returns a list of legal actions for an agent.
    Pacman is always at index 0, and ghosts are >= 1.

    `pacai.core.gamestate.AbstractGameState.generateSuccessor`:
    Get the successor game state after an agent takes an action.

    `pacai.core.directions.Directions.STOP`:
    The stop direction, which is always legal, but you may not want to include in your search.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    # Returns the minimax action from the current gameState
    
    def getAction(self, gamestate):
        # helper functs
        # pseudocode
        # function MAX-VALUE(state) returns a utility value
        # if TERMINAL-TEST(state) then return UTILITY(state)
        # v ← −∞
        # for each a in ACTIONS(state) do
        # v ← MAX(v, MIN-VALUE(RESULT(s, a)))
        # return v
        def maxValue(state, index, depth):
            # number of agents
            numAgents = state.getNumAgents()
            # game over or too deep
            if state.isOver() or depth >= self.getTreeDepth():
                return self.getEvaluationFunction()(state)
            value = -9999
            legalActions = state.getLegalActions(index)
            # take out stop
            if Directions.STOP in legalActions:
                legalActions.remove(Directions.STOP)
            for a in legalActions:
                successorGameState = state.generateSuccessor(index, a)
                # last one - pacman
                if index == numAgents - 1:
                    value = max(value, minValue(successorGameState, index, depth + 1))
                else:
                    value = max(value, minValue(successorGameState, index + 1, depth + 1))
            return value

        # pseudocode
        # function MIN-VALUE(state) returns a utility value
        # if TERMINAL-TEST(state) then return UTILITY(state)
        # v ← ∞
        # for each a in ACTIONS(state) do
        # v ← MIN(v, MAX-VALUE(RESULT(s, a)))
        # return v
        def minValue(state, index, depth):
            # number of agents
            numAgents = state.getNumAgents()
            # game over or too deep
            if state.isOver() or depth >= self.getTreeDepth():
                return self.getEvaluationFunction()(state)
            value = 9999
            legalActions = state.getLegalActions(index)
            # take out stop
            if Directions.STOP in legalActions:
                legalActions.remove(Directions.STOP)
            for a in legalActions:
                successorGameState = state.generateSuccessor(index, a)
                # last one - pacman
                if index == numAgents - 1:
                    value = min(value, maxValue(successorGameState, index, depth + 1))
                else:
                    value = min(value, maxValue(successorGameState, index + 1, depth + 1))
            return value
        
        # pseudocode
        # function MINIMAX-DECISION(state) returns an action
        # return arg maxa ∈ ACTIONS(s) MIN-VALUE(RESULT(state, a))
        legalActions = gamestate.getLegalActions(0)
        if Directions.STOP in legalActions:
            legalActions.remove(Directions.STOP)
        values = [(minValue(gamestate.generateSuccessor(0, a), 1, 1), a) for a in legalActions]
        bestValue, bestAction = max(values)
        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    A minimax agent with alpha-beta pruning.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    # Returns the minimax action from the current gameState
    def getAction(self, gamestate):
        # helper functs
        # pseudocode
        # function MAX-VALUE(state,α, β) returns a utility value
        # if TERMINAL-TEST(state) then return UTILITY(state)
        # v ← −∞
        # for each a in ACTIONS(state) do
        # v ← MAX(v, MIN-VALUE(RESULT(s,a),α, β))
        # if v ≥ β then return v
        # α ← MAX(α, v)
        # return v
        def maxValue(state, index, depth, alpha, beta):
            # number of agents
            numAgents = state.getNumAgents()
            # game over or too deep
            if state.isOver() or depth >= self.getTreeDepth():
                return self.getEvaluationFunction()(state)
            value = -9999
            legalActions = state.getLegalActions(index)
            # take out stop
            if Directions.STOP in legalActions:
                legalActions.remove(Directions.STOP)
            for a in legalActions:
                successorGameState = state.generateSuccessor(index, a)
                if index == numAgents - 1:
                    value = max(value, minValue(successorGameState, index, depth + 1, alpha, beta))
                else:
                    value = max(value, minValue(successorGameState, index + 1, depth + 1, alpha,
                    beta))
                alpha = max(alpha, value)
                if value >= beta:
                    return value
            return value
        
        # pseudocode
        # function MIN-VALUE(state,α, β) returns a utility value
        # if TERMINAL-TEST(state) then return UTILITY(state)
        # v ← +∞
        # for each a in ACTIONS(state) do
        # v ← MIN(v, MAX-VALUE(RESULT(s,a) ,α, β))
        # if v ≤ α then return v
        # β ← MIN(β, v)
        # return v
        def minValue(state, index, depth, alpha, beta):
            # number of agents
            numAgents = state.getNumAgents()
            # game over or too deep
            if state.isOver() or depth >= self.getTreeDepth():
                return self.getEvaluationFunction()(state)
            value = 9999
            legalActions = state.getLegalActions(index)
            # take out stop
            if Directions.STOP in legalActions:
                legalActions.remove(Directions.STOP)
            for a in legalActions:
                successorGameState = state.generateSuccessor(index, a)
                if index == numAgents - 1:
                    value = min(value, maxValue(successorGameState, index, depth + 1, alpha, beta))
                else:
                    value = min(value, maxValue(successorGameState, index + 1, depth + 1, alpha,
                    beta))
                beta = min(beta, value)
                if value <= alpha:
                    return value
            return value

        # pseudocode
        # function ALPHA-BETA-SEARCH(state) returns an action
        # v ← MAX-VALUE(state,−∞,+∞)
        # return the action in ACTIONS(state) with value v
        legalActions = gamestate.getLegalActions(0)
        if Directions.STOP in legalActions:
            legalActions.remove(Directions.STOP)
        # set alpha and beta
        alpha = -9999
        beta = 9999
        bestValue = -9999
        bestAction = legalActions[0]
        values = [(minValue(gamestate.generateSuccessor(0, a), 1, 1, alpha, beta), a) for a in
                  legalActions]
        for val in values:
            valValue, valAction = val
            if valValue > bestValue:
                bestValue = valValue
                bestAction = valAction
            alpha = max(alpha, bestValue)
            if bestValue >= beta:
                return bestAction
        return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    An expectimax agent.

    All ghosts should be modeled as choosing uniformly at random from their legal moves.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the expectimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    # Returns the expectimax action from the current gameState
    def getAction(self, gamestate):
        # helpers
        # pseudocode
        # def maxValue(s)
        # values = [value(s') for s' in successors(s)]
        # return max(values)
        def maxValue(state, index, depth):
            if state.isOver() or depth >= self.getTreeDepth():
                return self.getEvaluationFunction()(state)
            # without index change -> invalid index error (calling for ghosts)
            # only using this for pacman who has an index 0
            index = 0
            legalActions = state.getLegalActions(index)
            # remove stop
            if Directions.STOP in legalActions:
                legalActions.remove(Directions.STOP)
            values = [expValue(state.generateSuccessor(index, a), index + 1, depth + 1) for a in
                      legalActions]
            value = max(values)
            return value

        # pseudocode
        # def expValue(s)
        # values = [value(s') for s' in successors(s)]
        # weights = [probability(s, s') for s' in successors(s)]
        # return expectation(values, weights)
        def expValue(state, index, depth):
            # number of agents
            numAgents = state.getNumAgents()
            if state.isOver() or depth >= self.getTreeDepth():
                return self.getEvaluationFunction()(state)
            legalActions = state.getLegalActions(index)
            # remove stop
            if Directions.STOP in legalActions:
                legalActions.remove(Directions.STOP)
            value = 0
            # each has equal probability
            probability = 1.0 / len(legalActions)
            for action in legalActions:
                successorGameState = state.generateSuccessor(index, action)
                if index == numAgents - 1:
                    # multiply by probability/weight
                    value += probability * maxValue(successorGameState, index, depth + 1)
                else:
                    # multiply by probability/weight
                    value += probability * expValue(successorGameState, index + 1, depth)
            return value
        
        # pseudocode
        legalActions = gamestate.getLegalActions(0)
        # remove stop
        if Directions.STOP in legalActions:
            legalActions.remove(Directions.STOP)
        values = [(expValue(gamestate.generateSuccessor(0, a), 1, 1), a) for a in
                  legalActions]
        bestValue, bestAction = max(values)
        return bestAction

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable evaluation function.

    DESCRIPTION: I took my previous eval funtion and added consideration of how close the
    ghosts were and how scared the ghosts were.
    """

    newPosition = currentGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.getScaredTimer() for ghostState in newGhostStates]

    # avg dist to ghosts using manhattan
    distGhost = [dist.manhattan(newPosition, ghostState.getPosition()) for ghostState in
                 newGhostStates]
    ghostSum = sum(distGhost)
    # average the dist to ghosts (use number of agents minus pacman)
    ghostAvg = ghostSum / (currentGameState.getNumAgents() - 1)
    ghostTotal = ghostAvg
    # too close to a ghost
    closestGhost = min(distGhost)
    if (closestGhost < 2):
        ghostTotal -= 100
    # scared
    for scared in newScaredTimes:
        if scared == 0:
            ghostTotal -= 25
        # not scared
        else:
            ghostTotal += 50
    # reciprocal
    # error: division by 0
    # add something to the denominator so it never divides by 0
    # ghosts should lower the score so subtract later
    ghostScore = 1.0 / (ghostTotal + 0.001)

    # min dist to food
    distFood = [dist.manhattan(newPosition, food) for food in oldFood.asList()]
    # no food left, make it a large value
    if not distFood:
        # large number to lower food score but not be 0
        distFood = [1000]
    closestFood = min(distFood)
    # reciprocal
    # error: division by 0
    # add something to the denominator so it never divides by 0
    foodScore = 1.0 / (closestFood + 0.001)

    # the current score.
    score = currentGameState.getScore()

    f = score + foodScore - ghostScore

    return f

class ContestAgent(MultiAgentSearchAgent):
    """
    Your agent for the mini-contest.

    You can use any method you want and search to any depth you want.
    Just remember that the mini-contest is timed, so you have to trade off speed and computation.

    Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
    just make a beeline straight towards Pacman (or away if they're scared!)

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
