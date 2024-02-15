import random
from pacai.agents.capture.capture import CaptureAgent
from pacai.core.directions import Directions

# DO NOT MESS WITH THIS FUNCTION ANYMORE
def createTeam(firstIndex, secondIndex, isRed,
        first = 'FinalAttacker',
        second = 'FinalDefender'):
    """
    This function should return a list of two agents that will form the capture team,
    initialized using firstIndex and secondIndex as their agent indexed.
    isRed is True if the red team is being created,
    and will be False if the blue team is being created.
    """
    return [
        FinalAttacker(firstIndex),
        FinalDefender(secondIndex),
    ]

# A reflex agent that seeks food.
# This agent will give you an idea of what an offensive agent might look like,
# but it is by no means the best or only way to build an offensive agent.
class FinalAttacker(CaptureAgent):

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def registerInitialState(self, gameState):
        """
        This method handles the initial setup of the agent and populates useful fields,
        such as the team the agent is on and the `pacai.core.distanceCalculator.Distancer`.

        IMPORTANT: If this method runs for more than 15 seconds, your agent will time out.
        """

        super().registerInitialState(gameState)

        # Your initialization code goes here, if you need any.

    def chooseAction(self, gameState):
        # legal moves
        legalMoves = gameState.getLegalActions(self.index)

        # choose best action
        scores = [self.getScore(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        # random among the best
        chosenIndex = random.choice(bestIndices)

        return legalMoves[chosenIndex]
    
    def getScore(self, gameState, action):
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)

        # feature * weight
        score = sum(features[feature] * weights[feature] for feature in features)
        return score

    def getFeatures(self, gameState, action):
        # eval function
        features = {}
        successorGameState = gameState.generateSuccessor(self.index, action)

        newPosition = successorGameState.getAgentPosition(self.index)
        oldFood = gameState.getFood()
        # newGhostStates = successorGameState.getAgentStates()

        # distance to the nearest enemy ghost
        # find enemies
        enemies = [successorGameState.getAgentState(i) for i in
                   self.getOpponents(successorGameState)]
        # see if ghost
        ghosts = [e for e in enemies if not e.isPacman() and e.getPosition() is not None]

        # could give pacman a radius of like 3 squares and if a ghost enters it then
        # include the ghosts in your calculation, else get to the next pellet
        # radius for 'close' ghosts
        ghostRadius = 3

        # ghost is within (<=) the radius
        ghostsWithinRadius = [ghost for ghost in ghosts if
                              self.getMazeDistance(newPosition, ghost.getPosition()) <= ghostRadius]
        if ghostsWithinRadius:
            features['nearestGhost'] = 0
            features['ghostInRadius'] = 1
        # none in radius
        else:
            # min distance to food
            distFood = [self.getMazeDistance(newPosition, food) for food in oldFood.asList()]
            # no food left, make it a large value
            if not distFood:
                # large number to lower food score but not be 0
                distFood = [1000]
            closestFood = min(distFood)
            # reciprocal with a small constant to avoid division by zero
            features['foodScore'] = 1.0 / (closestFood + 0.001)
            # none in radius
            features['ghostInRadius'] = 0

        # distance to the nearest capsule
        capsules = self.getCapsules(gameState)
        if len(capsules) > 0:
            capDist = min([self.getMazeDistance(newPosition, capsule) for capsule in capsules])
            features['capsuleDistance'] = capDist
        else:
            features['capsuleDistance'] = 0

        # The current score.
        features['successorScore'] = successorGameState.getScore()

        return features

    def getWeights(self, gameState, action):
        return {
            'successorScore': 100,
            'foodScore': -1,
            'nearestGhost': -1,
            'capsuleDistance': -2,
            'ghostInRadius': 1
        }
    
class FinalDefender(CaptureAgent):
    """
    Builds off Defense Agent
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
    
    def chooseAction(self, gameState):
        # legal moves
        legalMoves = gameState.getLegalActions(self.index)

        # choose best action
        scores = [self.getScore(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        # random among best
        chosenIndex = random.choice(bestIndices)

        return legalMoves[chosenIndex]

    def getScore(self, gameState, action):
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        score = sum(features[feature] * weights[feature] for feature in features)
        return score

    def getFeatures(self, gameState, action):
        features = {}
        successorGameState = gameState.generateSuccessor(self.index, action)

        newPosition = successorGameState.getAgentPosition(self.index)
        oldFood = super().getFoodYouAreDefending(gameState)

        # min dist to food
        distFood = [super().getMazeDistance(newPosition, food) for food in oldFood.asList()]
        # no food left, make it a large value
        if not distFood:
            # large number to lower food score but not be 0
            distFood = [1000]
        closestFood = min(distFood)
        # reciprocal
        # error: division by 0
        # add something to the denominator so it never divides by 0
        features['foodScore'] = 1.0 / (closestFood + 0.001)

        # Computes whether we're on defense (1) or offense (0).
        myState = successorGameState.getAgentState(self.index)
        features['onDefense'] = 1
        if myState.isPacman():
            features['onDefense'] = 0

        # Computes distance to invaders we can see.
        enemies = [successorGameState.getAgentState(i) for i in
                   self.getOpponents(successorGameState)]
        invaders = [e for e in enemies if e.isPacman() and e.getPosition() is not None]
        features['numInvaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.getMazeDistance(newPosition, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)

        # look at ghost on other side before it crosses and invades as pacman
        ghosts = [e for e in enemies if not e.isPacman() and e.getPosition() is not None]
        if len(ghosts) > 0:
            dists = [self.getMazeDistance(newPosition, a.getPosition()) for a in ghosts]
            features['ghostDist'] = min(dists)
        else:
            features['ghostDist'] = 0

        if action == Directions.STOP:
            features['stop'] = 1

        rev = Directions.REVERSE[gameState.getAgentState(self.index).getDirection()]
        if action == rev:
            features['reverse'] = 1

        return features

    def getWeights(self, gameState, action):
        return {
            'numInvaders': -1000,
            'onDefense': 100,
            'invaderDistance': -20,
            'stop': -100,
            'reverse': -2,
            'foodScore': -1,
            'ghostDist': -1
        }