from pacai.agents.learning.reinforcement import ReinforcementAgent
from pacai.util import reflection
from pacai.util import probability
import random

class QLearningAgent(ReinforcementAgent):
    """
    A Q-Learning agent.

    Some functions that may be useful:

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getAlpha`:
    Get the learning rate.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getDiscountRate`:
    Get the discount rate.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getEpsilon`:
    Get the exploration probability.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getLegalActions`:
    Get the legal actions for a reinforcement agent.

    `pacai.util.probability.flipCoin`:
    Flip a coin (get a binary value) with some probability.

    `random.choice`:
    Pick randomly from a list.

    Additional methods to implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Compute the action to take in the current state.
    With probability `pacai.agents.learning.reinforcement.ReinforcementAgent.getEpsilon`,
    we should take a random action and take the best policy action otherwise.
    Note that if there are no legal actions, which is the case at the terminal state,
    you should choose None as the action.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.update`:
    The parent class calls this to observe a state transition and reward.
    You should do your Q-Value update here.
    Note that you should never call this function, it will be called on your behalf.

    DESCRIPTION: Similar to value iteration agent, but with Q instead of V
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

        # You can initialize Q-values here.
        self.qValues = {}

    def getQValue(self, state, action):
        """
        Get the Q-Value for a `pacai.core.gamestate.AbstractGameState`
        and `pacai.core.directions.Directions`.
        Should return 0.0 if the (state, action) pair has never been seen.
        """
        return self.qValues.get((state, action), 0.0)

    def getValue(self, state):
        """
        Return the value of the best action in a state.
        I.E., the value of the action that solves: `max_action Q(state, action)`.
        Where the max is over legal actions.
        Note that if there are no legal actions, which is the case at the terminal state,
        you should return a value of 0.0.

        This method pairs with `QLearningAgent.getPolicy`,
        which returns the actual best action.
        Whereas this method returns the value of the best action.
        """
        legalActions = self.getLegalActions(state)
        # terminal
        if not legalActions:
            return 0.0

        # max q value for possible actions
        values = (self.getQValue(state, a) for a in legalActions)
        return max(values)
    
    def getPolicy(self, state):
        """
        Return the best action in a state.
        I.E., the action that solves: `max_action Q(state, action)`.
        Where the max is over legal actions.
        Note that if there are no legal actions, which is the case at the terminal state,
        you should return a value of None.

        This method pairs with `QLearningAgent.getValue`,
        which returns the value of the best action.
        Whereas this method returns the best action itself.
        """
        legalActions = self.getLegalActions(state)
        # terminal state
        if not legalActions:
            return None
        
        results = ((self.getQValue(state, a), a) for a in legalActions)
        maxValue, maxAction = max(results)

        # action with max value
        return maxAction

    def getAction(self, state):
        """Compute the action to take in the current state.
    With probability `pacai.agents.learning.reinforcement.ReinforcementAgent.getEpsilon`,
    we should take a random action and take the best policy action otherwise.
    Note that if there are no legal actions, which is the case at the terminal state,
    you should choose None as the action."""
        legalActions = self.getLegalActions(state)
        # terminal state
        if not legalActions:
            return None
        
        epsilon = self.getEpsilon()
        if probability.flipCoin(epsilon):
            # take random action
            return random.choice(legalActions)
        else:
            # best policy action
            return self.getPolicy(state)

    def update(self, state, action, nextState, reward):
        """The parent class calls this to observe a state transition and reward.
    You should do your Q-Value update here.
    Note that you should never call this function, it will be called on your behalf."""
        alpha = self.getAlpha()
        gamma = self.getDiscountRate()
        # q(s, a)
        sQValue = self.getQValue(state, action)
        # q(s', a)
        spQValue = self.getValue(nextState)

        # sample = reward  + gamma * max action q(s', a)
        sample = reward + gamma * spQValue
        # q(s, a) = (1 - alpha) * q(s, a) + alpha * sample
        q = (1 - alpha) * sQValue + alpha * sample
        
        # update qValues
        self.qValues[(state, action)] = q

class PacmanQAgent(QLearningAgent):
    """
    Exactly the same as `QLearningAgent`, but with different default parameters.
    """

    def __init__(self, index, epsilon = 0.05, gamma = 0.8, alpha = 0.2, numTraining = 0, **kwargs):
        kwargs['epsilon'] = epsilon
        kwargs['gamma'] = gamma
        kwargs['alpha'] = alpha
        kwargs['numTraining'] = numTraining

        super().__init__(index, **kwargs)

    def getAction(self, state):
        """
        Simply calls the super getAction method and then informs the parent of an action for Pacman.
        Do not change or remove this method.
        """

        action = super().getAction(state)
        self.doAction(state, action)

        return action

class ApproximateQAgent(PacmanQAgent):
    """
    An approximate Q-learning agent.

    You should only have to overwrite `QLearningAgent.getQValue`
    and `pacai.agents.learning.reinforcement.ReinforcementAgent.update`.
    All other `QLearningAgent` functions should work as is.

    Additional methods to implement:

    `QLearningAgent.getQValue`:
    Should return `Q(state, action) = w * featureVector`,
    where `*` is the dotProduct operator.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.update`:
    Should update your weights based on transition.

    DESCRIPTION: Tweak functions to fit new formulas
    """

    def __init__(self, index,
            extractor = 'pacai.core.featureExtractors.IdentityExtractor', **kwargs):
        super().__init__(index, **kwargs)
        self.featExtractor = reflection.qualifiedImport(extractor)

        # You might want to initialize weights here.
        self.weights = {}

    def final(self, state):
        """
        Called at the end of each game.
        """

        # Call the super-class final method.
        super().final(state)

        # Did we finish training?
        if self.episodesSoFar == self.numTraining:
            # You might want to print your weights here for debugging.
            # *** Your Code Here ***
            return self.weights
        
    def getQValue(self, state, action):
        """Should return `Q(state, action) = w * featureVector`,
    where `*` is the dotProduct operator."""
        features = self.featExtractor.getFeatures(self, state, action)
        # print(features)
        # dot product sum(w*f) for each f key
        q = 0
        for f in features.keys():
            # no weight, set to 1
            if f not in self.weights:
                self.weights[f] = 1
            q += self.weights[f] * features[f]
        return q

    def update(self, state, action, nextState, reward):
        """Should update your weights based on transition."""
        alpha = self.getAlpha()
        gamma = self.getDiscountRate()
        # V'(s)
        vValue = self.getValue(nextState)
        # q(s, a)
        sQValue = self.getQValue(state, action)
        features = self.featExtractor.getFeatures(self, state, action)

        for f in features.keys():
            # correction = (R(s, a) + gamma * V'(s)) - Q(s, a)
            correction = (reward + gamma * vValue) - sQValue
            # wi = wi + alpha[correction] fi(s, a)
            self.weights[f] = self.weights[f] + alpha * correction * features[f]