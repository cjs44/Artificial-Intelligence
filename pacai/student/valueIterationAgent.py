from pacai.agents.learning.value import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
    A value iteration agent.

    Make sure to read `pacai.agents.learning` before working on this class.

    A `ValueIterationAgent` takes a `pacai.core.mdp.MarkovDecisionProcess` on initialization,
    and runs value iteration for a given number of iterations using the supplied discount factor.

    Some useful mdp methods you will use:
    `pacai.core.mdp.MarkovDecisionProcess.getStates`,
    `pacai.core.mdp.MarkovDecisionProcess.getPossibleActions`,
    `pacai.core.mdp.MarkovDecisionProcess.getTransitionStatesAndProbs`,
    `pacai.core.mdp.MarkovDecisionProcess.getReward`.

    Additional methods to implement:

    `pacai.agents.learning.value.ValueEstimationAgent.getQValue`:
    The q-value of the state action pair (after the indicated number of value iteration passes).
    Note that value iteration does not necessarily create this quantity,
    and you may have to derive it on the fly.

    `pacai.agents.learning.value.ValueEstimationAgent.getPolicy`:
    The policy is the best action in the given state
    according to the values computed by value iteration.
    You may break ties any way you see fit.
    Note that if there are no legal actions, which is the case at the terminal state,
    you should return None.
    """

    def __init__(self, index, mdp, discountRate = 0.9, iters = 100, **kwargs):
        super().__init__(index, **kwargs)

        self.mdp = mdp
        self.discountRate = discountRate
        self.iters = iters
        self.values = {}  # A dictionary which holds the q-values for each state.

        # Compute the values here.
        # Vi+1(s) = max action of sum(t * [r + discount*Vi])
        for i in range(self.iters):
            nextiValues = {}
            for state in self.mdp.getStates():
                # not in the terminal state
                if not self.mdp.isTerminal(state):
                    # Best action with the current policy
                    bestAction = self.getAction(state)
                    # Q for best action
                    nextiValues[state] = self.getQValue(state, bestAction)
            # "throw out" old vi values
            self.values = nextiValues
            
    def getValue(self, state):
        """
        Return the value of the state (computed in __init__).
        """

        return self.values.get(state, 0.0)

    def getAction(self, state):
        """
        Returns the policy at the state (no exploration).
        """

        return self.getPolicy(state)
    
    def getPolicy(self, state):
        """The policy is the best action in the given state
    according to the values computed by value iteration.
    You may break ties any way you see fit.
    Note that if there are no legal actions, which is the case at the terminal state,
    you should return None."""
        # pacai.core.mdp.MarkovDecisionProcess.getPossibleActions
        possibleActions = self.mdp.getPossibleActions(state)
        # no legal actions
        if not possibleActions:
            return None
        results = ((self.getQValue(state, a), a) for a in possibleActions)
        maxValue, maxAction = max(results)

        # action with max value
        return maxAction
    
    def getQValue(self, state, action):
        """The q-value of the state action pair (after the indicated number of value iteration
          passes). Note that value iteration does not necessarily create this quantity,
    and you may have to derive it on the fly."""
        q = 0
        for tState, tProb in self.mdp.getTransitionStatesAndProbs(state, action):
            q += tProb * (self.mdp.getReward(state, action, tState) + (self.discountRate
                                                                       * self.getValue(tState)))
        return q