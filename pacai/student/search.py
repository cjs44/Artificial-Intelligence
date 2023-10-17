"""
In this file, you will implement generic search algorithms which are called by Pacman agents.
"""
import pacai.util.stack as util1
import pacai.util.queue as util2
import pacai.util.priorityQueue as util3

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first [p 85].

    Your search algorithm needs to return a list of actions that reaches the goal.
    Make sure to implement a graph search algorithm [Fig. 3.7].

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    ```
    print("Start: %s" % (str(problem.startingState())))
    -> Start: (5, 5)
    print("Is the start a goal?: %s" % (problem.isGoal(problem.startingState())))
    -> Is the start a goal?: False
    print("Start's successors: %s" % (problem.successorStates(problem.startingState())))
    -> Start's successors: [((5, 4), 'South', 1), ((4, 5), 'West', 1)]
    ```
    """

    # *** Your Code Here ***
    # initialize the explored set to be empty
    explored = set()
    # list of actions to take
    # initialize empty
    actions = []
    # initialize the fringe with the start state of problem
    start = problem.startingState()
    fringe = util1.Stack()
    fringe.push((start, actions, 0))
    # loop
    while not fringe.isEmpty():
        # choose a leaf node and remove it from the frontier
        # actions list is what is in fringe already
        node, actions, dist = fringe.pop()
        # if node is not already explored
        if node not in explored:
            # if the node contains a goal state then return the corresponding solution
            if problem.isGoal(node):
                # return list of actions
                return actions
            # add the node to the explored set
            explored.add(node)
            #expand the chosen node, adding the resulting nodes to the frontier
            successors = problem.successorStates(node)
            for n, a, d in successors:
                # only if not in the frontier or explored set
                if n not in explored:
                    # add a to the an actions list
                    # ask how they did the actions and new actiosn part
                    newActions = actions + [a]
                    fringe.push((n, newActions, d))
    # return failure when the fringe is empty
    return None
    
def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first. [p 81]
    """

    # *** Your Code Here ***
    # initialize the explored set to be empty
    explored = set()
    # list of actions to take
    # initialize empty
    actions = []
    # initialize the fringe with the start state of problem
    start = problem.startingState()
    # changing it to a queue makes it BFS (FIFO)
    fringe = util2.Queue()
    fringe.push((start, actions, 0))
    # loop
    while not fringe.isEmpty():
        # choose a leaf node and remove it from the frontier
        # actions list is what is in fringe already
        node, actions, dist = fringe.pop()
        # if node is not already explored
        if node not in explored:
            # if the node contains a goal state then return the corresponding solution
            if problem.isGoal(node):
                # return list of actions
                return actions
            # add the node to the explored set
            explored.add(node)
            #expand the chosen node, adding the resulting nodes to the frontier
            successors = problem.successorStates(node)
            for n, a, d in successors:
                # only if not in the frontier or explored set
                if n not in explored:
                    # add a to the an actions list
                    newActions = actions + [a]
                    fringe.push((n, newActions, d))
    # return failure when the fringe is empty
    return None

def uniformCostSearch(problem):
    """
    Search the node of least total cost first.
    """

    # *** Your Code Here ***
    # initialize the explored set to be empty
    explored = set()
    # list of actions to take
    # initialize empty
    actions = []
    # initialize the fringe with the start state of problem
    start = problem.startingState()
    # priority is by total dist/cost
    fringe = util3.PriorityQueue()
    fringe.push((start, actions, 0), 0)
    # loop
    while not fringe.isEmpty():
        # choose a leaf node and remove it from the frontier
        # actions list is what is in fringe already
        node, actions, dist = fringe.pop()
        # if node is not already explored
        if node not in explored:
            # if the node contains a goal state then return the corresponding solution
            if problem.isGoal(node):
                # return list of actions
                return actions
            # add the node to the explored set
            explored.add(node)
            #expand the chosen node, adding the resulting nodes to the frontier
            successors = problem.successorStates(node)
            for n, a, d in successors:
                # only if not in the frontier or explored set
                if n not in explored:
                    # add a to the an actions list
                    newActions = actions + [a]
                    newDist = dist + d
                    # priority is the total dist/cost
                    fringe.push((n, newActions, newDist), newDist)
    # return failure when the fringe is empty
    return None
    # 619

def aStarSearch(problem, heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """

    # *** Your Code Here ***
    # initialize the explored set to be empty
    explored = set()
    # list of actions to take
    # initialize empty
    actions = []
    # initialize the fringe with the start state of problem
    start = problem.startingState()
    # priority is by total dist/cost
    fringe = util3.PriorityQueue()
    fringe.push((start, actions, 0), 0)
    # loop
    while not fringe.isEmpty():
        # choose a leaf node and remove it from the frontier
        # actions list is what is in fringe already
        # chooses the lowest-cost node in frontier 
        node, actions, dist = fringe.pop()
        # if node is not already explored
        if node not in explored:
            # if the node contains a goal state then return the corresponding solution
            if problem.isGoal(node):
                # return list of actions
                return actions
            # add the node to the explored set
            explored.add(node)
            #expand the chosen node, adding the resulting nodes to the frontier
            successors = problem.successorStates(node)
            for n, a, d in successors:
                # only if not in the frontier or explored set
                if n not in explored:
                    # add a to the an actions list
                    newActions = actions + [a]
                    newDist = dist + d
                    # priority is the total dist/cost + heuristic function result passing 
                    # node state and problem
                    fringe.push((n, newActions, newDist), heuristic(n, problem) + newDist)
    # return failure when the fringe is empty
    return None
    # 538 nodes
