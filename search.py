import util


class SearchProblem:
  
    def getStartState(self):
      
        util.raiseNotDefined()

    def isGoalState(self, state):
       
        util.raiseNotDefined()

    def getSuccessors(self, state):
        
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
       
        util.raiseNotDefined()


def tinyMazeSearch(problem):

    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):

    visited = set()
    stateStack = util.Stack()
    stateStack.push((problem.getStartState(), [], 0))

    while not stateStack.isEmpty():
        cState, cActions, cCost = stateStack.pop()

        if (problem.isGoalState(cState)):
            return cActions

        if (cState in visited):
            continue

        visited.add(cState)
        for successor in problem.getSuccessors(cState):
            sState, nextAction, nextCost = successor

            sActions = cActions + [nextAction]
            sCost = cCost + nextCost

            stateStack.push((sState, sActions, sCost))

    return []


def breadthFirstSearch(problem):

    visited = set()
    stateStack = util.Queue()
    stateStack.push((problem.getStartState(), [], 0))

    while not stateStack.isEmpty():
        cState, cActions, cCost = stateStack.pop()

        if (problem.isGoalState(cState)):
            return cActions

        if (cState in visited):
            continue

        visited.add(cState)
        for successor in problem.getSuccessors(cState):
            sState, nextAction, nextCost = successor

            sActions = cActions + [nextAction]
            sCost = cCost + nextCost

            stateStack.push((sState, sActions, sCost))

    return []


def uniformCostSearch(problem):
    
    visited = set()
    statePQ = util.PriorityQueue()
    statePQ.push((problem.getStartState(), [], 0), 0)

    while not statePQ.isEmpty():
        cState, cActions, cCost = statePQ.pop()

        if (problem.isGoalState(cState)):
            return cActions

        if (cState in visited):
            continue

        visited.add(cState)
        for successor in problem.getSuccessors(cState):
            sState, nextAction, nextCost = successor

            sActions = cActions + [nextAction]
            sCost = cCost + nextCost

            statePQ.push((sState, sActions, sCost), sCost)

    return []


def nullHeuristic(state, problem=None):
    
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    visited = set()
    statePQ = util.PriorityQueue()
    heuristicValue = heuristic(problem.getStartState(), problem)
    statePQ.push((problem.getStartState(), [], 0),heuristicValue)

    while not statePQ.isEmpty():
        cState, cActions, cCost = statePQ.pop()

        if (problem.isGoalState(cState)):
            return cActions

        if (cState in visited):
            continue

        visited.add(cState)
        for successor in problem.getSuccessors(cState):
            sState, nextAction, nextCost = successor

            sActions = cActions + [nextAction]
            sCost = cCost + nextCost

            heuristicValue = heuristic(sState, problem) + sCost
            statePQ.push((sState, sActions, sCost),
                         heuristicValue)

    return []


bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
