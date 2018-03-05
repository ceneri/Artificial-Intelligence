# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """


  def getAction(self, gameState):
    """
    You do not need to change this method, but you're welcome to.

    getAction chooses among the best options according to the evaluation function.

    Just like in the previous project, getAction takes a GameState and returns
    some Directions.X for some X in the set {North, South, West, East, Stop}
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    "Add more of your code here if you want to"

    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    Design a better evaluation function here.

    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.

    Print out these variables to see what you're getting, then combine them
    to create a masterful evaluation function.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPosition = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"
    eval = 0

    #Obtain ghost position tuples
    ghostPositions = []
    for ghostState in newGhostStates:
    	ghostPositions.append(ghostState.getPosition())

    #The farther you are from a ghost, the better the position is
    for ghostPos in ghostPositions:
    	eval = eval + manhattanDistance(ghostPos, newPosition)

	#Foods as a list
	food = oldFood.asList()
	foodLeft = len(food)

	#Obtain distance to closest food
	closestFood = 9999
	for foodPos in food:
		closestFood = min (closestFood, manhattanDistance(foodPos, newPosition) )

	#The closest a food item is, the better
	eval = eval - (closestFood*2)
 
    return eval

    #return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
  return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
  """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """

  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.treeDepth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (question 2)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.treeDepth
      and self.evaluationFunction.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
    "*** YOUR CODE HERE ***"

    def maxValue(state, depth, utilityFunction):

        #print depth
        PACMAN_INDEX = 0
        GHOST_INDEX = 1

        #TERMINAL_TEST (First line from books pseudocode)
        if depth == 0 or state.isWin() or state.isLose():
            return utilityFunction(state)

        #Second line from books pseudocode
        utility = -999999

        #Third line from books pseudocode
        legalMoves = state.getLegalActions(PACMAN_INDEX)
        for action in legalMoves:
            successorState = state.generateSuccessor(PACMAN_INDEX, action)

            #Fourth line from books pseudocode
            utility = max(utility, minValue(successorState, GHOST_INDEX, depth-1, utilityFunction) )
            
        return utility

    def minValue(state, agent_index, depth, utilityFunction):

        #TERMINAL_TEST (First line from books pseudocode)
        if depth == 0 or state.isWin() or state.isLose():
            return utilityFunction(state)

        #Second line from books pseudocode
        utility = 999999

        #Third line from books pseudocode
        legalMoves = state.getLegalActions(agent_index)

        #Generalize alg to handle multi ghost

        #Next one is pacman/MAX (Number of ghosts)
        if agent_index == (state.getNumAgents() - 1):

            for action in legalMoves:
                successorState = state.generateSuccessor(agent_index, action)
                
                #Fourth line from books pseudocode
                utility = min(utility, maxValue(successorState, depth-1, utilityFunction) )
        
        else:

            for action in legalMoves:
                successorState = state.generateSuccessor(agent_index, action)

                #Fourth line from books pseudocode
                utility = min(utility, minValue(successorState, agent_index+1, depth, utilityFunction) )


        return utility

    def minimaxDecision(state, treeDepth, utilityFunction):

        #Agent index
        PACMAN_INDEX = 0

        # Collect legal moves and successor states
        legalMoves = state.getLegalActions()
        #legalMoves.remove(Directions.STOP)

        #First line from books pseudocode
        argmax = Directions.STOP
        max_u = -999999
        for action in legalMoves:
            successorState = state.generateSuccessor(PACMAN_INDEX, action)
            utility = minValue(successorState, 1, treeDepth, utilityFunction)

            if utility > max_u:
               max_u = utility
               argmax = action

        return argmax

    #endof minimaxDecision

    #Variable declaration
    treeDepth = self.treeDepth
    utilityFunction = self.evaluationFunction

    return minimaxDecision(gameState, treeDepth, utilityFunction)

    #util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.treeDepth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"

    def ABmaxValue(state, alpha, beta, depth, utilityFunction):

        #print depth
        PACMAN_INDEX = 0
        GHOST_INDEX = state.getNumAgents() - 1

        #TERMINAL_TEST (First line from books pseudocode)
        if depth == 0 or state.isWin() or state.isLose():
            return utilityFunction(state)

        #Second line from books pseudocode
        utility = -999999

        #Third line from books pseudocode
        legalMoves = state.getLegalActions(PACMAN_INDEX)
        for action in legalMoves:
            successorState = state.generateSuccessor(PACMAN_INDEX, action)

            #Fourth line from books pseudocode
            utility = max(utility, ABminValue(successorState, alpha, beta, GHOST_INDEX, depth, utilityFunction) )         

            #check for prunning
            if utility >= beta:
                break
            alpha = max(alpha, utility)

        return utility

    def ABminValue(state, alpha, beta, agent_index, depth, utilityFunction):

        #TERMINAL_TEST (First line from books pseudocode)
        if depth == 0 or state.isWin() or state.isLose():
            return utilityFunction(state)

        #Second line from books pseudocode
        utility = 999999

        #Third line from books pseudocode
        legalMoves = state.getLegalActions(agent_index)

        #Generalize alg to handle multi ghost

        #Next one is pacman/MAX (Number of ghosts)
        if agent_index == (state.getNumAgents() - 1):

            for action in legalMoves:
                successorState = state.generateSuccessor(agent_index, action)
                
                #Fourth line from books pseudocode
                utility = min(utility, ABmaxValue(successorState, alpha, beta, depth-1, utilityFunction) )
                
                #Stop and prune
                if utility <= alpha:
                    break
                beta = min(beta, utility)

        else:

            for action in legalMoves:
                successorState = state.generateSuccessor(agent_index, action)

                #Fourth line from books pseudocode
                utility = min(utility, ABminValue(successorState, alpha, beta, agent_index+1, depth, utilityFunction) )
                #Stop and prune
                if utility <= alpha:
                    break
                beta = min(beta, utility)


        return utility

    def AlphaBetaSearch(state, treeDepth, utilityFunction):

        alpha = -float("inf")
        beta = float("inf")

        #Agent index
        PACMAN_INDEX = 0

        # Collect legal moves and successor states
        legalMoves = state.getLegalActions()
        #legalMoves.remove(Directions.STOP)

        #First line from books pseudocode
        argmax = Directions.STOP
        max_u = -999999
        for action in legalMoves:
            successorState = state.generateSuccessor(PACMAN_INDEX, action)
            utility = ABminValue(successorState, alpha, beta, 1, treeDepth, utilityFunction)

            if utility > max_u:
               max_u = utility
               argmax = action

            #Stop and prune
            if utility >= beta:
                argmax = action
                break

            alpha = max(alpha, utility)

        return argmax

    #endof AlphaBetaSearch

    #Variable declaration
    treeDepth = self.treeDepth
    utilityFunction = self.evaluationFunction

    return AlphaBetaSearch(gameState, treeDepth, utilityFunction)

    #util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (question 4)
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.treeDepth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
    "*** YOUR CODE HERE ***"

    def eMaxValue(state, depth, utilityFunction):

        #print depth
        PACMAN_INDEX = 0
        GHOST_INDEX = 1

        #TERMINAL_TEST (First line from books pseudocode)
        if depth == 0 or state.isWin() or state.isLose():
            return utilityFunction(state)

        #Second line from books pseudocode
        utility = -999999

        #Third line from books pseudocode
        legalMoves = state.getLegalActions(PACMAN_INDEX)
        for action in legalMoves:
            successorState = state.generateSuccessor(PACMAN_INDEX, action)

            #Fourth line from books pseudocode
            utility = max(utility, eMinValue(successorState, GHOST_INDEX, depth, utilityFunction) )
            
        return utility

    def eMinValue(state, agent_index, depth, utilityFunction):

        #TERMINAL_TEST (First line from books pseudocode)
        if depth == 0 or state.isWin() or state.isLose():
            return utilityFunction(state)

        #Second line from books pseudocode
        expectedUtility = 0

        #Third line from books pseudocode
        legalMoves = state.getLegalActions(agent_index)

        #Generalize alg to handle multi ghost

        #Next one is pacman/MAX (Number of ghosts)
        if agent_index == (state.getNumAgents() - 1):

            for action in legalMoves:
                successorState = state.generateSuccessor(agent_index, action)
                
                #Fourth line from books pseudocode
                expectedUtility = expectedUtility + eMaxValue(successorState, depth-1, utilityFunction)
        
        else:

            for action in legalMoves:
                successorState = state.generateSuccessor(agent_index, action)

                #Fourth line from books pseudocode
                expectedUtility = expectedUtility + eMinValue(successorState, agent_index+1, depth, utilityFunction)


        #Each action has equal chance of taking place
        expectedUtility = expectedUtility / len (legalMoves)
        return expectedUtility

    def expectimax(state, treeDepth, utilityFunction):

        #Agent index
        PACMAN_INDEX = 0

        # Collect legal moves and successor states
        legalMoves = state.getLegalActions()
        #legalMoves.remove(Directions.STOP)

        #First line from books pseudocode
        argmax = Directions.STOP
        max_u = -999999
        for action in legalMoves:
            successorState = state.generateSuccessor(PACMAN_INDEX, action)
            utility = eMinValue(successorState, 1, treeDepth, utilityFunction)

            if utility > max_u:
               max_u = utility
               argmax = action

        return argmax

    #endof expectimax

    #Variable declaration
    treeDepth = self.treeDepth
    utilityFunction = self.evaluationFunction

    return expectimax(gameState, treeDepth, utilityFunction)

    #util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
        Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
        evaluation function (question 5).

        DESCRIPTION: <write something here so we know what you did>

        The evaluation is similar to the original evaluation with additional tweaks to hopefully
        have a better end result

        The first cosntraint is checking for win and loose states. If a win state is found
        a large value is returned, while the opposite is done with a loose state

        Otherwise the distance from ghosts is calculated as a basis for survival. To promote 
        Pacman to eat while ghost are in opposite side of layout, when a ghost is far away
        an extra bonus value of 100 is added to the evaluation. I tried different multipliers, 
        and 10 seems to be adequate to keep survival as a high priority

        The next important value to consider is distance from food. DIstances to all foods are 
        substracted to promote being as close to food a spossible.

        Finally we also give extra bonus of 100 to positions that conmtain a capsule
        hoping to give said position an extra incentive to be chosen

        I also tried to promote moving closer towards scared ghosts but implementing that
        was quite problematic due to the object type of the ghosts and the provided 
        scaredtimes. Plus no notisable benefit was seen when implemented 
    """

    newPosition = currentGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    capsules = currentGameState.getCapsules()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    eval = 0

    "*** YOUR CODE HERE ***"

    if currentGameState.isWin():
        return 999999
    elif currentGameState.isLose():
        return -999999
    else:

        #Obtain ghost position tuples
        ghostPositions = []
        for ghostState in newGhostStates:
            ghostPositions.append(ghostState.getPosition())

        #The farther you are from a ghost, the better the position is
        for ghostPos in ghostPositions:
            if manhattanDistance(ghostPos, newPosition) > 8:
                eval = eval + 100
                continue
            eval = eval + manhattanDistance(ghostPos, newPosition)

        #Survival is very important
        eval = eval * 10

        #Foods as a list
        food = oldFood.asList()
        foodLeft = len(food)

        #The farther you are from a food, the worst the position is
        for foodPos in food:
            eval = eval - manhattanDistance(foodPos, newPosition)


        for capsulePos in capsules:
            if capsulePos == newPosition:
                eval = eval + 100

        return eval

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """

  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()
