# myTeam.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from captureAgents import CaptureAgent
import random, time, util
import collections
from game import Directions, Actions
import game
import distanceCalculator
from util import nearestPoint


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'DefensiveCampingAgent', second = 'DummyOffensiveAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########



class DummyAgent(CaptureAgent):
    """
    A Dummy agent to serve as an example of the necessary agent structure.
    You should look at baselineTeam.py for more details about how to
    create an agent as this is the bare minimum.
    """

    # ### The below code is going to be called by the game ### #

    def __init__(self, *args):
        '''
        Initialize agent
        '''

        CaptureAgent.__init__(self, *args)

    def registerInitialState(self, gameState):
        """
        This method handles the initial setup of the
        agent to populate useful fields (such as what team
        we're on).

        A distanceCalculator instance caches the maze distances
        between each pair of positions, so your agents can use:
        self.distancer.getDistance(p1, p2)

        IMPORTANT: This method may run for at most 15 seconds.
        """

        '''
        Make sure you do not delete the following line. If you would like to
        use Manhattan distances instead of maze distances in order to save
        on initialization time, please take a look at
        CaptureAgent.registerInitialState in captureAgents.py.
        '''
        CaptureAgent.registerInitialState(self, gameState)
        
    def isGhost(self, gameState, index):
        """
        Returns true if agent can be seen and is a ghost.
        """
        position = gameState.getAgentPosition(index)
        if position is None:
            return False
        
        width_of_map = gameState.getWalls().width
        red_zone_border = width_of_map / 2
        on_red_side = False
        
        if position[0] < red_zone_border:
        	on_red_side = True
        is_red_team = gameState.isOnRedTeam(index)

        return not (is_red_team ^ on_red_side)

    def isScared(self, gameState, index):
        """
        Says whether or not the given agent is scared
        """
        is_scared = bool(gameState.data.agentStates[index].scaredTimer)
        return is_scared


    def isPacman(self, gameState, index):
        """
        Returns true if agent can be seen and is a pacman.
        """
        position = gameState.getAgentPosition(index)
        if position is None:
            return False
        
        width_of_map = gameState.getWalls().width
        red_zone_border = width_of_map / 2
        on_red_side = False
        
        if position[0] < red_zone_border:
        	on_red_side = True
        is_red_team = gameState.isOnRedTeam(index)

        return (is_red_team ^ on_red_side)

    
    

    # ## A Star Search ## #

    def aStarSearch(self, startPosition, gameState, goalPositions, avoidPositions=[], returnPosition=False):
        """
        Finds the distance between the agent with the given index and its nearest goalPosition
        """
        walls = gameState.getWalls()
        width = walls.width
        height = walls.height
        walls = walls.asList()

        actions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]
        actionVectors = [Actions.directionToVector(action) for action in actions]
        # Change action vectors to integers so they work correctly with indexing
        actionVectors = [tuple(int(number) for number in vector) for vector in actionVectors]

        # Values are stored a 3-tuples, (Position, Path, TotalCost)

        currentPosition, currentPath, currentTotal = startPosition, [], 0
        # Priority queue uses the maze distance between the entered point and its closest goal position to decide which comes first
        queue = util.PriorityQueueWithFunction(lambda entry: entry[2] +   # Total cost so far
                                               width * height if entry[0] in avoidPositions else 0 +  # Avoid enemy locations like the plague
                                               min(self.distancer.getDistance(entry[0], endPosition) for endPosition in goalPositions)) #util.manhattanDistance

        # Keeps track of visited positions
        visited = set([currentPosition])

        while currentPosition not in goalPositions:

            possiblePositions = [((currentPosition[0] + vector[0], currentPosition[1] + vector[1]), action) for vector, action in zip(actionVectors, actions)]
            legalPositions = [(position, action) for position, action in possiblePositions if position not in walls]

            for position, action in legalPositions:
                if position not in visited:
                    visited.add(position)
                    queue.push((position, currentPath + [action], currentTotal + 1))

            # This shouldn't ever happen...But just in case...
            if len(queue.heap) == 0:
                return None
            else:
                currentPosition, currentPath, currentTotal = queue.pop()

        if returnPosition:
            return currentPath, currentPosition
        else:
            return currentPath

    def positionIsHome(self, position, gameWidth):
        isHome = not (self.red ^ (position[0] < gameWidth / 2))
        return isHome

    def getFlowNetwork(self, gameState, startingPositions=None, endingPositions=None, defenseOnly=True):
        '''
        Returns the flow network.
        If starting positions are provided, also returns the source node
        If ending positions are provided, also returns the sink node
        Note: Always returns tuple
        '''
        source = (-1, -1)
        sink = (-2, -2)

        walls = gameState.getWalls()
        wallPositions = walls.asList()
        possiblePositions = [(x, y) for x in range(walls.width) for y in range(walls.height) if (x, y) not in wallPositions and (not defenseOnly or self.positionIsHome((x, y), walls.width))]

        actions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]
        actionVectors = [Actions.directionToVector(action) for action in actions]
        # Change vectors from float to int
        actionVectors = [tuple(int(number) for number in vector) for vector in actionVectors]

        # Make source and sink

        network = FlowNetwork()

        # Add all vertices
        for position in possiblePositions:
            network.AddVertex(position)
        network.AddVertex(source)
        network.AddVertex(sink)

        # Add normal edges
        edges = EdgeDict()
        for position in possiblePositions:
            for vector in actionVectors:
                newPosition = (position[0] + vector[0], position[1] + vector[1])
                if newPosition in possiblePositions:
                    edges[(position, newPosition)] = 1

        # Add edges attached to source
        for position in startingPositions or []:
            edges[(source, position)] = float('inf')

        for position in endingPositions or []:
            edges[(position, sink)] = float('inf')

        for edge in edges:
            network.AddEdge(edge[0], edge[1], edges[edge])

        retval = (network,)

        if startingPositions is not None:
            retval = retval + (source,)
        if endingPositions is not None:
            retval = tuple(retval) + (sink,)

        return retval

    def findBottleneckWithMostPacdots(self, gameState):

        startingPositions = self.getMiddlePositions(gameState)
        endingPositions = self.getFoodYouAreDefending(gameState).asList()
        network, source = self.getFlowNetwork(gameState, startingPositions=startingPositions)

        bottleneckCounter = collections.Counter()

        for dot in endingPositions:
            bottlenecks = network.FindBottlenecks(source, dot)
            if len(bottlenecks) == 1:
                bottleneckCounter[bottlenecks[0]] += 1
            network.reset()

        maxBottleneck = max(bottleneckCounter or [None], key=lambda vertex: bottleneckCounter[vertex])
        return maxBottleneck, bottleneckCounter[maxBottleneck]

    def getMiddlePositions(self, gameState):

        # Find the positions closest to the middle line so we can start there
        walls = gameState.getWalls()
        wallPositions = walls.asList()
        possiblePositions = [(x, y) for x in range(walls.width) for y in range(walls.height) if (x, y) not in wallPositions and self.positionIsHome((x, y), walls.width)]
        startX = walls.width / 2 - 1 if self.red else walls.width / 2
        startingPositions = [position for position in possiblePositions if position[0] == startX]
        return startingPositions


"""
Ford-Fulkerson algorithm, taken from third link on google: https://github.com/bigbighd604/Python/blob/master/graph/Ford-Fulkerson.py
"""

class Edge(object):
    def __init__(self, u, v, w):
        self.source = u
        self.target = v
        self.capacity = w

    def __repr__(self):
        return "%s->%s:%s" % (self.source, self.target, self.capacity)

    def __eq__(self, other):
        return self.source == other.source and self.target == other.target


class FlowNetwork(object):
    def __init__(self):
        self.adj = {}
        self.flow = {}

    def AddVertex(self, vertex):
        self.adj[vertex] = []

    def GetEdges(self, v):
        return self.adj[v]

    def AddEdge(self, u, v, w=0):
        if u == v:
            raise ValueError("u == v")
        edge = Edge(u, v, w)
        redge = Edge(v, u, w)
        edge.redge = redge
        redge.redge = edge
        self.adj[u].append(edge)
        self.adj[v].append(redge)
        # Intialize all flows to zero
        self.flow[edge] = 0
        self.flow[redge] = 0

    def FindPath(self, source, target):

        currentVertex, currentPath, currentTotal = source, [], 0
        # Priority queue uses the maze distance between the entered point and its closest goal position to decide which comes first
        queue = util.PriorityQueueWithFunction(lambda entry: entry[2] + util.manhattanDistance(entry[0], target))

        visited = set()

        # Keeps track of visited positions
        while currentVertex != target:

            possibleVertices = [(edge.target, edge) for edge in self.GetEdges(currentVertex)]

            for vertex, edge in possibleVertices:
                residual = edge.capacity - self.flow[edge]
                if residual > 0 and not (edge, residual) in currentPath and (edge, residual) not in visited:
                    visited.add((edge, residual))
                    queue.push((vertex, currentPath + [(edge, residual)], currentTotal + 1))

            if queue.isEmpty():
                return None
            else:
                currentVertex, currentPath, currentTotal = queue.pop()

        return currentPath

    def FindBottlenecks(self, source, target):
        maxflow, leadingEdges = self.MaxFlow(source, target)
        paths = leadingEdges.values()

        bottlenecks = []
        for path in paths:
            for edge, residual in path:
                # Save the flows so we don't mess up the operation between path findings
                if self.FindPath(source, edge.target) is None:
                    bottlenecks.append(edge.source)
                    break
        assert len(bottlenecks) == maxflow
        return bottlenecks

    def MaxFlow(self, source, target):
        # This keeps track of paths that go to our destination
        leadingEdges = {}
        path = self.FindPath(source, target)
        while path:
            leadingEdges[path[0]] = path
            flow = min(res for edge, res in path)
            for edge, res in path:
                self.flow[edge] += flow
                self.flow[edge.redge] -= flow

            path = self.FindPath(source, target)
        maxflow = sum([self.flow[edge] for edge in self.GetEdges(source)])
        return maxflow, leadingEdges

    def reset(self):
        for edge in self.flow:
            self.flow[edge] = 0


class EdgeDict(dict):
    '''
    Keeps a list of undirected edges. Doesn't matter what order you add them in.
    '''
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)

    def __getitem__(self, key):
        return dict.__getitem__(self, tuple(sorted(key)))

    def __setitem__(self, key, val):
        return dict.__setitem__(self, tuple(sorted(key)), val)

    def __contains__(self, key):
        return dict.__contains__(self, tuple(sorted(key)))

    def getAdjacentPositions(self, key):
        edgesContainingKey = [edge for edge in self if key in edge]
        adjacentPositions = [[position for position in edge if position != key][0] for edge in edgesContainingKey]
        return adjacentPositions




class ReflexCaptureAgent(DummyAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """
  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    return random.choice(bestActions)

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}

class DummyOffensiveAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).
    
    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    CaptureAgent.registerInitialState(self, gameState)

    '''
    Your initialization code goes here, if you need any.
    '''

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    return random.choice(bestActions)

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)

    # Compute distance to the nearest food
    foodList = self.getFood(successor).asList()
    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance
    return features

    # Computes distance to defenders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    defenders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numDefenders'] = len(defenders)
    if len(defenders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in defenders]
      features['defenderDistance'] = min(dists)

  def getWeights(self, gameState, action):
    return {'successorScore': 100, 'distanceToFood': -1, 'numDefenders': -1000, 'defenderDistance': -10}




class DefensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
         
      if self.isScared(gameState, self.index):
      	features['invaderDistance'] = min(dists) * -1
      else:
		features['invaderDistance'] = min(dists)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 200, 'invaderDistance': -100, 'stop': -50, 'reverse': -2}







class DefensiveCampingAgent(DefensiveReflexAgent):
    def __init__(self, *args):
        #self.defenseMode = False
        self.defenseMode = True	
        self.GoToSpot = None
        DummyAgent.__init__(self, *args)

    def registerInitialState(self, gameState):

        DummyAgent.registerInitialState(self, gameState)
        self.checkForBottleneck(gameState)

    def chooseAction(self, gameState):
        # If we were scared and aren't anymore, re-check for bottleneck
        if self.getPreviousObservation():
            if self.isScared(self.getPreviousObservation(), self.index) and not self.isScared(gameState, self.index):
                self.checkForBottleneck(gameState)

        if self.defenseMode and not self.isScared(gameState, self.index):
            position = gameState.getAgentPosition(self.index)
            opponentPositions = [gameState.getAgentPosition(i) for i in self.getOpponents(gameState) if self.isPacman(gameState, i)]
            if opponentPositions:
                pathToOpponents = self.aStarSearch(position, gameState, opponentPositions)
                if (len(pathToOpponents) % 2 == 1 and not  # We want the path length to be odd
                   (len(self.aStarSearch(self.GoToSpot, gameState, [position])) < len(self.aStarSearch(self.GoToSpot, gameState, opponentPositions)))):  # We want to be closer to our spot than they are
                    return pathToOpponents[0]

            pathToSpot = self.aStarSearch(position, gameState, opponentPositions or [self.GoToSpot]) or [Directions.STOP]
            # Paths an odd distance away have a better chance of working
            return pathToSpot[0]
        return DefensiveReflexAgent.chooseAction(self, gameState)

    def checkForBottleneck(self, gameState):
        bottleneckPosition, numDots = self.findBottleneckWithMostPacdots(gameState)
        if numDots >=2:
            self.defenseMode = True
            self.GoToSpot = bottleneckPosition
        else:
            self.defenseMode = False
            self.goToSpot = None