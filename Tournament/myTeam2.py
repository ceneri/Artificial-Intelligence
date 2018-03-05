# myTeam.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
#from util import nearestPoint
import util
from qlearningAgents import ApproximateQAgent
from featureExtractors import *
import distanceCalculator

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'DefensiveReflexAgent', second = 'DummyOffensiveAgent', *args, **kwargs):
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
    if pos != util.nearestPoint(pos):
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


class QDummyAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """
  def __init__(self, index, timeForComputing = .1, actionFn = None, alpha=0.2, epsilon=0.05, gamma=0.8, numTraining = 100, extractor='SimpleExtractor' ):
    """
    Sets options, which can be passed in via the Pacman command line using -a alpha=0.5,...
    alpha    - learning rate
    epsilon  - exploration rate
    gamma    - discount factor
    numTraining - number of training episodes, i.e. no learning after these many episodes
    """
    if actionFn == None:
        actionFn = lambda state: state.getLegalActions()
    self.alpha = float(alpha)
    self.epsilon = float(epsilon)
    self.discountRate = float(gamma)
    self.numTraining = int(numTraining)
    self.actionFn = actionFn
    self.episodesSoFar = 0
    self.accumTrainRewards = 0.0
    self.accumTestRewards = 0.0
    self.featExtractor = util.lookup(extractor, globals())()

    # Create a containers for q values
    self.q_values = util.Counter()
    # You might want to initialize weights here.
    self.weights = util.Counter()

    #Capture Agent constructor
    CaptureAgent.__init__(self, index, timeForComputing)

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
    self.distancer = distanceCalculator.Distancer(gameState.data.layout)

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != util.nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor
    
    
  ####################################
  #    Override These Functions      #
  ####################################
  def getQValue(self, state, action):
    """
      Should return Q(state,action) = w * featureVector
      where * is the dotProduct operator
    """
    """ YOUR CODE HERE """
    # initialilze q value to zero
    q_value = 0
    # get all the features for state, action
    features = self.featExtractor.getFeatures(state, action, self)
    # step through each feature, multiply by appropriate weight
    for feature in features:
      q_value += features[feature] #* self.weights[feature]
    return q_value

  def getValue(self, state):
    """
      Returns max_action Q(state,action)
      where the max is over legal actions.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return a value of 0.0.
    """

    """ YOUR CODE HERE """
    # initialize best value to 0.0
    best_value = 0.0
    # loop though all possible actions
    for action in self.getLegalActions(state):
      # check if new value is better than previous best
      if best_value < self.getQValue(state, action):
        best_value = self.getQValue(state, action)
    return best_value

  def getPolicy(self, state):
    """
      Compute the best action to take in a state.  Note that if there
      are no legal actions, which is the case at the terminal state,
      you should return None.
    """

    """ YOUR CODE HERE """
    # initilize best action
    best_policy = None
    best_value = 0.0
    # loop though all possible actions
    for action in self.getLegalActions(state):
      if best_policy == None or self.getQValue(state, action) > best_value:
        best_policy = action
        best_value = self.getQValue(state, action)
    return best_policy

  def getAction(self, state):
    """
      Compute the action to take in the current state.  With
      probability self.epsilon, we should take a random action and
      take the best policy action otherwise.  Note that if there are
      no legal actions, which is the case at the terminal state, you
      should choose None as the action.

      HINT: You might want to use util.flipCoin(prob)
      HINT: To pick randomly from a list, use random.choice(list)
    """
    # Pick Action
    legalActions = self.getLegalActions(state)
    action = None

    """ YOUR CODE HERE """
    # if there are no legal action, return default action (none)
    if not len(legalActions): return action

    # flip coin with prob epsilon
    if util.flipCoin(self.epsilon):
      # make a random choice of action
      action = random.choice(legalActions)
    else:
      action = self.getPolicy(state)
    
    #combined qlearning and pacmanqagent get actions
    self.doAction(state,action)
    return action

  def final(self, state):
    "Called at the end of each game."
    # call the super-class final method
    self.final(self, state)

  def update(self, state, action, nextState, reward):
    """
       Should update your weights based on transition
    """
    """ YOUR CODE HERE """
    # get all the features for state, action
    features = self.featExtractor.getFeatures(state, action, self)
    # get old q value and calculate new value
    old_q_val = self.getQValue(state, action)
    new_q_val = reward + self.discountRate * self.getValue(nextState)
    for feature in features:
      weight_adjustment = self.alpha * features[feature]*(new_q_val - old_q_val)
      self.weights[feature] = self.weights[feature] + weight_adjustment

  def getLegalActions(self,state):
    """
      Get the actions available for a given
      state. This is what you should use to
      obtain legal actions for a state
    """
    return self.actionFn(state)

  def observeTransition(self, state,action,nextState,deltaReward):
    """
      Called by environment to inform agent that a transition has
      been observed. This will result in a call to self.update
      on the same arguments

      NOTE: Do *not* override or call this function
    """
    self.episodeRewards += deltaReward
    self.update(state,action,nextState,deltaReward)

  def startEpisode(self):
    """
      Called by environment when new episode is starting
    """
    self.lastState = None
    self.lastAction = None
    self.episodeRewards = 0.0

  def stopEpisode(self):
    """
      Called by environment when episode is done
    """
    if self.episodesSoFar < self.numTraining:
      self.accumTrainRewards += self.episodeRewards
    else:
      self.accumTestRewards += self.episodeRewards
    self.episodesSoFar += 1
    if self.episodesSoFar >= self.numTraining:
      # Take off the training wheels
      self.epsilon = 0.0    # no exploration
      self.alpha = 0.0      # no learning

  def isInTraining(self):
      return self.episodesSoFar < self.numTraining

  def isInTesting(self):
      return not self.isInTraining()

  ################################
  # Controls needed for Crawler  #
  ################################
  def setEpsilon(self, epsilon):
    self.epsilon = epsilon

  def setLearningRate(self, alpha):
    self.alpha = alpha

  def setDiscount(self, discount):
    self.discountRate = discount

  def doAction(self,state,action):
    """
        Called by inherited class when
        an action is taken in a state
    """
    self.lastState = state
    self.lastAction = action

  ###################
  # Pacman Specific #
  ###################
  def observationFunction(self, state):
    """
        This is where we ended up after our last action.
        The simulation should somehow ensure this is called
    """
    if not self.lastState is None:
        reward = state.getScore() - self.lastState.getScore()
        self.observeTransition(self.lastState, self.lastAction, state, reward)
    return state

  def registerInitialState(self, state):
    self.startEpisode()
    if self.episodesSoFar == 0:
        print 'Beginning %d episodes of Training' % (self.numTraining)

  def final(self, state):
    """
      Called by Pacman game at the terminal state
    """
    deltaReward = state.getScore() - self.lastState.getScore()
    self.observeTransition(self.lastState, self.lastAction, state, deltaReward)
    self.stopEpisode()

    # Make sure we have this var
    if not 'episodeStartTime' in self.__dict__:
        self.episodeStartTime = time.time()
    if not 'lastWindowAccumRewards' in self.__dict__:
        self.lastWindowAccumRewards = 0.0
    self.lastWindowAccumRewards += state.getScore()

    NUM_EPS_UPDATE = 100
    if self.episodesSoFar % NUM_EPS_UPDATE == 0:
        print 'Reinforcement Learning Status:'
        windowAvg = self.lastWindowAccumRewards / float(NUM_EPS_UPDATE)
        if self.episodesSoFar <= self.numTraining:
            trainAvg = self.accumTrainRewards / float(self.episodesSoFar)
            print '\tCompleted %d out of %d training episodes' % (
                   self.episodesSoFar,self.numTraining)
            print '\tAverage Rewards over all training: %.2f' % (
                    trainAvg)
        else:
            testAvg = float(self.accumTestRewards) / (self.episodesSoFar - self.numTraining)
            print '\tCompleted %d test episodes' % (self.episodesSoFar - self.numTraining)
            print '\tAverage Rewards over testing: %.2f' % testAvg
        print '\tAverage Rewards for last %d episodes: %.2f'  % (
                NUM_EPS_UPDATE,windowAvg)
        print '\tEpisode took %.2f seconds' % (time.time() - self.episodeStartTime)
        self.lastWindowAccumRewards = 0.0
        self.episodeStartTime = time.time()

    if self.episodesSoFar == self.numTraining:
        msg = 'Training Done (turning off epsilon and alpha)'
        print '%s\n%s' % (msg,'-' * len(msg))

class ReflexCaptureAgent(CaptureAgent):
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

class OffensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
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

  def getWeights(self, gameState, action):
    return {'successorScore': 100, 'distanceToFood': -1}

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
      features['invaderDistance'] = min(dists)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}
