# qlearningAgents.py
# ------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
  """
    Q-Learning Agent

    Functions you should fill in:
      - getQValue
      - getAction
      - getValue
      - getPolicy
      - update

    Instance variables you have access to
      - self.epsilon (exploration prob)
      - self.alpha (learning rate)
      - self.discountRate (discount rate)

    Functions you should use
      - self.getLegalActions(state)
        which returns legal actions
        for a state
  """
  def __init__(self, **args):
    "You can initialize Q-values here..."
    ReinforcementAgent.__init__(self, **args)

    # Create a containers for q values
    self.q_values = util.Counter()


  def getQValue(self, state, action):
    """
      Returns Q(state,action)
      Should return 0.0 if we never seen
      a state or (state,action) tuple
    """

    """ YOUR CODE HERE """
    # check if we've seen state/action tuple
    if (state, action) in self.q_values:
      return self.q_values[(state, action)]
    else:
      # if tuple never seen before, return 0
      return 0.0
    
    """ END CODE """



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


    """ END CODE """

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

    """ END CODE """

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
    return action


    """ END CODE """

    return action

  def update(self, state, action, nextState, reward):
    """
      The parent class calls this to observe a
      state = action => nextState and reward transition.
      You should do your Q-Value update here

      NOTE: You should never call this function,
      it will be called on your behalf
    """


    """ YOUR CODE HERE """
    # get old q value
    current_q_value = self.getQValue(state, action)
    # calculate sample
    if len(self.getLegalActions(nextState)) == 0:
      sample = reward
    else:
      sample = reward + (self.discountRate * self.getValue(nextState))
    # update equation
    self.q_values[(state, action)] = (1-self.alpha)*current_q_value + self.alpha * sample


    """ END CODE """

class PacmanQAgent(QLearningAgent):
  "Exactly the same as QLearningAgent, but with different default parameters"

  def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
    """
    These default parameters can be changed from the pacman.py command line.
    For example, to change the exploration rate, try:
        python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

    alpha    - learning rate
    epsilon  - exploration rate
    gamma    - discount factor
    numTraining - number of training episodes, i.e. no learning after these many episodes
    """
    args['epsilon'] = epsilon
    args['gamma'] = gamma
    args['alpha'] = alpha
    args['numTraining'] = numTraining
    self.index = 0  # This is always Pacman
    QLearningAgent.__init__(self, **args)

  def getAction(self, state):
    """
    Simply calls the getAction method of QLearningAgent and then
    informs parent of action for Pacman.  Do not change or remove this
    method.
    """
    action = QLearningAgent.getAction(self,state)
    self.doAction(state,action)
    return action


class ApproximateQAgent(PacmanQAgent):
  """
     ApproximateQLearningAgent

     You should only have to overwrite getQValue
     and update.  All other QLearningAgent functions
     should work as is.
  """
  def __init__(self, extractor='IdentityExtractor', **args):
    self.featExtractor = util.lookup(extractor, globals())()
    PacmanQAgent.__init__(self, **args)

    # You might want to initialize weights here.
    self.weights = util.Counter()

  def getQValue(self, state, action):
    """
      Should return Q(state,action) = w * featureVector
      where * is the dotProduct operator
    """
    """ YOUR CODE HERE """
    # initialilze q value to zero
    q_value = 0
    # get all the features for state, action
    features = self.featExtractor.getFeatures(state, action)
    # step through each feature, multiply by appropriate weight
    for feature in features:
      q_value += features[feature] * self.weights[feature]
    return q_value
    """ END CODE """

  def update(self, state, action, nextState, reward):
    """
       Should update your weights based on transition
    """
    """ YOUR CODE HERE """
    # get all the features for state, action
    features = self.featExtractor.getFeatures(state, action)
    # get old q value and calculate new value
    old_q_val = self.getQValue(state, action)
    new_q_val = reward + self.discountRate * self.getValue(nextState)
    for feature in features:
      weight_adjustment = self.alpha * features[feature]*(new_q_val - old_q_val)
      self.weights[feature] = self.weights[feature] + weight_adjustment


    """ END CODE """

  def final(self, state):
    "Called at the end of each game."
    # call the super-class final method
    PacmanQAgent.final(self, state)

    # did we finish training?
    #if self.episodesSoFar == self.numTraining:
      # you might want to print your weights here for debugging
     