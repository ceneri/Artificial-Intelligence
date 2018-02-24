# valueIterationAgents.py
# -----------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
  """
      * Please read learningAgents.py before reading this.*

      A ValueIterationAgent takes a Markov decision process
      (see mdp.py) on initialization and runs value iteration
      for a given number of iterations using the supplied
      discount factor.
  """
  def __init__(self, mdp, discountRate = 0.9, iters = 100):
    """
      Your value iteration agent should take an mdp on
      construction, run the indicated number of iterations
      and then act according to the resulting policy.

      Some useful mdp methods you will use:
          mdp.getStates()
          mdp.getPossibleActions(state)
          mdp.getTransitionStatesAndProbs(state, action)
          mdp.getReward(state, action, nextState)
    """
    self.mdp = mdp
    self.discountRate = discountRate
    self.iters = iters
    self.values = util.Counter() # A Counter is a dict with default 0

    """Description:
    For every state, we calculate "the best possible outcome", we do so by looking at every 
    action possible from each state, and calculate its q value i.e. the best we can do after 
    that action from current state and acting optimally thereafter

    We do this for every state, multiple times. Specifically the specified number of 'iters'

    Put it simply, V*(s) = max Q*()
    """
    """ YOUR CODE HERE """

    states = mdp.getStates() 

    #print states    

    #Do for all iterations
    for n in range(iters):

      #Per piazza post, make a copy of Counter
      updatedValues = self.values.copy()

      #Fo every state see whats the best you can do (value)
      for state in states:

        value = None

        if mdp.isTerminal(state):
          value = 0

        else:

          actions = mdp.getPossibleActions(state)

          #Do so by calculating q values of each action
          for action in actions:

            q_value = self.getQValue(state, action)

            if q_value > value:

              value = q_value

          #if value == None:
           # value = 0

          updatedValues[state] = value

      #Update values
      self.values = updatedValues

    #util.raiseNotDefined()
    """ END CODE """

  def getValue(self, state):
    """
      Return the value of the state (computed in __init__).
    """
    return self.values[state]

    """Description:
    [Enter a description of what you did here.]
    """
    """ YOUR CODE HERE """
    util.raiseNotDefined()
    """ END CODE """

  def getQValue(self, state, action):
    """
      The q-value of the state action pair
      (after the indicated number of value iteration
      passes).  Note that value iteration does not
      necessarily create this quantity and you may have
      to derive it on the fly.
    """
    """Description:
    [Enter a description of what you did here.]
    """
    """ YOUR CODE HERE """
    q_val = 0

    mdp = self.mdp
    discountRate = self.discountRate
    values = self.values

    stateAndProbs = mdp.getTransitionStatesAndProbs(state, action)
    num_tStates = len(stateAndProbs)

    for i in range(num_tStates):

      i_state = stateAndProbs[i][0]
      i_prob = stateAndProbs[i][1]

      i_reward = mdp.getReward(state, action, i_state)

      future_val = discountRate * values[i_state]

      i_val = i_prob * (i_reward + future_val)

      q_val += i_val

    
    return q_val
    #for index in range(len(stateAndProbs)):
     # probability 


    #util.raiseNotDefined()
    """ END CODE """

  def getPolicy(self, state):
    """
      The policy is the best action in the given state
      according to the values computed by value iteration.
      You may break ties any way you see fit.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return None.
    """

    """Description:
    The best possible action should be the one with the biggest Q*value, unless you are in a terminal state
    then you shoulds return None
    """
    """ YOUR CODE HERE """

    mdp = self.mdp

    #there are no legal actions
    if mdp.isTerminal(state):
      return None

    actions = mdp.getPossibleActions(state)

    best_action = None
    value = -999999

    for action in actions:

      q_val = self.getQValue(state, action)

      if q_val > value:
        value = q_val
        best_action = action

    return best_action

    #util.raiseNotDefined()
    """ END CODE """

  def getAction(self, state):
    "Returns the policy at the state (no exploration)."
    return self.getPolicy(state)
