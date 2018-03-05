# featureExtractors.py
# --------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"Feature extractors for Pacman game states"

from game import Directions, Actions
import distanceCalculator
import util
from capture import * 

class FeatureExtractor:  
  def getFeatures(self, state, action):    
    """
      Returns a dict from features to counts
      Usually, the count will just be 1.0 for
      indicator functions.  
    """
    util.raiseNotDefined()

class IdentityExtractor(FeatureExtractor):
  def getFeatures(self, state, action):
    feats = util.Counter()
    feats[(state,action)] = 1.0
    return feats

def closestFood(pos, food, walls):
  """
  closestFood -- this is similar to the function that we have
  worked on in the search project; here its all in one place
  """
  fringe = [(pos[0], pos[1], 0)]
  expanded = set()
  while fringe:
    pos_x, pos_y, dist = fringe.pop(0)
    if (pos_x, pos_y) in expanded:
      continue
    expanded.add((pos_x, pos_y))
    # if we find a food at this location then exit
    if food[pos_x][pos_y]:
      return dist
    # otherwise spread out from the location to its neighbours
    nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
    for nbr_x, nbr_y in nbrs:
      fringe.append((nbr_x, nbr_y, dist+1))
  # no food found
  return None

class SimpleExtractor(FeatureExtractor):
  """
  Returns simple features for a basic reflex Pacman:
  - whether food will be eaten
  - how far away the next food is
  - whether a ghost collision is imminent
  - whether a ghost is one step away
  """
  
  def getFeatures(self, state, action, agent):

    features = util.Counter()
    successor = agent.getSuccessor(state, action)

    myPos = successor.getAgentState(agent.index).getPosition()

    

    # extract the grid of food and wall locations and get the ghost locations
    walls = state.getWalls()
    
    #features["bias"] = 1.0

    features['successorScore'] = agent.getScore(successor)

    # Compute distance to the nearest food
    foodList = agent.getFood(successor).asList()
    print "************", foodList
    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(agent.index).getPosition()
      minDistance = min([agent.distancer.getMazeDistances(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance
    
    # Computes distance to defenders we can see
    enemies = [successor.getAgentState(i) for i in agent.getOpponents(successor)]
    defenders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(defenders)
    if len(defenders) > 0:
      dists = [agent.distancer(myPos, a.getPosition()) for a in defenders]
      features['defenderDistance'] = min(dists)

    return features
