import util
from sets import Set
import search

from game import Directions
from game import Agent
from game import Actions
import util
import time
import search
import searchAgents

def mazeDistance(point1, point2, gameState):
  """
  Returns the maze distance between any two points, using the search functions
  you have already built.  The gameState can be any game state -- Pacman's position
  in that state is ignored.
  
  Example usage: mazeDistance( (2,4), (5,6), gameState)
  
  This might be a useful helper function for your ApproximateSearchAgent.
  """
  x1, y1 = point1
  x2, y2 = point2
  walls = gameState.getWalls()
  assert not walls[x1][y1], 'point1 is a wall: ' + point1
  assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
  prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False)


  return len(search.bfs(prob))

class PositionSearchProblem(search.SearchProblem):
  """
  A search problem defines the state space, start state, goal test,
  successor function and cost function.  This search problem can be 
  used to find paths to a particular point on the pacman board.
  
  The state space consists of (x,y) positions in a pacman game.
  
  Note: this search problem is fully specified; you should NOT change it.
  """
  
  def __init__(self, gameState, costFn = lambda x: 1, goal=(1,1), start=None, warn=True):
    """
    Stores the start and goal.  
    
    gameState: A GameState object (pacman.py)
    costFn: A function from a search state (tuple) to a non-negative number
    goal: A position in the gameState
    """
    self.walls = gameState.getWalls()
    self.startState = gameState.getPacmanPosition()
    if start != None: self.startState = start
    self.goal = goal
    self.costFn = costFn
    if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
      print 'Warning: this does not look like a regular search maze'

    # For display purposes
    self._visited, self._visitedlist, self._expanded = {}, [], 0

  def startingState(self):
    return self.startState

  def isGoal(self, state):
    isGoal = state == self.goal 
     
    # For display purposes only
    if isGoal:
      self._visitedlist.append(state)
      import __main__
      if '_display' in dir(__main__):
        if 'drawExpandedCells' in dir(__main__._display): #@UndefinedVariable
          __main__._display.drawExpandedCells(self._visitedlist) #@UndefinedVariable
       
    return isGoal  
   
  def successorStates(self, state):
    """
    Returns successor states, the actions they require, and a cost of 1.
    
     As noted in search.py:
         For a given state, this should return a list of triples, 
     (successor, action, stepCost), where 'successor' is a 
     successor to the current state, 'action' is the action
     required to get there, and 'stepCost' is the incremental 
     cost of expanding to that successor
    """
    
    successors = []
    for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
      x,y = state
      dx, dy = Actions.directionToVector(action)
      nextx, nexty = int(x + dx), int(y + dy)
      if not self.walls[nextx][nexty]:
        nextState = (nextx, nexty)
        cost = self.costFn(nextState)
        successors.append( ( nextState, action, cost) )
        
    # Bookkeeping for display purposes
    self._expanded += 1 
    if state not in self._visited:
      self._visited[state] = True
      self._visitedlist.append(state)
      
    return successors

  def actionsCost(self, actions):
    """
    Returns the cost of a particular sequence of actions.  If those actions
    include an illegal move, return 999999
    """
    if actions == None: return 999999
    x,y= self.startingState()
    cost = 0
    for action in actions:
      # Check figure out the next state and see whether its' legal
      dx, dy = Actions.directionToVector(action)
      x, y = int(x + dx), int(y + dy)
      if self.walls[x][y]: return 999999
      cost += self.costFn((x,y))
    return cost
