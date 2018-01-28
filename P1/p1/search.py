# search.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""
In search.py, you will implement generic search algorithms which are called 
by Pacman agents (in searchAgents.py).
"""

import util
from sets import Set

class SearchProblem:
  """
  This class outlines the structure of a search problem, but doesn't implement
  any of the methods (in object-oriented terminology: an abstract class).
  
  You do not need to change anything in this class, ever.
  """
  
  def startingState(self):
    """
    Returns the start state for the search problem 
    """
    util.raiseNotDefined()

  def isGoal(self, state): #isGoal -> isGoal
    """
    state: Search state

    Returns True if and only if the state is a valid goal state
    """
    util.raiseNotDefined()

  def successorStates(self, state): #successorStates -> successorsOf
    """
    state: Search state
     For a given state, this should return a list of triples, 
     (successor, action, stepCost), where 'successor' is a 
     successor to the current state, 'action' is the action
     required to get there, and 'stepCost' is the incremental 
     cost of expanding to that successor
    """
    util.raiseNotDefined()

  def actionsCost(self, actions): #actionsCost -> actionsCost
    """
      actions: A list of actions to take
 
     This method returns the total cost of a particular sequence of actions.  The sequence must
     be composed of legal moves
    """
    util.raiseNotDefined()
           

def tinyMazeSearch(problem):
  """
  Returns a sequence of moves that solves tinyMaze.  For any other
  maze, the sequence of moves will be incorrect, so only use this for tinyMaze
  """
  from game import Directions
  s = Directions.SOUTH
  w = Directions.WEST
  return  [s,s,w,s,w,w,s,w]

def dfsAlg(problem, state, stack, visited):
  """
  Recursive method invoked by DFS, checks for goal otherwise pushes succerssors and
  calls itself again recursively
  """
  
  #Check for solution  
  if problem.isGoal(state):
    return True

  #Enter all successors reversed
  for successor in reversed(problem.successorStates(state)):
    
    #If the state has not been visited in the past, add to stack and set
    if successor[0] not in visited:
      
      stack.push(successor)
      visited.add(successor[0])

      #Recursive call
      if dfsAlg(problem, successor[0], stack, visited):
        return True
      else:
        stack.pop()

  #No path found
  return False;
  

def depthFirstSearch(problem):
  """
  Search the deepest nodes in the search tree first [p 85].
  
  Your search algorithm needs to return a list of actions that reaches
  the goal.  Make sure to implement a graph search algorithm [Fig. 3.7].
  
  To get started, you might want to try some of these simple commands to
  understand the search problem that is being passed in:
  
  print "Start:", problem.startingState()
  print "Is the start a goal?", problem.isGoal(problem.startingState())
  print "Start's successors:", problem.successorStates(problem.startingState())
  """
  #print "Start:", problem.startingState()
  #print "Is the start a goal?", problem.isGoal(problem.startingState())
  #print "Start's successors:", problem.successorStates(problem.startingState())

  #Initialize ADSs
  solution = []
  visited = Set()
  stack = util.Stack()
  
  #Get an enter start state to stack
  startState = problem.startingState()
  visited.add(startState)

  #If recursive DFS found a solution, extract info from stack
  if dfsAlg(problem, startState, stack, visited):

    while (not stack.isEmpty()):
      nxtDirection = stack.pop()[1]
      solution.append(nxtDirection)

    #Instructions are in reverse order
    solution.reverse()

    return solution
  
  else:
    
    #Solution not found
    return None


def breadthFirstSearch(problem):
  "Search the shallowest nodes in the search tree first. [p 81]"
  
  #Initialize ADSs
  path = []
  solution = []
  visited = Set()
  queue = util.Queue()

  #Get starting state
  startState = problem.startingState()
  visited.add(startState)

  #Dummy start state to be pushed into a path
  dummyStartNode = [(startState), 0, 0]

  #initial path to be pushed to queue
  path.append(dummyStartNode)
  queue.push(path)
  
  while (not queue.isEmpty()):

    #Current solution path
    path = queue.pop()
    nextNode = path[-1]
    nodeState = nextNode[0]

    for successor in problem.successorStates(nodeState):

      #Check for solution when new node is discovered
      if problem.isGoal(successor[0]):
        
        #if solution is found add lates node to the solution path
        path.append(successor)
        #obtain solution list from path  
        for i in range(1,len(path)):
            solution.append(path[i][1])
            
        return solution
      
      else:
        
        #If node has not been visited before
        if successor[0] not in visited:
          
          #add end of current path, add to visited and push path to queue
          visited.add(successor[0])
          newPath = list(path)
          newPath.append(successor)
          queue.push(newPath)
  
  #No solution found
  return None;

def getPathCost(path):

  cost = 0
  
  for i in range(1,len(path)):
    cost += path[i][2]

  return cost
      
def uniformCostSearch(problem):
  "Search the node of least total cost first. "
  
  #Initialize ADSs
  path = []
  solution = []
  visited = Set()
  pQueue = util.PriorityQueue()

  #Get starting state
  startState = problem.startingState()

  #Dummy start state to be pushed into a path
  dummyStartNode = [(startState), 0, 0]

  #initial path to be pushed to queue
  path.append(dummyStartNode)
  pQueue.push(path, 0)
  
  while (not pQueue.isEmpty()):

    #Current solution path
    path = pQueue.pop()
    nextNode = path[-1]
    nodeState = nextNode[0]

    #Check for solution when new node is discovered
    if problem.isGoal(nodeState):
        
      #obtain solution list from path  
      for i in range(1,len(path)):
        solution.append(path[i][1])
            
      return solution

    #Add to explored only after it has been chosen for expansion
    visited.add(nodeState)

    for successor in problem.successorStates(nodeState):
      
        #If node has not been visited before
        if successor[0] not in visited:
          
          #add end of current path, add to visited and push path to queue
          newPath = list(path)
          newPath.append(successor)
          cost = getPathCost(newPath)
          pQueue.push(newPath, cost)

        #elif successor :
        
  #No solution found
  return None;

def nullHeuristic(state, problem=None):
  """
  A heuristic function estimates the cost from the current state to the nearest
  goal in the provided SearchProblem.  This heuristic is trivial.
  """
  return 0

def aStarSearch(problem, heuristic=nullHeuristic):
  "Search the node that has the lowest combined cost and heuristic first."
  #Initialize ADSs
  path = []
  solution = []
  visited = Set()
  pQueue = util.PriorityQueue()

  #Get starting state
  startState = problem.startingState()

  #Dummy start state to be pushed into a path
  dummyStartNode = [(startState), 0, 0]

  #initial path to be pushed to queue
  path.append(dummyStartNode)
  pQueue.push(path, 0)
  
  while (not pQueue.isEmpty()):

    #Current solution path
    path = pQueue.pop()
    nextNode = path[-1]
    nodeState = nextNode[0]

    #Check for solution when new node is discovered
    if problem.isGoal(nodeState):
        
      #obtain solution list from path  
      for i in range(1,len(path)):
        solution.append(path[i][1])
            
      return solution

    #Add to explored only after it has been chosen for expansion
    visited.add(nodeState)

    for successor in problem.successorStates(nodeState):
      
        #If node has not been visited before
        if successor[0] not in visited:
          
          #add end of current path, add to visited and push path to queue
          newPath = list(path)
          newPath.append(successor)
          cost = getPathCost(newPath)
          priority = cost + heuristic(successor[0], problem)
          pQueue.push(newPath, priority)

        #elif successor :
        
  #No solution found
  return None;
    

  
# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch




  
