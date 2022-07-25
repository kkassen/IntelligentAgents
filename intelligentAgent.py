# Author: Kyle Arick Kassen

# Import statements
from collections import deque
import heapq
import time

class Node():
    # Initialization
    def __init__(self, state, parent=None, action=None, cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.cost = cost

    # Node expansion: what's the next step(s)?
    def expand(self, problem):
        paths = []
        for action in problem.options(self.state):
            current = problem.transition(self.state, action)
            next = Node(current, self, action, problem.tileCost(self.cost, current, action, self.state))
            paths.append(next)
        return paths

    # Returns the path of moves performed by the agent
    def movesPath(self, node):
        path = []
        for node in self.trace(node)[:]:
            path.append(node.action)
        return path

    # Returns the g(n) cost path
    def costPath_gn(self, node):
        path = []
        for node in self.trace(node)[:]:
            path.append(node.cost)
        return path

    # Returns the h1(n) cost path
    def costPath_h1(self, node):
        path = []
        total = 0
        for node in self.trace(node)[:]:
            total += problem.h1(node)
            path.append(total)
        return path

    # Returns the cost path for f(n) = g(n) + h1(n) for A* Search I
    # Where g(n) = path cost and h1(n) = No. of misplaced tiles relative to the goal
    def costPath_a1(self, node):
        path = []
        for node in self.trace(node)[:]:
            path.append(node.cost + problem.h1(node))
        return path

    # Returns the cost path for f(n) = g(n) + h2(n) for A* Search II
    # Where g(n) = path cost and h2(n) = Manhattan Distance
    def costPath_a2(self, node):
        path = []
        for node in self.trace(node)[:]:
            path.append(node.cost + problem.h2(node))
        return path

    # Returns the path traveled by the agent from start to goal
    def trace(self, node):
        path = []
        while node is not None:
            path.append(node)
            node = node.parent
        return path[::-1]

    # For viewing the Node Objects in the output terminal
    def __repr__(self):
        return '{}-->'.format(self.state)

    # Allows for Node Object comparison in Priority Queue
    def __lt__(self, node):
        return self.state < node.state

class PriorityQueue():
    # Initialization
    def __init__(self, nodes=(), fn=lambda n: n):
        self.fn = fn
        self.nodes = []
        for node in nodes:
            self.add(node)

    # Adds an item to the Priority Queue
    def add(self, node):
        heapq.heappush(self.nodes, (self.fn(node), node))

    # Removes an item from the Priority Queue
    def pop(self):
        return heapq.heappop(self.nodes)[1]

    # Allows for use of the len() function for Priority Queue Objects
    def __len__(self):
        return len(self.nodes)

    # For viewing Priority Queue Objects in the output terminal
    def __repr__(self):
        return "{}-->".format(self.nodes)

class Puzzle():
    # Initialization
    def __init__(self, initial, goal):
        self.initial = initial
        self.goal = goal
        self.moves = {'UP': -3, 'LEFT': -1, 'RIGHT': 1, 'DOWN': 3}

    # Finding the zero index position on the board
    def getZero(self, state):
        return state.index(0)

    # Returns eligible moves via process of elimination
    def options(self, state):
        legalMoves = ['UP', 'LEFT', 'RIGHT', 'DOWN']
        zero = self.getZero(state)
        UP = [0, 1, 2]
        LEFT = [0, 3, 6]
        RIGHT = [2, 5, 8]
        DOWN = [6, 7, 8]
        # Illegal to move up
        if zero in UP:
            legalMoves.remove('UP')
        # Illegal to move left
        if zero in LEFT:
            legalMoves.remove('LEFT')
        # Illegal to move right
        if zero in RIGHT:
            legalMoves.remove('RIGHT')
        # Illegal to move down
        if zero in DOWN:
            legalMoves.remove('DOWN')
        # returns a list of remaining legal moves
        return legalMoves

    # Returns the 'state transition'; i.e., agent executes a move
    def transition(self, state, action):
        # Locating the zero index position on the board
        zero = self.getZero(state)
        # Casting to list type
        stateChange = list(state)
        # Dictating the number of spaces to move during the transition
        position = zero + self.moves[action]
        # The actual transition via swapping
        stateChange[zero], stateChange[position] = stateChange[position], stateChange[zero]
        # Casting to tuple type and returning the result
        stateChange = tuple(stateChange)
        return stateChange

    # Returns true once we have reached our goal node; until then, are we there yet?
    def goalTest(self, state):
        return state == goal

    # Capturing the cost; based on values of the tiles being moved
    def tileCost(self, cost, next, action, current):
        # Locating the zero index position on the board
        zero = self.getZero(current)
        # Casting to list type
        next = list(current)
        # Dictating the number of spaces to move during the transition
        position = zero + self.moves[action]
        # totaling the cost based on the tile value
        cost += next[position]
        # returning the cost
        return cost

    # h1(n) = No. of misplaced tiles relative to goal
    def h1(self, node):
        total = 0
        # Keeping a running total of the No. of misplaced tiles relative to the goal for the state of this node
        for i in range(len(node.state)):
            if node.state[i] != self.goal[i]:
                total += 1
        return total

    # h2(n) = Manhattan Distance
    def h2(self, node):
        for i in range(len(node.state)):
            x = abs((node.state.index(i) // 3) - (goal.index(i) // 3))
            y = abs((node.state.index(i) % 3) - (goal.index(i) % 3))
        return (x+y)

class SearchStrategies():

    # Breadth-First Search
    def breadthFirstSearch(self, problem):
        # Time = No. of Nodes popped from the queue
        time = 0
        # Space = Size of the queue at it's maximum
        space = 0
        # Get the starting state (s0) of our puzzle
        node = Node(problem.initial)
        # Preliminary check
        if problem.goalTest(node.state):
            return node
        # Our fifo queue for Breadth First Search
        frontier = deque([node])
        # For repeated state checking
        stateChecking = {problem.initial}
        # Begin looping over our nodes
        while frontier is not None:
            # retrieve a node from the queue
            node = frontier.popleft()
            # Capturing time for the metrics table
            time += 1
            # Node expansion for next steps
            for child in node.expand(problem):
                # Check if we've reached the goal node yet
                if problem.goalTest(child.state):
                    # printing metrics to terminal as per the assignment requirements
                    print('<-- Algorithm: Breadth-First Search -->')
                    print('Path: ' + str(child.trace(child)))
                    print('Actions: ' + str(child.movesPath(child)))
                    print('Length: ' + str(len(child.trace(child))))
                    print('Cost-Path: ' + str(child.costPath_gn(child)))
                    print('Total Cost-Path: ' + str(child.cost))
                    print('Time: ' + str(time))
                    print('Space: ' + str(space))
                    return 'Message: Puzzle Solved!'
                # Repeated state checking
                if child.state not in stateChecking:
                    # Adds to repeated state checking
                    stateChecking.add(child.state)
                    # Adds to the queue
                    frontier.append(child)
                # Capturing space for the metrics table
                space = len(frontier)
        # Sometimes we fail to find a solution
        return 'Failure'

    # Depth-First Search
    # I reverse the expansion paths which provided significant improvements for every aspect of the metric table
    # The thinking was inspired by the Kosaraju's-Sharir Algorithm
    def depthFirstSearch(self, problem):
        # Time = No. of Nodes popped from the queue
        time = 0
        # Space = Size of the queue at it's maximum
        space = 0
        # Get the starting state (s0) of our puzzle
        # our lifo queue for Depth-First Search
        frontier = [(Node(problem.initial))]
        # For repeated state checking
        stateChecking = {problem.initial}
        # Begin looping over our nodes
        while frontier is not None:
            # retrieve a node from the queue
            node = frontier.pop()
            # Capturing time for the metrics table
            time += 1
            # Check if we've reached the goal node yet
            if problem.goalTest(node.state):
                # printing metrics to terminal as per the assignment requirements
                print('<-- Depth-First Search -->')
                print('Path: ' + str(node.trace(node)))
                print('Actions: ' + str(node.movesPath(node)))
                print('Length: ' + str(len(node.trace(node))))
                print('Cost-Path: ' + str(node.costPath_gn(node)))
                print('Total Cost-Path: ' + str(node.cost))
                print('Time: ' + str(time))
                print('Space: ' + str(space))
                return 'Message: Puzzle Solved!'
            # Adds to repeated state checking
            stateChecking.add(node.state)
            # Reverse node expansion for next steps
            for child in node.expand(problem)[::-1]:
                # Repeated state checking
                if child.state not in stateChecking and child not in frontier:
                    # Adds to the queue
                    frontier.append(child)
                    # Adds to repeated state checking
                    stateChecking.add(child.state)
            # Capturing space for the metrics table
            space = len(frontier)
        # Sometimes we fail to find a solution
        return 'Failure'

    # General Search Function
    # Covers Best-First Search, Uniform-Cost Search, A* Star Search I, and A* Star Search II
    # Avoiding redundant code
    def generalSearch(self, problem, heuristic, type):
        # Time = No. of Nodes popped from the queue
        time = 0
        # Space = Size of the queue at it's maximum
        space = 0
        # Get the starting state (s0) of our puzzle
        node = Node(problem.initial)
        # Our priority queue
        frontier = PriorityQueue([node], fn=heuristic)
        # For repeated state checking
        stateChecking = {problem.initial: node}
        # Begin looping over our nodes
        while frontier is not None:
            # Retrieve a node from the priority queue
            node = frontier.pop()
            # Capturing time for the metrics table
            time += 1
            # Check if we've reached the goal node yet
            if problem.goalTest(node.state):
                # printing metrics to terminal as per the assignment requirements
                print('Path: ' + str(node.trace(node)))
                print('Actions: ' + str(node.movesPath(node)))
                print('Length: ' + str(len(node.trace(node))))
                # Determining which Algorithm in order to print the proper 'Cost-Path' and 'Total Cost-Path'
                # Note: A trade-off between these 12 lines and repeating the code for every algorithm
                if type == 'Best-First Search':
                    print('Cost-Path: ' + str(node.costPath_h1(node)))
                    print('Total Cost-Path: ' + str(node.costPath_h1(node)[-1]))
                elif type == 'Uniform-Cost Search':
                    print('Cost-Path: ' + str(node.costPath_gn(node)))
                    print('Total Cost-Path: ' + str(node.cost))
                elif type == 'A* Search I':
                    print('Cost-Path: ' + str(node.costPath_a1(node)))
                    print('Total Cost-Path: ' + str(node.cost + problem.h1(node)))
                elif type == 'A* Search II':
                    print('Cost-Path: ' + str(node.costPath_a2(node)))
                    print('Total Cost-Path: ' + str(node.cost + problem.h2(node)))
                print('Time: ' + str(time))
                print('Space: ' + str(space))
                return 'Message: Puzzle Solved!'
            # Node expansion for next steps
            for child in node.expand(problem):
                # Repeated state checking + heuristic comparison
                if child.state not in stateChecking or child < stateChecking[child.state]:
                    # Adds to repeated state checking
                    stateChecking[child.state] = child
                    # Adds to the priority queue
                    frontier.add(child)
            # Capturing space for the metrics table
            space = len(frontier)
        # Sometimes we fail to find a solution
        return 'Failure'

# A basic agent class; serving as a wrapper for our 'intelligent agent'...
# ...and provides clarity for the structuring of the concept.
class Agent():
    # Initialization
    def __init__(self, strategy):
        self.strategy = strategy

# Tuples are used because of immutability
# Goal State
goal = (1, 2, 3, 8, 0, 4, 7, 6, 5)
# Initial State: s0 = easy
easy = (1, 3, 4, 8, 6, 2, 7, 0, 5)
# Initial State: s0 = medium
medium = (2, 8, 1, 0, 4, 3, 7, 6, 5)
# Initial State: s0 = hard
hard = (5, 6, 7, 4, 0, 8, 3, 2, 1)

# The Interface: a simple text interface for interacting with the basic problem solving agent
while True:
    # Retrieving the difficulty level from the user
    difficulty = input('Menu Options: 0) Easy, 1) Medium, 2) Hard, *) Exit\n'
                           'Please type the corresponding number: ')
    if difficulty == '0':
        s0 = easy
    elif difficulty == '1':
        s0 = medium
    elif difficulty == '2':
        s0 = hard
    elif difficulty == '*':
        # Exit the program gracefully
        quit()

    # Creating the 'problem'; a Puzzle object
    # Our Puzzle object takes an initial state (s0) and the goal as arguments; defining the 'problem'
    problem = Puzzle(s0, goal)

    # Creating an instance of a SearchStrategies object
    # Our Search Strategies object takes problem and heuristic arguments
    # Note: no heuristic arguments for Breadth-First Search and Depth-First Search
    strategy = SearchStrategies()

    # Retrieving the search strategy from the user
    searchStrategy = input('Menu Options: 0) Breadth-First Search, 1) Depth-First Search, 2) Best-First Search, '
                    '3) Uniform-Cost Search, 4) A* Search I, 5) A* Search II, *) Exit\n'
                    'Please type the corresponding number: ')
    if searchStrategy == '0':
        # Starting the stopwatch
        start = time.time()
        # Our agent object takes a search strategy argument
        agent = Agent(print(strategy.breadthFirstSearch(problem)))
        # Stopping the stopwatch
        print("Stopwatch: " + str(time.time() - start))
    elif searchStrategy == '1':
        start = time.time()
        agent = Agent(print(strategy.depthFirstSearch(problem)))
        print("Stopwatch: " + str(time.time() - start))
    elif searchStrategy == '2':
        print('<-- Algorithm: Best-First Search -->')
        start = time.time()
        agent = Agent(print(strategy.generalSearch(problem, lambda n: problem.h1(n), 'Best-First Search')))
        print("Stopwatch: " + str(time.time() - start))
    elif searchStrategy == '3':
        print('<-- Algorithm: Uniform-Cost Search -->')
        start = time.time()
        agent = Agent(print(strategy.generalSearch(problem, lambda n: n.cost, 'Uniform-Cost Search')))
        print("Stopwatch: " + str(time.time() - start))
    elif searchStrategy == '4':
        print('<-- Algorithm: A* Search I -->')
        start = time.time()
        agent = Agent(print(strategy.generalSearch(problem, lambda n: n.cost + problem.h1(n), 'A* Search I')))
        print("Stopwatch: " + str(time.time() - start))
    elif searchStrategy == '5':
        print('<-- Algorithm: A* Search II -->')
        start = time.time()
        agent = Agent(print(strategy.generalSearch(problem, lambda n: n.cost + problem.h2(n), 'A* Search II')))
        print("Stopwatch: " + str(time.time() - start))
    elif searchStrategy == '*':
        quit()