total=int(input('Enter no. of bananas at starting '))
distance=int(input('Enter distance you want to cover '))
capacity=int(input('Enter max load capacity of your camel '))

lose=0
start=total
for i in range(distance):
    while start>0:
        start=start-capacity
        if start==1:
            lose=lose-1
        lose=lose+2

    lose=lose-1
    start=total-lose
    if start==0:
        break

print("The bananas left", start)



-------------------------

def graph_coloring(graph):
    V = len(graph)
    result = [-1] * V
    result[0] = 0
    available = [False] * V

    for u in range(1, V):
        for i in graph[u]:
            if result[i] != -1:
                available[result[i]] = True

        color = next(c for c, available in enumerate(available) if not available)
        result[u] = color

        for i in graph[u]:
            if result[i] != -1:
                available[result[i]] = False

    chromatic_number = max(result) + 1
    for u in range(V):
        print(f"Vertex {u} --->  Color {result[u]}")
    return chromatic_number

graph = [
    [1, 2, 3],
    [0, 2, 4],
    [0, 1],
    [0, 4],
    [1, 3]
]

chromatic_number = graph_coloring(graph)
print(f"Chromatic Number: {chromatic_number}")


----------


from itertools import permutations

def solve_cryptarithmetic(equation):
    letters = ''.join(set(filter(str.isalpha, equation)))

    for perm in permutations('0123456789', len(letters)):
        if '0' in perm[:len(set(word[0] for word in equation.replace('+', ' ').replace('=', ' ').split() if word.isalpha()))]:
            continue

        table = str.maketrans(letters, ''.join(perm))
        translated = equation.translate(table)

        try:
            if eval(translated.replace('=', '==')):
                return {letters[i]: perm[i] for i in range(len(letters))}
        except:
            continue

    return None

# Example usage
equation = "SCOOBY + DOOO = BUSTED"
solution = solve_cryptarithmetic(equation)
if solution:
    print("Solution found!")
    print(solution)
else:
    print("No solution exists.")





--------------------    --------




    #BFS
from collections import deque

def bfs(graph, start):
    visited = [False] * len(graph)
    queue = deque([start])
    visited[start] = True
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in graph[node]:
            if not visited[neighbor]:
                queue.append(neighbor)
                visited[neighbor] = True
    return order

graph = [
    [1, 2, 3],
    [0, 2, 4],
    [0, 1],
    [0, 4],
    [1, 3]
]

print(f"BFS Order starting from vertex 0: {bfs(graph, 0)}")



#DFS
def dfs(graph, start):
    visited = [False] * len(graph)
    stack = [start]
    order = []
    while stack:
        node = stack.pop()
        if not visited[node]:
            visited[node] = True
            order.append(node)
            stack.extend(reversed(graph[node]))
    return order

graph = [
    [1, 2, 3],
    [0, 2, 4],
    [0, 1],
    [0, 4],
    [1, 3]
]

print(f"DFS Iterative Order starting from vertex 0: {dfs(graph, 0)}")


-----------------------

#Best First Search
from queue import PriorityQueue

def best_first_search(graph, start, goal):
    visited = set()
    queue = PriorityQueue()
    queue.put((0, start))
    visited.add(start)
    while not queue.empty():
        cost, node = queue.get()
        print(f"Visiting node {node} with cost {cost}")
        if node == goal:
            print("Goal reached!")
            return True
        for neighbor, neighbor_cost in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.put((neighbor_cost, neighbor))
    print("Goal not reached.")
    return False


graph = {
    'A': [('B', 1), ('C', 3)],
    'B': [('A', 1), ('D', 5)],
    'C': [('A', 3)],
    'D': [('B', 5)]
}

start_node = 'A'
goal_node = 'D'
best_first_search(graph, start_node, goal_node)





-------------------------------------
#A* search
from queue import PriorityQueue

def heuristic(node, goal):
    return abs(node[0] - goal[0]) + abs(node[1] - goal[1])

def astar_search(graph, start, goal):
    visited = set()
    queue = PriorityQueue()
    queue.put((0, start))
    visited.add(start)

    while not queue.empty():
        cost, node = queue.get()
        if node == goal:
            return True
        for neighbor, neighbor_cost in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                total_cost = cost + neighbor_cost + heuristic(neighbor, goal)
                queue.put((total_cost, neighbor))
    return False

graph = {
    (0, 0): [((0, 1), 1), ((1, 0), 1)],
    (0, 1): [((0, 0), 1), ((0, 2), 1)],
    (0, 2): [((0, 1), 1), ((1, 2), 1)],
    (1, 0): [((0, 0), 1), ((2, 0), 1)],
    (1, 2): [((0, 2), 1), ((2, 2), 1)],
    (2, 0): [((1, 0), 1), ((2, 1), 1)],
    (2, 1): [((2, 0), 1), ((2, 2), 1)],
    (2, 2): [((1, 2), 1), ((2, 1), 1)]
}

start_node = (0, 0)
goal_node = (2, 2)

print("Goal reached!" if astar_search(graph, start_node, goal_node) else "Goal not reached.")



-----------------------------------------------


def simple_hill_climbing(numbers):
    i = 0
    while True:
        if i + 1 < len(numbers):
            if numbers[i] < numbers[i + 1]:
                i += 1
            else:
                return numbers[i]
        else:
            return numbers[i]

numbers = [1, 3, 7, 12, 54]
max_number = simple_hill_climbing(numbers)

print(f"The maximum number in the list is: {max_number}")



-------------------------------------------------


import math

def minimax(state, depth, is_maximizing):
    if depth == 0 or abs(state) == 1:
        return state

    best_score = -math.inf if is_maximizing else math.inf
    for move in range(1, 4):  # Assuming there are only 3 possible moves
        score = minimax(state + move if is_maximizing else state - move, depth - 1, not is_maximizing)
        best_score = max(score, best_score) if is_maximizing else min(score, best_score)
    return best_score

initial_state = 0
print("Score:", minimax(initial_state, 3, True))


------------------------------------------------------------


def unify(x, y, theta={}):
    if theta is None:
        return None
    elif x == y:
        return theta
    elif isinstance(x, str):
        return unify_var(x, y, theta)
    elif isinstance(y, str):
        return unify_var(y, x, theta)
    elif isinstance(x, list) and isinstance(y, list):
        if len(x) != len(y):
            return None
        for x_i, y_i in zip(x, y):
            theta = unify(x_i, y_i, theta)
            if theta is None:
                return None
        return theta
    else:
        return None

def unify_var(var, x, theta):
    if var in theta:
        return unify(theta[var], x, theta)
    elif x in theta:
        return unify(var, theta[x], theta)
    else:
        theta[var] = x
        return theta

# Example usage
print(unify(['x', 'y'], ['a', 'b']))
print(unify(['x', 'y'], ['y', 'b']))
print(unify(['x', 'y'], ['a', 'b', 'c']))



------------------------------------------------------


import random

def monty_hall_simulation(num_trials, switch=True):
    wins = 0
    for _ in range(num_trials):
        prize_door = random.randint(1, 3)
        chosen_door = random.randint(1, 3)
        opened_door = random.choice([door for door in range(1, 4) if door != chosen_door and door != prize_door])
        if switch:
            chosen_door = next(door for door in range(1, 4) if door != chosen_door and door != opened_door)
        if chosen_door == prize_door:
            wins += 1
    return wins / num_trials


num_trials = 10000
switch_win_percentage = monty_hall_simulation(num_trials, switch=True)
stay_win_percentage = monty_hall_simulation(num_trials, switch=False)

print("Win percentage with switching:", switch_win_percentage)
print("Win percentage without switching:", stay_win_percentage)




--------------------------------


from itertools import combinations as c

def belief_from_plausibility(p):
    return 1 - p

def combine(f, b):
    return {(): 1.0, **{s: sum((-1) ** (len(l) - len(s)) * b[l] for l in c(s, len(s))) for s in f}}

# Example usage
plausibility = 0.6
belief = belief_from_plausibility(plausibility)
print("Belief:", belief)

focal_elements = ('A', 'B', 'C')
belief_functions = {('A',): 0.2, ('B',): 0.3, ('C',): 0.4, ('A', 'B'): 0.1, ('A', 'C'): 0.1, ('B', 'C'): 0.2}
combined_belief = combine(focal_elements, belief_functions)
print("Combined belief:", combined_belief)

