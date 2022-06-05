# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Kelvin Ma (kelvinm2@illinois.edu) on 01/24/2021

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,fast)


# Feel free to use the code below as you wish
# Initialize it with a list/tuple of objectives
# Call compute_mst_weight to get the weight of the MST with those objectives
# TODO: hint, you probably want to cache the MST value for sets of objectives you've already computed...

import collections
import heapq
from copy import deepcopy

def DISTANCE(tup1, tup2):
    return abs(tup1[0] - tup2[0]) + abs(tup1[1] - tup2[1])

class MST:
    def __init__(self, objectives):
        self.elements = {key: None for key in objectives}

        # TODO: implement some distance between two objectives 
        # ... either compute the shortest path between them, or just use the manhattan distance between the objectives
        self.distances   = {
                (i, j): DISTANCE(i, j)
                for i, j in self.cross(objectives)
            }
    

        
    # Prim's algorithm adds edges to the MST in sorted order as long as they don't create a cycle
    def compute_mst_weight(self):
        weight = 0
        for distance, i, j in sorted((self.distances[(i, j)], i, j) for (i, j) in self.distances):
            if self.unify(i, j):
                weight += distance
        return weight

    # helper checks the root of a node, in the process flatten the path to the root
    def resolve(self, key):
        path = []
        root = key 
        while self.elements[root] is not None:
            path.append(root)
            root = self.elements[root]
        for key in path:
            self.elements[key] = root
        return root
    
    # helper checks if the two elements have the same root they are part of the same tree
    # otherwise set the root of one to the other, connecting the trees
    def unify(self, a, b):
        ra = self.resolve(a) 
        rb = self.resolve(b)
        if ra == rb:
            return False 
        else:
            self.elements[rb] = ra
            return True

    # helper that gets all pairs i,j for a list of keys
    def cross(self, keys):
        return (x for y in (((i, j) for j in keys if i < j) for i in keys) for x in y)

def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """


    s = maze.start  #start point
    path = []       #the solution path
    visited = []    #visited nodes
    visited.append(s)
    queue = collections.deque() #nodes in queue to add neighbors to
    queue.appendleft(s)
    parent = {s: None}  #backpointers    

    while queue:
        current = queue.popleft()   #get each node starting from the starting point
        
        
        if current == maze.waypoints[0]:
            x = maze.waypoints[0]       #start the backtracking at the goal node
            while parent[x] != None:    #backtrack until None which is what the start Node is assigned to
                temp = parent[x]        #hold value which is the previous node to current node
                path.append(x)          #add it to the path     
                x = temp                #update x to be that previous node bc we are going backwards
            path.append(s)              #add the start node
            return path[::-1]
        neighbors = maze.neighbors(current[0], current[1])  #gets neighbors of the current node
        for neighbor in neighbors:
            if neighbor not in visited:
                parent[neighbor] = current   
                queue.append(neighbor)  #add neighbor onto queue
                visited.append(neighbor)    #to keep track which nodes have been visited
       

          




def astar_single(maze):
    """
    Runs A star for part 2 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    """
    1. Start
    2. Add neighbors to PQ
    3. Pop neigbor with smallest distance 
    4. Add neighbors of that neighbor and do the same until the goal is popped off PQ
    """

    
    s = maze.start  #start point
    tup = (0, s)
    path = []       #the solution path
    frontier = []   #the priority queue to add neighbors to
    heapq.heappush(frontier, tup)
    parent = {s: None}  #backpointers    
    cost_so_far = {s: 0} 

    while frontier:
        current = heapq.heappop(frontier)   #get each node starting from the starting point, gets the smallest one 
        neighbors = maze.neighbors(current[1][0], current[1][1])  #gets neighbors of the current node
        
        if current[1] == maze.waypoints[0]:
            x = maze.waypoints[0]       #start the backtracking at the goal node
            while parent[x] != None:    #backtrack until None which is what the start Node is assigned to
                print(len(path))
                temp = parent[x]        #hold value which is the previous node to current node
                path.append(x)          #add it to the path     
                x = temp                #update x to be that previous node bc we are going backwards
            path.append(s)              #add the start node
            return path[::-1]

        for neighbor in neighbors:
            new_cost = cost_so_far[current[1]] + 1 #+1 each time we add neighbors
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                parent[neighbor] = current[1]  
                cost_so_far[neighbor] = new_cost    #update the cost to be previous cost 
                priority = new_cost + abs(neighbor[0] - maze.waypoints[0][0]) + abs(neighbor[1] - maze.waypoints[0][1])
                tup = (priority, neighbor)
                heapq.heappush(frontier, tup)  #add neighbor onto priority queue
               
            
        


def astar_multiple(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """

    
    s = maze.start                                  #start point
    unvisited_WP = []
    for waypoint in maze.waypoints:                 #initialize list of waypoints 
        unvisited_WP.append(waypoint)    
    state = (s, tuple(unvisited_WP))        
    tup = (0, state)                                #gonna have [(weight, (i, j), list of waypoints havent been to)]                             
    path = []                                       #the solution path
    frontier = []                                   #the priority queue to add neighbors to
    heapq.heappush(frontier, tup)                   #initialize frontier with state
    parent = {state: None}                              #backpointers    
    cost_so_far = {tuple(state): 0}                            #initialize weights
    mst_dict = {tuple(maze.waypoints): MST(unvisited_WP).compute_mst_weight()}

    while frontier:
        _, state = heapq.heappop(frontier)           #get each node starting from the starting point, gets the smallest one 
        coor, unvisited_WP = state
        neighbors = maze.neighbors(coor[0], coor[1])  #gets coordinates of neighbors of the current node: (i, j)
        
        if coor in unvisited_WP:
            unvisited_WP = list(unvisited_WP)     
            unvisited_WP.remove(coor)         #removes whole tuple/coordinates from list of WP
            mst_dict[tuple(unvisited_WP)] = MST(tuple(unvisited_WP)).compute_mst_weight()     #map the MST weight for each state of removed waypoints

            if len(unvisited_WP) == 0:                    #if list of waypoints is empty that means they have all been reached
                x = state                          #start the backtracking at the current node
                while parent[x] != None:                #backtrack until None which is what the start Node is assigned to
                    temp = parent[x]                    #hold value which is the previous node to current node
                    path.append(x[0])                      #add it to the path     ``
                    x = temp                            #update x to be that previous node bc we are going backwards
                path.append(s)                          #add the start node
                return path[::-1]

        for neighbor in neighbors:
            new_cost = cost_so_far[state] + 1  #+1 each time we add neighbors
            if (neighbor, tuple(unvisited_WP)) not in cost_so_far.keys() or new_cost < cost_so_far[(neighbor, tuple(unvisited_WP))]:
                parent[(neighbor, tuple(unvisited_WP))] = state  
                cost_so_far[(neighbor, tuple(unvisited_WP))] = new_cost       #update the cost to be previous cost 
                priority = new_cost + mst_dict[tuple(unvisited_WP)] 
                tup = (priority, (neighbor, tuple(unvisited_WP))) #AKA the state
                heapq.heappush(frontier, tup)       #add neighbor onto priority queue

    return []

      
            

def fast(maze):
    """
    Runs suboptimal search algorithm for part 4.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    s = maze.start                                  #start point
    unvisited_WP = []
    for waypoint in maze.waypoints:                 #initialize list of waypoints 
        unvisited_WP.append(waypoint)    
    state = (s, tuple(unvisited_WP))        
    tup = (0, state)                                #gonna have [(weight, (i, j), list of waypoints havent been to)]                             
    path = []                                       #the solution path
    frontier = []                                   #the priority queue to add neighbors to
    heapq.heappush(frontier, tup)                   #initialize frontier with state
    parent = {state: None}                              #backpointers    
    cost_so_far = {tuple(state): 0}                            #initialize weights
    mst_dict = {tuple(maze.waypoints): MST(unvisited_WP).compute_mst_weight()}

    while frontier:
        _, state = heapq.heappop(frontier)           #get each node starting from the starting point, gets the smallest one 
        coor, unvisited_WP = state
        neighbors = maze.neighbors(coor[0], coor[1])  #gets coordinates of neighbors of the current node: (i, j)
        
        if coor in unvisited_WP:
            unvisited_WP = list(unvisited_WP)     
            unvisited_WP.remove(coor)         #removes whole tuple/coordinates from list of WP
            mst_dict[tuple(unvisited_WP)] = MST(tuple(unvisited_WP)).compute_mst_weight()     #map the MST weight for each state of removed waypoints

            if len(unvisited_WP) == 0:                    #if list of waypoints is empty that means they have all been reached
                x = state                          #start the backtracking at the current node
                while parent[x] != None:                #backtrack until None which is what the start Node is assigned to
                    temp = parent[x]                    #hold value which is the previous node to current node
                    path.append(x[0])                      #add it to the path     ``
                    x = temp                            #update x to be that previous node bc we are going backwards
                path.append(s)                          #add the start node
                return path[::-1]

        for neighbor in neighbors:
            new_cost = cost_so_far[state] + 1  #+1 each time we add neighbors
            if (neighbor, tuple(unvisited_WP)) not in cost_so_far.keys() or new_cost < cost_so_far[(neighbor, tuple(unvisited_WP))]:
                parent[(neighbor, tuple(unvisited_WP))] = state  
                cost_so_far[(neighbor, tuple(unvisited_WP))] = new_cost       #update the cost to be previous cost 
                priority = new_cost + 2*mst_dict[tuple(unvisited_WP)] 
                tup = (priority, (neighbor, tuple(unvisited_WP))) #AKA the state
                heapq.heappush(frontier, tup)       #add neighbor onto priority queue

    return []
    
            
