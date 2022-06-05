# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains search functions.
"""
# Search should return the path and the number of states explored.
# The path should be a list of tuples in the form (alpha, beta, gamma) that correspond
# to the positions of the path taken by your search algorithm.
# Number of states explored should be a number.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs)
# You may need to slight change your previous search functions in MP1 since this is 3-d maze

from collections import deque
from const import SHAPE
from heapq import heappop, heappush

def search(maze, searchMethod):
    return {
        "bfs": bfs,
    }.get(searchMethod, [])(maze)

def bfs(maze, ispart1=False):
    # Write your code here
    """
    This function returns optimal path in a list, which contains start and objective.
    If no path found, return None. 

    Args:
        maze: Maze instance from maze.py
        ispart1: pass this variable when you use functions such as getNeighbors and isObjective. DO NOT MODIFY THIS
    """
    s = maze.getStart()  #start point
    path = []       #the solution path
    visited = []    #visited nodes
    visited.append(s)
    queue = deque() #nodes in queue to add neighbors to
    queue.appendleft(s)
    parent = {s: None}  #backpointers    

    while queue:
        current = queue.popleft()   #get each node starting from the starting point
        if maze.isObjective(current[0], current[1], current[2], ispart1):
            x = current       #start the backtracking at the goal node
            while parent[x] != None:    #backtrack until None which is what the start Node is assigned to
                temp = parent[x]        #hold value which is the previous node to current node
                path.append(x)          #add it to the path     
                x = temp                #update x to be that previous node bc we are going backwards
            path.append(s)              #add the start node
            if len(path) == 0:
                return None
            else:
                return path[::-1]
        neighbors = maze.getNeighbors(current[0], current[1], current[2], ispart1)  #gets neighbors of the current node   
        for neighbor in neighbors:
            if neighbor not in visited and neighbor not in queue:
                parent[neighbor] = current   
                queue.append(neighbor)  #add neighbor onto queue
                visited.append(neighbor)    #to keep track which nodes have been visited
    return None