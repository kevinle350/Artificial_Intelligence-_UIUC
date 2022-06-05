# geometry.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by James Gao (jamesjg2@illinois.edu) on 9/03/2021
# Inspired by work done by Jongdeog Lee (jlee700@illinois.edu)

"""
This file contains geometry functions necessary for solving problems in MP2
"""

import math
import numpy as np
from alien import Alien
def sqr(x):
    return x * x 
def dist2(v, w):
    return sqr(v[0] - w[0]) + sqr(v[1] - w[1]) 
def distToSegmentSquared(start, end, point):
    l2 = dist2(end, point)
    if l2 == 0:
        return dist2(start, end)
    t = ((start[0] - end[0]) * (point[0] - end[0]) + (start[1] - end[1]) * (point[1] - end[1])) / l2
    t = max(0, min(1, t))
    other = (end[0] + t * (point[0] - end[0]), end[1] + t * (point[1] - end[1]))
    return dist2(start, other)
def min_dist_to_line(start, end, point):
    # reference: https://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment
    return np.sqrt(distToSegmentSquared(start, end, point))

# def minDistance(S, E, P):
#     # vector SE
#     SE = [None, None]
#     SE[0] = E[0] - S[0]
#     SE[1] = E[1] - S[1]
#     # vector EP
#     EP = [None, None]
#     EP[0] = P[0] - E[0]
#     EP[1] = P[1] - E[1]
#     # vector SP
#     SP = [None, None]
#     SP[0] = P[0] - S[0]
#     SP[1] = P[1] - S[1]
#     # Variables to store dot product
#     # Calculating the dot product
#     SE_EP = SE[0] * EP[0] + SE[1] * EP[1]
#     SE_SP = SE[0] * SP[0] + SE[1] * SP[1]
#     # Minimum distance from
#     # point E to the line segment
#     reqAns = 0
#     # Case 1
#     if (SE_EP > 0) :
#         # Finding the magnitude
#         y = P[1] - E[1]
#         x = P[0] - E[0]
#         reqAns = np.sqrt(x * x + y * y)
#     # Case 2
#     elif (SE_SP < 0) :
#         y = P[1] - S[1]
#         x = P[0] - S[0]
#         reqAns = np.sqrt(x * x + y * y)
#     # Case 3
#     else: 
#         # Finding the perpendicular distance
#         x1 = SE[0]
#         y1 = SE[1]
#         x2 = SP[0]
#         y2 = SP[1]
#         mod = np.sqrt(x1 * x1 + y1 * y1)
#         reqAns = abs(x1 * y2 - y1 * x2) / mod
#     return reqAns

#reference: https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
def onSegment(p, q, r):
    if ( (q[0] <= max(p[0], r[0])) and (q[0] >= min(p[0], r[0])) and
           (q[1] <= max(p[1], r[1])) and (q[1] >= min(p[1], r[1]))):
        return True
    return False

def orientation(p, q, r):  
    val = (float(q[1] - p[1]) * (r[0] - q[0])) - (float(q[0] - p[0]) * (r[1] - q[1]))
    if (val > 0):       # Clockwise orientation
        return 1
    elif (val < 0):     # Counterclockwise orientation
        return 2
    else:               # Collinear orientation
        return 0
 
# the line segment 'p1q1' and 'p2q2' intersect.
def doIntersect(p1,q1,p2,q2):
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)
    if ((o1 != o2) and (o3 != o4)): # General case
        return True
    # Special Cases
    if ((o1 == 0) and onSegment(p1, p2, q1)):       # p1 , q1 and p2 are collinear and p2 lies on segment p1q1
        return True 
    if ((o2 == 0) and onSegment(p1, q2, q1)):       # p1 , q1 and q2 are collinear and q2 lies on segment p1q1
        return True
    if ((o3 == 0) and onSegment(p2, p1, q2)):       # p2 , q2 and p1 are collinear and p1 lies on segment p2q2
        return True
    if ((o4 == 0) and onSegment(p2, q1, q2)):       # p2 , q2 and q1 are collinear and q1 lies on segment p2q2
        return True
    return False                                    # If none of the cases




def does_alien_touch_wall(alien, walls,granularity):
    """Determine whether the alien touches a wall

        Args:
            alien (Alien): Instance of Alien class that will be navigating our map
            walls (list): List of endpoints of line segments that comprise the walls in the maze in the format [(startx, starty, endx, endx), ...]
            granularity (int): The granularity of the map

        Return:
            True if touched, False if not
    """
    if alien.is_circle():
        rad = alien.get_width()
        center = alien.get_centroid()
        for wall in walls:
            s = (wall[0], wall[1])
            e = (wall[2], wall[3])
            distance = min_dist_to_line(center, s, e)     
            if np.isclose(distance, rad + granularity/np.sqrt(2)) or distance < rad + granularity/np.sqrt(2):
                return True

    else:
        rad = alien.get_width()
        center = alien.get_centroid()
        head = alien.get_head_and_tail()[0]
        tail = alien.get_head_and_tail()[1]
        for wall in walls:
            s = (wall[0], wall[1])
            e = (wall[2], wall[3])
            distance1 = min_dist_to_line(head, s, e)          
            distance2 = min_dist_to_line(tail, s, e)      
            distance3 = min_dist_to_line(s, head, tail)    
            distance4 = min_dist_to_line(e, head, tail)      
            distance5 = min(distance1, distance2, distance3, distance4)   
            if np.isclose(distance5, rad + granularity/np.sqrt(2)) or distance5 < rad + granularity/np.sqrt(2) or doIntersect(head, tail, s, e):
                return True
            
    return False
 

def does_alien_touch_goal(alien, goals):
    """Determine whether the alien touches a goal
        
        Args:
            alien (Alien): Instance of Alien class that will be navigating our map
            goals (list): x, y coordinate and radius of goals in the format [(x, y, r), ...]. There can be multiple goals
        
        Return:
            True if a goal is touched, False if not.
    """
    if alien.is_circle():
        rad = alien.get_width()
        center = alien.get_centroid()
        for goal in goals:
            distSq = (center[0] - goal[0]) * (center[0] - goal[0]) + (center[1] - goal[1]) * (center[1] - goal[1])
            radSumSq = (rad + goal[2]) * (rad + goal[2])
            if np.isclose(distSq, radSumSq) or distSq < radSumSq:
                return True
    else:
        rad = alien.get_width()
        head = alien.get_head_and_tail()[0]
        tail = alien.get_head_and_tail()[1]
        for goal in goals:
            distSq = min_dist_to_line(goal, head, tail) 
            radSumSq = (rad + goal[2])
            if np.isclose(distSq, radSumSq) or distSq < radSumSq:
                return True
    return False
   
    


def is_alien_within_window(alien, window,granularity):
    """Determine whether the alien stays within the window
        
        Args:
            alien (Alien): Alien instance
            window (tuple): (width, height) of the window
            granularity (int): The granularity of the map
    """
    walls = [(0, 0, window[0], 0), (0, 0, 0, window[1]), (0, window[1], window[0], window[1]), (window[0], window[1], window[0], 0)]
    
    return not does_alien_touch_wall(alien, walls, granularity)
    

                           
        
        
                                               
        



if __name__ == '__main__':
    #Walls, goals, and aliens taken from Test1 map
    walls =   [(0,100,100,100),  
                (0,140,100,140),
                (100,100,140,110),
                (100,140,140,130),
                (140,110,175,70),
                (140,130,200,130),
                (200,130,200,10),
                (200,10,140,10),
                (175,70,140,70),
                (140,70,130,55),
                (140,10,130,25),
                (130,55,90,55),
                (130,25,90,25),
                (90,55,90,25)]
    goals = [(110, 40, 10)]
    window = (220, 200)

    def test_helper(alien : Alien, position, truths):
        alien.set_alien_pos(position)
        config = alien.get_config()

        touch_wall_result = does_alien_touch_wall(alien, walls, 0) 
        touch_goal_result = does_alien_touch_goal(alien, goals)
        in_window_result = is_alien_within_window(alien, window, 0)

        assert touch_wall_result == truths[0], f'does_alien_touch_wall(alien, walls) with alien config {config} returns {touch_wall_result}, expected: {truths[0]}'
        assert touch_goal_result == truths[1], f'does_alien_touch_goal(alien, goals) with alien config {config} returns {touch_goal_result}, expected: {truths[1]}'
        assert in_window_result == truths[2], f'is_alien_within_window(alien, window) with alien config {config} returns {in_window_result}, expected: {truths[2]}'

    #Initialize Aliens and perform simple sanity check. 
    alien_ball = Alien((30,120), [40, 0, 40], [11, 25, 11], ('Horizontal','Ball','Vertical'), 'Ball', window)
    test_helper(alien_ball, alien_ball.get_centroid(), (False, False, True))

    alien_horz = Alien((30,120), [40, 0, 40], [11, 25, 11], ('Horizontal','Ball','Vertical'), 'Horizontal', window)	
    test_helper(alien_horz, alien_horz.get_centroid(), (False, False, True))

    alien_vert = Alien((30,120), [40, 0, 40], [11, 25, 11], ('Horizontal','Ball','Vertical'), 'Vertical', window)	
    test_helper(alien_vert, alien_vert.get_centroid(), (True, False, True))

    edge_horz_alien = Alien((50, 100), [100, 0, 100], [11, 25, 11], ('Horizontal','Ball','Vertical'), 'Horizontal', window)
    edge_vert_alien = Alien((200, 70), [120, 0, 120], [11, 25, 11], ('Horizontal','Ball','Vertical'), 'Vertical', window)

    alien_positions = [
                        #Sanity Check
                        (0, 100),

                        #Testing window boundary checks
                        (25.6, 25.6),
                        (25.5, 25.5),
                        (194.4, 174.4),
                        (194.5, 174.5),

                        #Testing wall collisions
                        (30, 112),
                        (30, 113),
                        (30, 105.5),
                        (30, 105.6), # Very close edge case
                        (30, 135),
                        (140, 120),
                        (187.5, 70), # Another very close corner case, right on corner
                        
                        #Testing goal collisions
                        (110, 40),
                        (145.5, 40), # Horizontal tangent to goal
                        (110, 62.5), # ball tangent to goal
                        
                        #Test parallel line oblong line segment and wall
                        (50, 100),
                        (200, 100),
                        (205.5, 100) #Out of bounds
                    ]

    #Truths are a list of tuples that we will compare to function calls in the form (does_alien_touch_wall, does_alien_touch_goal, is_alien_within_window)
    alien_ball_truths = [
                            (True, False, False),
                            (False, False, True),
                            (False, False, True),
                            (False, False, True),
                            (False, False, True),
                            (True, False, True),
                            (False, False, True),
                            (True, False, True),
                            (True, False, True),
                            (True, False, True),
                            (True, False, True),
                            (True, False, True),
                            (False, True, True),
                            (False, False, True),
                            (True, True, True),
                            (True, False, True),
                            (True, False, True),
                            (True, False, True)
                        ]
    alien_horz_truths = [
                            (True, False, False),
                            (False, False, True),
                            (False, False, False),
                            (False, False, True),
                            (False, False, False),
                            (False, False, True),
                            (False, False, True),
                            (True, False, True),
                            (False, False, True),
                            (True, False, True),
                            (False, False, True),
                            (True, False, True),
                            (True, True, True),
                            (False, True, True),
                            (True, False, True),
                            (True, False, True),
                            (True, False, False),
                            (True, False, False)
                        ]
    alien_vert_truths = [
                            (True, False, False),
                            (False, False, True),
                            (False, False, False),
                            (False, False, True),
                            (False, False, False),
                            (True, False, True),
                            (True, False, True),
                            (True, False, True),
                            (True, False, True),
                            (True, False, True),
                            (True, False, True),
                            (False, False, True),
                            (True, True, True),
                            (False, False, True),
                            (True, True, True),
                            (True, False, True),
                            (True, False, True),
                            (True, False, True)
                        ]

    for i in range(len(alien_positions)):
        test_helper(alien_ball, alien_positions[i], alien_ball_truths[i])
        test_helper(alien_horz, alien_positions[i], alien_horz_truths[i])
        test_helper(alien_vert, alien_positions[i], alien_vert_truths[i])

    #Edge case coincide line endpoints
    test_helper(edge_horz_alien, edge_horz_alien.get_centroid(), (True, False, False))
    test_helper(edge_horz_alien, (110,55), (True, True, True))
    test_helper(edge_vert_alien, edge_vert_alien.get_centroid(), (True, False, True))


    print("Geometry tests passed\n")