import numpy as np


# This finds in what direction the attraction and repulsion is, to approach the point
# We should swap it out with another approach if this gets stuck for to long as it is only ment for mostly open terrain
def F_APF(pos, goal_pos, obst_dist, obst_dir, k_rep, k_att, radius, safe_dist, eps=1e-6):
    """
    pos, goal_pos: (x, y)
    obst_dist: distance to nearest obstacle
    obst_dir: unit vector pointing AWAY from Orca (Orca -> obstacle)
    r: a safe distance from the obstacles (to not let it go needlesly far away from obstacles)
    """
    obst_dist -= radius
    if obst_dist < 0:
        raise ValueError(f"Orca is inside an obstacle at point {pos}")
    # attractive (to goal) Changes linearly with distance
    Fx = -k_att * (pos[0] - goal_pos[0]) 
    Fy = -k_att * (pos[1] - goal_pos[1])

    # repulsive (only within the influence radius)
    if obst_dist <= safe_dist:
        dist = max(obst_dist, eps)  # avoid division by zero

        rep_factor = (0.5 * k_rep * (1.0/dist - 1.0/safe_dist)) / (dist*dist)
        Fx -= rep_factor * obst_dir[0]
        Fy -= rep_factor * obst_dir[1]

    return [Fx, Fy]







