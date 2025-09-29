import numpy as np
from field import Field
from path import Path
import copy
from Thomas_alg import Thomas



# -Δx_(i-1) + 2 * Δx_i - Δx_(i+1) = w_t * (w_s * (2*x_i - x_(i-1) - x_(i+1)) + w_o * g_i)
def CHOMP(field: Field, start_path: Path, weight_obst: float, weight_smoothnes: float, weight_total: float) -> Path: # antar at start og slutt er start og slutt i start_path

    path = copy.deepcopy(start_path)
    np_path = path.to_np_array()

    # Obstacle math:
        # g_i = nabla dist_punishment = (trough the infamus chain rule) = dist_punishment'(clearance) * nabla clearance(x) 
    
    # The punishment itselfe is as of now 1/2 * (clearance)**2 but this could change if i for example want a big safe distance OSLT
    def change_in_dist_punishment_per_c(clearance):
        return clearance

    def obstacle_effect(point_xy):
        dpunish_dclearance = change_in_dist_punishment_per_c(field.dist_to_closest(point_xy=point_xy)) 
        obst_direction = field.dir_to_closest(point_xy=point_xy)
        return [weight_obst * dpunish_dclearance * (-i) for i in obst_direction] # minus because it points toward the obstacle, but the gradient should be away

    # Smoothnes math:

        # {insert mathematics here}

    # I have to do the smothnes while i add the obstacle as i have to kind of solve the entire thing to solve for the individual shifts. (aka its not modular)



    obs_i = np.array([obstacle_effect(point_xy=p) for p in np_path])
    for i in range(len(obs_i)):
        if i == 0: # Remove punishment for endpoints (they are stuck)
            obs_i[0] = 0
            continue
        if i == len(obs_i)-1:
            obs_i[i] = 0
            continue
        
        obs_i[i] += weight_smoothnes * (2*np_path[i] -  np_path[i-1] - np_path[i+1])
        obs_i[i] *= weight_total

        

    return Path(list(Thomas(np_path, obs_i)))
