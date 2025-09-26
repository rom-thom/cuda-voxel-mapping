import numpy
from field import Field
from path import Path
import copy




def CHOMP(field: Field, start_path: Path, weight_obst: float, weight_smoothnes: float, weight_total: float) -> Path: # antar at start og slutt er start og slutt i start_path

    path = copy.deepcopy(start_path)

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

    # I have to do the smothnes while i add the obstacle as i have to kind of solve the entire thing for doing the Thomas algorythm

    