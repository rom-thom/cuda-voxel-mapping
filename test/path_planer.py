from path import Path
from field import Field, _xy_to_rc
import numpy as np
import APF

def dist_between(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

class PathPlaner:
    def __init__(self, start_pos, end_pos, field: Field, collision_radius, current_paths: list[Path] = []):
        self.pos = start_pos
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.field = field
        self.esdf = self.field.esdf
        self.paths = current_paths
        self.points = []
        self.radius = collision_radius

    def clairance(self, point_xy=None, radius=None):
        if point_xy is None:
            point_xy = self.start_pos
        if radius is None:
            radius = self.radius

        return self.field.dist_to_closest(point_xy=point_xy) - radius
    
    def is_line_coliding(self, p1_xy, p2_xy, eps = 1e-6, max_steps = 1000): # Using sphare marching
        x1, y1 = p1_xy
        x2, y2 = p2_xy

        if self.field.is_of_grid(p1_xy) or self.field.is_of_grid(p2_xy):
            raise IndexError(f"a point is of grid: p1_xy: {p1_xy}, or p2_xy: {p2_xy}")
            
        total_dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)


        if total_dist == 0:
            point_for_np = _xy_to_rc(point)
            obs_dist = self.esdf[point_for_np[0]][point_for_np[1]]
            return obs_dist < eps
        

        current_point = lambda t: (x1 + t* (x2-x1), y1 + t* (y2-y1)) # Here t is the part of the line it has come to (t = 0 => in p1, t = 0.5 => in the midle) (x, y) = A + t (B-A)
        current_t = 0
        for _ in range(max_steps):

            point = current_point(current_t)
            point_for_np = _xy_to_rc(point)
            step_size = self.esdf[point_for_np[0]][point_for_np[1]]
            if step_size <= eps:
                return True
            
            current_t += step_size/total_dist
            if current_t > 1:
                return False
        
        raise RuntimeError("The function did not find the end after max iterations")
            
    
    # TODO implement this beter
    def collision_between_orcas(self, p1_xy: tuple[float, float], p2_xy: tuple[float, float])->bool: # Finds  wether orca could have gone there
        dist_between = np.sqrt((p2_xy[0]-p1_xy[0])**2 + (p2_xy[1]-p1_xy[1])**2)
            
        # TODO Swap this for sphare marching oslt right now we check needlesly much space
        
        sub_path = Path.line_path_spacing(start_xy=p1_xy, end_xy=p2_xy, spacing=self.radius*2, include_end=False)

        # self.paths.append(sub_path) # !!! For testing only
        # self.points.extend(sub_path.path) # !!! For testing only

        for nr, point in enumerate(sub_path.path):
            if nr % 2 != 0: # We check every other element just with extra radius, to be able to see for the neighboring ones, and then it wil not me any intersection to check for
                if self.clairance(point_xy=point, radius=3*self.radius) < 0:
                    return True
                
        if len(sub_path.path) % 2 == 0:  # i also have to check the last element if that is odd, as otherwise the last element could be coliding
            if self.clairance(point_xy=sub_path.path[-1], radius=3*self.radius) < 0:
                return True


        
        # TODO lets do some ray marching (https://typhomnt.github.io/teaching/ray_tracing/raymarching_intro/)

        if self.clairance(p1_xy) <= 0 or self.clairance(p2_xy) <= 0: # seing if the points themselves are coliding with a wall
            return True
        
        return False

    def punish_point(self, point_xy = None, weight = 5, strength = 2):
        if point_xy is None:
            point_xy = self.start_pos

        c = self.clairance(point_xy=point_xy)
        if c <= 0: # Is inside walls
            # TODO return da big value so that it doesnt try to go there
            return 100000000

        return weight/(c**strength)

    def display(self, display_obstacle_dir=False, points_to_display = [], display_paths = False):
        paths_to_display = self.paths if display_paths else []
        points_to_display.append(self.start_pos); points_to_display.append(self.end_pos); points_to_display.extend(self.points) 
        if display_obstacle_dir:
            dir = self.field.dir_to_closest(point_xy=self.pos)
            dist = self.field.dist_to_closest(self.pos) # if i want to display the exact distance i should change the radius * 2 prt with this
            self.field.display_fields(paths_to_display, mark_xy_list=points_to_display, mark_radius_m=self.radius, arrow_dir_xy=dir, arrow_len_m=self.radius*2, arrow_start_xy=self.pos)

        else:  

            self.field.display_fields(paths_to_display, mark_xy_list=points_to_display, mark_radius_m=self.radius)


    def APF(self, start_xy, goal_xy, max_iter = 10000): # Gives a path to the end or None if it goes to deep into the loop without finding it
        def next_point(point):
            index_point = (int(point[0]), int(point[1]))
            dist_obs = self.field.dist_to_closest(index_point)
            dir_obs = self.field.dir_to_closest(index_point)
            gx, gy =  APF.F_APF(point, goal_xy, dist_obs, dir_obs, 2000, 1, self.radius, self.radius * 200)
            return (point[0]+gx*0.005, point[1] + gy*0.005)
        
        current_point = start_xy
        path = Path([start_xy])
        for i in range(max_iter):

            current_point = next_point(current_point)
            path.add_point((current_point[0], current_point[1]))
            if dist_between(current_point, goal_xy) < self.radius:
                print(f"Iter count: {i}")
                return path
        
        return path

    
if __name__ == "__main__":
    field_obj = Field(height=120, width=160, seed=2, spacing=14, resolution=1)

    # test_point = (100, 60)
    start = (140, 100)
    end = (10, 30)

    current_paths = []

    radius = 5

    pp = PathPlaner(start, end, field_obj, radius)
    
    line_spacing = Path.line_path_spacing(start_xy=start, end_xy=end, spacing=pp.radius*2)
    line = Path.line_path(start_xy=start, end_xy=end, n = 3)
    pp.paths.append(line_spacing)

    test_path = pp.APF(start, end)
    
    pp.paths.append(test_path)

    pp.paths.extend(current_paths)
    pp.display(display_obstacle_dir=True, display_paths=True, points_to_display=[start])

