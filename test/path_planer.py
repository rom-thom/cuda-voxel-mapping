from path import Path
from field import Field
import numpy as np


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
    
    def collision_between_points(self, p1_xy: tuple[float, float], p2_xy: tuple[float, float])->bool:
        dist_between = np.sqrt((p2_xy[0]-p1_xy[0])**2 + (p2_xy[1]-p1_xy[1])**2)
            
        # TODO Swap this for sphare marching right now we check needlesly much space
        
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
            return 10000000

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



if __name__ == "__main__":
    field_obj = Field(height=120, width=160, seed=5, spacing=14, resolution=1)

    test_point = (60, 80)
    start = (140, 100)
    end = (10, 30)

    current_paths = [Path([start, (80, 80), (70, 85), (65, 90)])]

    radius = 5

    pp = PathPlaner(start, end, field_obj, radius)
    
    line_spacing = Path.line_path_spacing(start_xy=start, end_xy=end, spacing=pp.radius*2)
    line = Path.line_path(start_xy=start, end_xy=end, n = 3)
    pp.paths.append(line_spacing)


    print(pp.collision_between_points(start, test_point))

    pp.display(display_obstacle_dir=True, display_paths=True, points_to_display=[])

