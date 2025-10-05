from path import Path
from field import Field, _xy_to_rc, _closest_point, _rc_to_xy
import numpy as np
import APF
from CHOMP import CHOMP

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

    def clearance(self, point_xy=None, radius=None):
        if point_xy is None:
            point_xy = self.start_pos
        if radius is None:
            radius = self.radius

        return self.field.dist_to_closest(point_xy=point_xy) - radius - np.sqrt(2) * self.field.resolution
    
    def is_line_coliding(self, p1_xy, p2_xy, eps = 1e-6, max_steps = 1000): # Using sphare marching
        x1, y1 = p1_xy
        x2, y2 = p2_xy

        if self.field.is_of_grid(p1_xy) or self.field.is_of_grid(p2_xy):
            raise IndexError(f"a point is of grid: p1_xy: {p1_xy}, or p2_xy: {p2_xy}")
            
        total_dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)


        if total_dist == 0:
            point_for_np = _xy_to_rc(p1_xy)
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
            

    def is_orca_colliding(
        self,
        p1_xy: tuple[float, float],
        p2_xy: tuple[float, float],
        radius = None,
        max_steps: int = 1000,
        eps: float = 1e-6,
        k = 4 # This is how much ahead of the clearance distance you can check
    ) -> bool:
        """
        True if the road from p1_xy to p2_xy with radius collides.
        Uses look-ahead steps (k) certified by the 1-Lipschitz property to avoid creeping.
        """
        x1, y1 = p1_xy
        x2, y2 = p2_xy

        r = radius if radius is not None else self.radius

        dx, dy = x2 - x1, y2 - y1
        L = np.hypot(dx, dy)
        if L == 0:
            return (self.field.esdf_at_xy(x1, y1) - r) < 0.0
        ux, uy = dx / L, dy / L


        # Early check
        if self.clearance(p1_xy, radius=r) <= 0.0:
            return True

        t = 0.0
        steps = 0
        while t < L - eps and steps < max_steps:
            steps += 1
            px, py = x1 + ux * t, y1 + uy * t
            d0 = self.clearance((px, py), radius=r)
            if d0 <= 0.0:
                return True

            # If current clearance already exceeds remaining distance, we can certify the rest.
            if d0 >= (L - t):
                return False

            # Propose a big step (k>1). We'll *certify* it before accepting.
            s = min(k * d0 + eps, L - t) # Look for colitions ahead of the target (however we cant garante safety outside d0)
            
            # Certifying the step
            while True:
                p1x, p1y = px + ux * s, py + uy * s
                d1 = self.clearance((p1x, p1y), radius=r)

                # Lipschitz certificate: whole chunk [t, t+s] is safe
                if min(d0, d1) - s >= 0.0:
                    t += s
                    break

                # Found an unsafe endpoint -> collision somewhere in [t, t+s]
                if d1 <= 0.0: # This wil only run once at the end of the serch
                    # self.points.append((p1x, p1y)) #! debuging only
                    #TODO the this might not be needed but i have it so that i later can pinpoint the position of impact (dont know for sertain if it works as i havent debuged it)
                    # lo, hi = t, t + s # Lowest and highest posible values
                    # for _ in range(40):                # logarithmic refinement
                    #     if hi - lo <= eps:
                    #         break
                    #     mid = 0.5 * (lo + hi)
                    #     dm = self.clearance((x1 + ux * mid, y1 + uy * mid), radius=r)
                    #     if dm < 0.0: hi = mid
                    #     else:        lo = mid
                    return True

                # Not certified safe, and not unsafe at the far end -> shrink step and retry
                s *= 0.5
                if s <= max(d0, eps):
                    # Fall back to the local safe step to keep making progress
                    t += max(d0, eps)
                    break

        return False



    def punish_point(self, point_xy = None, weight = 5, strength = 2):
        if point_xy is None:
            point_xy = self.start_pos

        c = self.clearance(point_xy=point_xy)
        if c <= 0: # Is inside walls
            # TODO return da big value so that it doesnt try to go there
            return 100000000

        return weight/(c**strength)

    def display(self, display_obstacle_dir=False, points_to_display = [], display_paths = False, acumulate_diaplays = False):
        paths_to_display = self.paths if display_paths else []
        points_to_display.append(self.start_pos); points_to_display.append(self.end_pos); points_to_display.extend(self.points) 
        if display_obstacle_dir:
            dir = self.field.dir_to_closest(point_xy=self.pos)
            dist = self.field.dist_to_closest(self.pos) # if i want to display the exact distance i should change the radius * 2 prt with this
            self.field.display_fields(paths_to_display, mark_xy_list=points_to_display, mark_radius_m=self.radius, arrow_dir_xy=dir, arrow_len_m=self.radius*2, arrow_start_xy=self.pos, acumulate_displays=acumulate_diaplays)

        else:  

            self.field.display_fields(paths_to_display, mark_xy_list=points_to_display, mark_radius_m=self.radius, acumulate_displays=acumulate_diaplays)

    def APF(self, start_xy, goal_xy, max_iter = 10000): # Gives a path to the end or None if it goes to deep into the loop without finding it
        def next_point(point): # -> next xy_point
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
    from RRT import RRT
    field_obj = Field(height=120, width=160, seed=2, spacing=14, resolution=1, empty=False)

    start = (140, 70)
    end = (10, 20)

    radius = 5

    pp = PathPlaner(start, end, field_obj, radius)
    
    initial_path = RRT(pp, start_xy=start, goal_xy=end, ROWS=160, COLS=120)
    if initial_path is None:
        initial_path = Path.line_path_spacing(start_xy=start, end_xy=end, spacing=pp.radius*2)

    
    start_test = initial_path
    pp.paths.append(start_test)
    ny_test_path = start_test

    pp.display(display_obstacle_dir=True, display_paths=True, points_to_display=[start], acumulate_diaplays=True)
    for _ in range(200):
        ny_test_path += CHOMP(field=field_obj, start_path=ny_test_path, weight_obst=1, weight_smoothnes=0.2, weight_total=0.1)

    pp.paths.clear()
    pp.points.clear()
    pp.paths.append(ny_test_path)

    pp.display(display_obstacle_dir=True, display_paths=True, points_to_display=[start])