import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt, binary_dilation
from matplotlib.patches import Circle

from path import Path





# -----Helpers-----

def gradient_3x3(patch: np.ndarray, dx, dy):
    assert patch.shape == (3, 3)

    #    Kolonne i pos retning   -  kolonna i negativ retning (- fordi det er negativ side)
    g_x = (patch[:, 2].sum() - patch[:, 0].sum()) / (6*dx)

    # toppen er positiv retning
    g_y = (patch[2, :].sum() - patch[0, :].sum()) / (6*dy)

    return (g_x, g_y)


        
def _xy_to_rc(xy):
    """(x,y) -> (r,c) for numpy indexing."""
    x, y = xy
    return int(y), int(x)  # r=y, c=x

def _rc_to_xy(rc):
    """(r,c) -> (x,y) from numpy indexing."""
    r, c = rc
    return int(c), int(r)  # r=y, c=x


def _closest_point(start_rc, dir_xy):
    """
    Return the next grid cell in direction dir_xy. Not my problem if it is outside the grid
    
    Args:
        start_rc: (r, c) current integer cell (NumPy row/col).
        dir_xy: (dx, dy) direction in math coords (x right, y up).
        grid_shape: optional (rows, cols) to clip; if next step exits, returns None.
    """
    r0, c0 = start_rc
    dx, dy = dir_xy
    if dx == 0 and dy == 0:
        raise ValueError("it has to move at leats a bit.")

    # from xy direction to rc
    dc = float(dx)
    dr = float(-dy)

    if abs(dc) > abs(dr): # move vertical
        step_c = 1 if dc > 0  else -1
        return (r0, c0 + step_c)
    if abs(dc) < abs(dr): # move the other vertical
        step_r = 1 if dr > 0  else -1
        return (r0 + step_r, c0)
    step_r = 1 if dr > 0  else -1
    step_c = 1 if dc > 0  else -1
    return (r0 + step_r, c0 + step_c)
    

class Field:
    def __init__(self, height=120, width=160, seed=0, spacing=12, resolution=0.05, empty=False):
        self.height = height          # cells in y
        self.width = width            # cells in x
        self.seed = seed
        self.spacing = spacing        # cells
        self.resolution = resolution  # meters / cell
        self.is_empty = empty

        # build once; you can rebuild later if needed
        self.fields = self.make_fields()
        self.esdf = self.fields["esdf"]

    # ---------- helpers: index conventions ----------
    def _meters_to_cells(self, xy_m):
        """(x,y) meters -> (x,y) cells (float)."""
        x, y = xy_m
        return x / self.resolution, y / self.resolution

    def _cells_to_meters(self, xy_c):
        """(x,y) cells -> (x,y) meters (float)."""
        x, y = xy_c
        return x * self.resolution, y * self.resolution

    # ---------- build fields ----------
    def make_fields(self):
        """
        Returns a dict with:
          'occ'  : (H,W) uint8, 0=free, 1=occupied
          'esdf' : (H,W) float, signed meters (+free, -inside)
          'res'  : meters per cell (float)
        """
        rng = np.random.default_rng(self.seed)
        occ = np.zeros((self.height, self.width), dtype=np.uint8)

        def disk(r):
            y, x = np.ogrid[-r:r+1, -r:r+1]
            return (x*x + y*y) <= r*r

        blocked = np.zeros_like(occ, dtype=bool)
        sep = disk(self.spacing)

        # circles only (sparser map)
        Y = np.arange(self.height)[:, None]
        X = np.arange(self.width)[None, :]

        for _ in range(5):
            placed = False
            for _try in range(300):
                rad = int(rng.integers(8, 16))
                r = int(rng.integers(rad, self.height - rad))
                c = int(rng.integers(rad, self.width - rad))
                mask = (Y - r)**2 + (X - c)**2 <= rad*rad
                if (blocked & mask).any():
                    continue
                if not self.is_empty:
                    occ[mask] = 1
                blocked = binary_dilation(occ.astype(bool), structure=sep)
                placed = True
                break
            if not placed:
                pass

        # ESDF: positive outside obstacles, negative inside
        occ_b = occ.astype(bool)
        d_out = distance_transform_edt(~occ_b) * self.resolution
        d_in  = distance_transform_edt( occ_b) * self.resolution
        esdf = d_out - d_in

        return {"occ": occ, "esdf": esdf, "res": float(self.resolution)}


    # ---------- display ----------
    def display_fields(
        self,
        paths=None,                      # iterable of Path or list-of-(x,y)
        show_points_in_paths = True,        # Showing the point nodes in each path
        mark_xy_list=None,                    # [(x,y)] meters: draw a circle center at each
        mark_radius_m=3,              # circle radius (meters)
        arrow_start_xy=None,             # (x,y) meters: where arrow starts
        arrow_dir_xy=None,               # (dx,dy) direction (normalized inside)
        arrow_len_m=10,                 # arrow length (meters)
        figsize=(10, 4),
        acumulate_displays=False       # if i want to display several at a time. then i have to saay plt.show at the end
    ):
        """
        path_xy:      list of (x,y) in METERS (origin=(0,0))
        mark_xy:      (x,y) in METERS – draw a circle here (set None to skip)
        mark_radius_m:circle radius in METERS
        arrow_start_xy:(x,y) in METERS – where the arrow starts (set None to skip)
        arrow_dir_xy: (dx,dy) direction in METERS or arbitrary units (will normalize)
        arrow_len_m:  length of the arrow in METERS
        """

        occ, esdf, res = self.fields["occ"], self.fields["esdf"], self.fields["res"]

        fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

        # occupancy
        axes[0].imshow(occ, origin='lower', cmap='gray_r', interpolation='nearest')
        axes[0].set_title("Occupancy (1 = occupied)")
        axes[0].set_xlabel("x (cells)")
        axes[0].set_ylabel("y (cells)")
        axes[0].set_xlim(0, occ.shape[1])
        axes[0].set_ylim(0, occ.shape[0])
        axes[0].set_aspect('equal')

        # esdf
        im = axes[1].imshow(esdf, origin='lower', cmap='coolwarm', interpolation='nearest')
        axes[1].set_title("ESDF (m): +free, −inside")
        axes[1].set_xlabel("x (cells)")
        axes[1].set_xlim(0, esdf.shape[1])
        axes[1].set_ylim(0, esdf.shape[0])
        axes[1].set_aspect('equal')
        fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label="meters")

        # helper: convert meters → cells
        def m_to_cells_xy(xy):
            x, y = xy
            return (x / res, y / res)


        # mark a point with a circle (same on both panels)
        for mark_xy in mark_xy_list:
            if mark_xy is not None:
                mx_c, my_c = m_to_cells_xy(mark_xy)
                rad_cells = mark_radius_m / res
                circ0 = Circle((mx_c, my_c), rad_cells, fill=False, linewidth=2)
                circ1 = Circle((mx_c, my_c), rad_cells, fill=False, linewidth=2)
                axes[0].add_patch(circ0)
                axes[1].add_patch(circ1)
                # also dot in the center for clarity
                axes[0].plot([mx_c], [my_c], marker='o')
                axes[1].plot([mx_c], [my_c], marker='o')

        # arrow from a start point in a given direction
        if (arrow_start_xy is not None) and (arrow_dir_xy is not None):
            sx_c, sy_c = m_to_cells_xy(arrow_start_xy)
            dx, dy = arrow_dir_xy
            # normalize the direction; if zero-vector, skip
            norm = (dx*dx + dy*dy) ** 0.5
            if norm > 0:
                ux, uy = dx / norm, dy / norm
                dx_cells = (ux * arrow_len_m) / res
                dy_cells = (uy * arrow_len_m) / res
                for ax in axes:
                    ax.arrow(
                        sx_c, sy_c, dx_cells, dy_cells,
                        length_includes_head=True,
                        head_width=1.5, head_length=2.5, linewidth=2
                    )
        


        # Drawing the paths:

        # NEW: small knobs for showing waypoints
        point_size   = 12     # dot size

        # helper: accept Path OR list-of-(x,y)
        def _points_from(path_like):
            if path_like is None:
                return None
            if hasattr(path_like, "path"):   # duck-typing a Path object
                return path_like.path

            return path_like                 # assume already an iterable of (x,y)

        # Collect all paths to draw
        all_paths = []
        if paths is not None:
            for p in paths:
                all_paths.append(_points_from(p))

        # Draw paths
        for pts in all_paths:
            if not pts:
                continue
            xs_cells = [x / res for (x, _y) in pts]
            ys_cells = [y / res for (_x, y) in pts]
            axes[0].plot(xs_cells, ys_cells, linewidth=2)
            axes[1].plot(xs_cells, ys_cells, linewidth=2)

            # Dots on each waypoint (both panels)
            if show_points_in_paths:
                axes[0].scatter(xs_cells, ys_cells, s=point_size, zorder=3)
                axes[1].scatter(xs_cells, ys_cells, s=point_size, zorder=3)

        if not acumulate_displays:
            plt.show()


    @staticmethod
    def make_straight_path(p0, p1, n=100):
        """Return n points along the straight line between p0=(x0,y0) and p1=(x1,y1) in meters."""
        x0, y0 = p0
        x1, y1 = p1
        xs = np.linspace(x0, x1, n)
        ys = np.linspace(y0, y1, n)
        return list(zip(xs, ys))
    
    def is_of_grid(self, point_xy):
        int_xy = (int(point_xy[0]), int(point_xy[1]))
        return int_xy[0] < 0 or int_xy[1] < 0 or int_xy[0] > self.width or int_xy[1] > self.height

    # ---------- distance & direction at a point (cells or meters) ----------
    def dist_to_closest(self, point_xy, in_meters=False):
        """
        ESDF value (meters) at a point.
        point_xy is (x,y). If in_meters=True, interpreted in meters; otherwise in cells.
        """
        if in_meters:
            x_c, y_c = self._meters_to_cells(point_xy)
        else:
            x_c, y_c = point_xy
        r, c = _xy_to_rc((np.floor(x_c), np.floor(y_c)))
        if 0 <= r < self.esdf.shape[0] and 0 <= c < self.esdf.shape[1]:
            return float(self.esdf[r, c])
        return -np.inf
    
    def extract_3x3_at_xy(self, point_xy):
        """
        Return a 3x3 numpy array of ESDF values around point_xy = (x,y) in meters.
        If near the border, it will pad with np.inf (so you don't crash).
        """
        # convert (x,y in meters) -> (row, col)
        x, y = point_xy
        c = int(np.floor(x / self.resolution))
        r = int(np.floor(y / self.resolution))

        # make an empty 3x3 with padding
        grid_3x3 = np.full((3, 3), np.inf, dtype=float)

        for dr in range(-1, 2):
            for dc in range(-1, 2):
                rr, cc = r + dr, c + dc
                if 0 <= rr < self.esdf.shape[0] and 0 <= cc < self.esdf.shape[1]:
                    grid_3x3[dr+1, dc+1] = self.esdf[rr, cc]

        return grid_3x3


        # gradient arrays: dy/dr and dx/dc; pass spacing to get meters scaling
    def gradient_at_xy(self, point_xy, in_meters: bool = True):
        """
        ESDF gradient at (x,y). Returns (d/dx, d/dy).
        - in_meters=True  -> units are per meter (dx=dy=self.resolution)
        - in_meters=False -> units are per cell   (dx=dy=1)
        """
        dx = dy = self.resolution if in_meters else 1.0
        patch = self.extract_3x3_at_xy(point_xy)     # centered 3×3 around (x,y)
        gx, gy = gradient_3x3(patch, dx, dy)         # two scalars
        return float(gx), float(gy)





    def dir_to_closest(self, point_xy, in_meters=True, eps=1e-9):
        """
        Unit vector pointing toward the nearest obstacle from (x,y),
        which is approximately the **negative** ESDF gradient direction.
        Returns (dx, dy) as a numpy array.
        """

        gx, gy = self.gradient_at_xy(point_xy, in_meters=in_meters)
        vx, vy = -gx, -gy # the gradients naturaly points towards the highest values (we want it to point to the obstacle aka negative dir)
        n = (vx*vx + vy*vy)**0.5
        if n < eps:
            return (0.0, 0.0)
        return np.array([vx/n, vy/n])




if __name__ == "__main__":
    field_obj = Field(height=100, width=120, seed=5, spacing=14, resolution=1)

    fields = field_obj.fields  # occ/esdf/res
    print("shapes:", fields["occ"].shape, fields["esdf"].shape)

    # straight path in METERS
    path = Field.make_straight_path((20, 30), (80, 40), n=150)

    # distance / direction at a specific point (meters)
    d = field_obj.dist_to_closest((80, 40), in_meters=True)
    ux, uy = field_obj.dir_to_closest((80, 40), in_meters=True)
    print("esdf distance:", d, "dir to closest obstacle:", (ux, uy))


    p1 = Path([(5,5), (10,7), (20,11), (40, 30)])
    p2 = [(5,25), (15,22), (22,18), (30,12), (40, 35)]

    q = (2.0, 1.2)                               # mark this point
    u = field_obj.dir_to_closest(q, in_meters=True)  # arrow direction (unit)

    print(field_obj.is_of_grid((161, 121)))

    field_obj.display_fields(
        paths=[p1, p2],
        mark_xy_list=[q], mark_radius_m=0.3,
        arrow_start_xy=q, arrow_dir_xy=u, arrow_len_m=0.6
    )
