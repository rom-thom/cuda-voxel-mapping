
import numpy as np



class Path:
    def __init__(self, array):
        """
        array = a list of points_xy tht is the path to follow
        """
        self.path: list = array

    def add_point(self, point_xy):
        self.path.append(point_xy)

    def pop(self):
        if len(self.path) == 0:
            raise IndexError("Couldnt pop as it was the last element")
        
        return self.path.pop()


    @staticmethod
    def line_path(start_xy: tuple[float, float], end_xy: tuple[float, float], n: int) -> 'Path':
        """
            Straight path from start to end with n number of lines between start and end. Aka n+1 points
        """
        x0, y0 = start_xy
        x1, y1 = end_xy
        if n < 1:
            raise ValueError("must have at least a line between")
        points: list[tuple[float, float]] = [
            (x0 + (x1 - x0) * i / (n),
            y0 + (y1 - y0) * i / (n))
            for i in range(n + 1)
        ]
        return Path(points)

    @staticmethod
    def line_path_spacing(start_xy: tuple[float, float],
                        end_xy: tuple[float, float],
                        spacing: float,
                        include_end: bool = True) -> 'Path':
        """
        Straight path from start to end with points spaced ~'spacing' apart (in same units as start/end).
        Adds the exact end point if include_end=True (last segment may be shorter).
        """
        if spacing <= 0:
            raise ValueError("spacing must be > 0")

        x0, y0 = start_xy
        x1, y1 = end_xy
        dx, dy = x1 - x0, y1 - y0
        dist = np.sqrt(dx**2 + dy**2)

        # Degenerate: start == end
        if dist <= spacing:
            return Path([start_xy] if not include_end else [start_xy, end_xy])

        # Unit direction
        ux, uy = dx / dist, dy / dist

        # Number of full spacing steps (not overshooting the end)
        n_steps = int(dist // spacing)  # floor

        points = []
        for i in range(n_steps + 1):  # includes start at i=0
            s = i * spacing
            px = x0 + ux * s
            py = y0 + uy * s
            points.append((px, py))

        # Append exact end point if needed
        if include_end:
            points.append((x1, y1))

        return Path(points)
