import numpy as np
from field import Field
from path import Path
from Thomas_alg import Thomas
import copy



def chomp(
    field,
    start_path,
    weight_obst: float,
    weight_smooth: float,
    epsilon: float = 20,      # obstacle influence distance
    alpha: float = 10,        # step size
    max_iters: int = 80,
    tol: float = 1e-4,
):
    """
    CHOMP-style trajectory optimization using the user's Thomas solver.

    Solves per iteration: A Δx = -g, with A from the discrete second-difference
    stencil [-1, 2, -1], then x <- x + α Δx.
    """

    path = copy.deepcopy(start_path)
    X = path.to_np_array().astype(float)   # (N, d)
    N, d = X.shape

    def smooth_grad(X):
        """g_s[i] = 2*X[i] - X[i-1] - X[i+1] on interior; 0 at endpoints."""
        g = np.zeros_like(X)
        g[1:-1] = 2*X[1:-1] - X[:-2] - X[2:]
        return g

    def obstacle_grad(p):
        """
        Hinge on clearance:
          C = 0.5 * max(0, ε - c)^2
          ∇C = -(ε - c)_+ * n_hat,
        where n_hat points OUTWARD (increasing clearance).
        field.dir_to_closest points TOWARD obstacle, so outward is -dir.
        """
        c = float(field.dist_to_closest(point_xy=p))
        if c >= epsilon:
            return np.zeros_like(p)
        # outward normal (unit)
        n_toward = np.array(field.dir_to_closest(point_xy=p), dtype=float)
        norm = np.linalg.norm(n_toward)
        if norm == 0:
            return np.zeros_like(p)
        n_out = -n_toward / norm
        return -(epsilon - c) * n_out

    for _ in range(max_iters):
        g_s = smooth_grad(X)
        g_o = np.zeros_like(X)
        for i in range(1, N-1):         # keep endpoints fixed (no obstacle gradient there)
            g_o[i] = obstacle_grad(X[i])

        g = weight_smooth * g_s + weight_obst * g_o

        # Enforce fixed endpoints on RHS explicitly (your solver also enforces Δx endpoints = 0)
        rhs = -g
        rhs[0] = 0.0
        rhs[-1] = 0.0

        # Solve A ΔX = rhs with your Thomas, vectorized across dims
        dX = Thomas(X, rhs)

        # Update interior only
        X_new = X.copy()
        X_new[1:-1] += alpha * dX[1:-1]

        # Convergence check on the step
        step_norm = np.linalg.norm((X_new - X)[1:-1], ord=np.inf)
        X = X_new
        if step_norm < tol:
            break

    return Path(list(X))