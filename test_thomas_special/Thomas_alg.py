import numpy as np


def Thomas(x, RHS):
    """
    Løyser -Δx[i-1]+2Δx[i]-Δx[i+1] = obs_i[i], i=1..N-2, med Δx[0]=Δx[N-1]=0
    x: (N,d) eller (N,), obs_i same shape
    """
    dx = np.zeros_like(x, dtype=float)
    c  = np.zeros_like(x, dtype=float)  # bruker vektorform; same c per komponent
    d  = np.zeros_like(x, dtype=float)

    N = len(x)

    for i in range(N):
        if i == 0:
            c[0] = 0
            d[0] = 0
            continue
        if i == 1:
            c[1] = -0.5
            d[1] = RHS[1] / 2.0
            continue
        if i == N - 2:
            c[i] = 0
            d[i] = (RHS[i] + d[i-1]) / (2.0 + c[i-1])
            continue
        if i == N - 1:
            c[i] = 0
            d[i] = 0
            continue

        c[i] = -1.0 / (2.0 + c[i-1])
        d[i] = (RHS[i] + d[i-1]) / (2.0 + c[i-1])

    dx[N-1] = 0.0  # Dirichlet
    for i_0 in range(N-1):
        i = N - 2 - i_0
        dx[i] = d[i] - c[i] * dx[i+1]
    dx[0] = 0.0    # Dirichlet
    return dx







# Testing

def build_A(N: int) -> np.ndarray:
    A = np.zeros((N, N), dtype=float)
    A[0, 0] = 1.0
    A[-1, -1] = 1.0
    for i in range(1, N-1):
        A[i, i] = 2.0
        A[i, i-1] = -1.0
        A[i, i+1] = -1.0
    
    # print(np.array_str(A, precision=0, suppress_small=True)) # For debuging

    return A


def run_test(N, d=1):
    rng = np.random.default_rng(42 + N + d)
    if d == 1:
        RHS = rng.normal(size=N)
    else:
        RHS = rng.normal(size=(N, d))
    RHS[0] = 0.0
    RHS[-1] = 0.0

    # Direct solution
    A = build_A(N)
    x_direct = np.linalg.solve(A, RHS)

    # Thomas solution
    x0 = np.zeros_like(RHS)
    x_thomas = Thomas(x0, RHS)

    # Compare
    ok = np.allclose(x_thomas, x_direct, rtol=1e-12, atol=1e-12)
    print(f"N={N}, d={d}: {'PASS' if ok else 'FAIL'}")
    if not ok:
        print("  Thomas:", x_thomas)
        print("  Direct:", x_direct)


if __name__ == "__main__":
    # Smallest meaningful system
    RHS = np.array([0.0, 1.0, 0.0])
    x0 = np.zeros_like(RHS)
    x_thomas = Thomas(x0, RHS)
    print("Manual check N=3:", x_thomas)  # expected [0, 0.5, 0]

    # Run a few randomized tests
    for N in [5, 10, 25]:
        run_test(N, d=1)

    for N, d in [(6, 2), (8, 3)]:
        run_test(N, d)