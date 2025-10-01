import numpy as np







# For solving -x_(i-1) + 2*x_i - x_(i+1) = 
def Thomas_spesial(x, RHS):
    """
    Løyser -Δx[i-1]+2Δx[i]-Δx[i+1] = obs_i[i], i=1..N-2, med Δx[0]=Δx[N-1]=0
    x: (N,d) eller (N,), obs_i same shape
    """
    x = np.zeros_like(x, dtype=float)
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

    x[N-1] = 0.0  # Dirichlet
    for i_0 in range(N-1):
        i = N - 2 - i_0
        x[i] = d[i] - c[i] * x[i+1]
    x[0] = 0.0    # Dirichlet
    return  x




def Thomas(RHS, sub: np.ndarray, sup: np.ndarray, diag: np.ndarray):
    """
    Løyser A * x = RHS, der A har
    sub-, main- og super-diagonal: `sub`, `diag`, `sup`.

    Returnerer løysinga x.

    """
    RHS = np.asarray(RHS, dtype=float)
    sub = np.asarray(sub, dtype=float)
    sup = np.asarray(sup, dtype=float)
    diag = np.asarray(diag, dtype=float)

    # For å generalisere slik at ein kan passe det til punkt av fleire dimensjonar
    if RHS.ndim == 1:
        RHS = RHS[:, None]
        squeeze_out = True
    elif RHS.ndim == 2:
        squeeze_out = False
    else:
        raise ValueError("RHS må vere 1D eller 2D (N,).")

    N = RHS.shape[0]
    if diag.shape != (N,):
        raise ValueError(f"main diagonal må ha form (N,), fekk {diag.shape} kor N={N}")
    if sub.shape != (N-1,):
        raise ValueError(f"sub-diagonalen må ha størelse (N-1,) {sup.shape} kor N={N}")
    if sup.shape != (N-1,):
        raise ValueError(f"super-diagonalen må ha størelse (N-1,) {sup.shape} kor N={N}")

    d = RHS.copy()
    c = np.zeros(N)

    # start verdiar
    denom = diag[0]
    if np.isclose(denom, 0.0):
        raise ZeroDivisionError("delar på null")
    c[0] = sup[0] / denom if N > 1 else 0.0
    d[0, :] = d[0, :] / denom

    # Midtverdiar
    for i in range(1, N):
        denom = diag[i] - sub[i-1] * c[i-1]
        if np.isclose(denom, 0.0):
            raise ZeroDivisionError(f"oh nooooooooo devide by zero")
        if i < N - 1:
            c[i] = sup[i] / denom
        d[i, :] = (d[i, :] - sub[i-1] * d[i-1, :]) / denom

    # Sveipar tilbake og finn x
    x = np.zeros_like(d)
    x[-1, :] = d[-1, :]
    for i in range(N - 2, -1, -1):
        x[i, :] = d[i, :] - c[i] *   x[i + 1, :]

    if squeeze_out:
        return  x[:, 0]
    return  x







# These tests are reliable, chat did it
# ---- Helper to build dense tridiagonal matrix ---- 
def build_tridiag(sub, diag, sup):
    N = diag.size
    A = np.zeros((N, N), dtype=float)
    A[np.arange(N), np.arange(N)] = diag
    A[np.arange(1, N), np.arange(N-1)] = sub
    A[np.arange(N-1), np.arange(1, N)] = sup
    return A

# ---- Manual tests ----
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # Test 1: simple Poisson matrix with Dirichlet boundaries
    N = 6
    sub = -np.ones(N-1)
    sup = -np.ones(N-1)
    diag = 2*np.ones(N)
    # Impose Dirichlet: first/last rows
    diag[0] = diag[-1] = 1
    sub[-1] = 0
    sup[0] = 0
    RHS = np.arange(N, dtype=float)

    A = build_tridiag(sub, diag, sup)
    ref = np.linalg.solve(A, RHS)
    sol = Thomas(RHS, sub, sup, diag)
    assert np.allclose(sol, ref)

    # Test 2: random diagonally dominant system, single RHS
    N = 8
    sub = rng.normal(size=N-1)
    sup = rng.normal(size=N-1)
    diag = np.abs(rng.normal(size=N)) + 3.0
    RHS = rng.normal(size=N)
    A = build_tridiag(sub, diag, sup)
    ref = np.linalg.solve(A, RHS)
    sol = Thomas(RHS, sub, sup, diag)
    assert np.allclose(sol, ref)

    # Test 3: multiple RHS
    N, d = 10, 3
    sub = -np.ones(N-1)
    sup = -np.ones(N-1)
    diag = 2*np.ones(N)
    RHS = rng.normal(size=(N, d))
    A = build_tridiag(sub, diag, sup)
    ref = np.linalg.solve(A, RHS)
    sol = Thomas(RHS, sub, sup, diag)
    assert np.allclose(sol, ref)

    # Test 4: shape round trip
    RHS1 = np.arange(5, dtype=float)
    RHS2 = np.vstack([RHS1, RHS1+1]).T
    sub = -np.ones(4)
    sup = -np.ones(4)
    diag = 2*np.ones(5)
    diag[0] = diag[-1] = 1
    sup[0] = sub[-1] = 0
    sol1 = Thomas(RHS1, sub, sup, diag)
    sol2 = Thomas(RHS2, sub, sup, diag)
    assert sol1.shape == (5,)
    assert sol2.shape == (5, 2)

    print("All tests passed!")