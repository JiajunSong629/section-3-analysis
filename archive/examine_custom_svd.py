import numpy as np


def custom_svd(M):
    n, p = M.shape
    if n > p:
        V, S, Ut = custom_svd(M.T)
        return Ut.T, S, V.T

    G = M @ M.T
    S2, U = np.linalg.eigh(G)
    S = np.sqrt(np.abs(S2))

    sorted_indices = np.argsort(S)[::-1]
    S = S[sorted_indices]
    U = U[:, sorted_indices]

    V = M.T @ U @ np.diag(1 / S)

    return U, S, V.T


def case_1():
    dh = 5
    nh = 32
    dm = dh * nh  # 160
    K = 10  # 10 , dm, dh

    x = np.array([np.random.randn(dm, dh) @ np.random.randn(dh, dm) for _ in range(K)])
    x = x.reshape(-1, dm)  # x: (K*dm, dm), rank <= K*dh < nh * dh = dm

    u1, s1, vt1 = custom_svd(x)
    u2, s2, vt2 = np.linalg.svd(x)

    print("-" * 100)

    v1 = vt1.T
    v2 = vt2.T
    for i in range(v1.shape[1]):
        print(i, np.sum(v1[:, i]), np.sum(v2[:, i]))


# x = (1600, 160) = U S Vt, rank(x) = 50
# Vt[:, :50] correct

# x = (10 * 4096, 4096), rank(x) = 10 * dh = 10 * 128 = 1280
# rank <= 1000, correct

# x = (40960, 4906)
# xx = x.T @ x,  (4096, 4096) xx = U S2 U.T
case_1()
