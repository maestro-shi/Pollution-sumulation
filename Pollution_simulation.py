import math
import numpy as np

L = 300.0

BUILDINGS = [
    {"name": "100_long", "x_min": 69, "x_max": 87, "y_min": 12, "y_max": 90},
    {"name": "100_short", "x_min": 87, "x_max": 99, "y_min": 12, "y_max": 30},
    {"name": "102", "x_min": 138, "x_max": 156, "y_min": 12, "y_max": 30},
    {"name": "104_long", "x_min": 192, "x_max": 210, "y_min": 12, "y_max": 90},
    {"name": "104_short", "x_min": 210, "x_max": 222, "y_min": 12, "y_max": 30},
    {"name": "106", "x_min": 261, "x_max": 279, "y_min": 12, "y_max": 30},
    {"name": "108", "x_min": 69, "x_max": 87, "y_min": 111, "y_max": 129},
    {"name": "110_long", "x_min": 138, "x_max": 156, "y_min": 51, "y_max": 129},
    {"name": "110_short", "x_min": 126, "x_max": 138, "y_min": 111, "y_max": 129},
    {"name": "112", "x_min": 192, "x_max": 210, "y_min": 111, "y_max": 129},
    {"name": "114_long", "x_min": 261, "x_max": 279, "y_min": 51, "y_max": 129},
    {"name": "114_short", "x_min": 249, "x_max": 261, "y_min": 111, "y_max": 129},
    {"name": "116_long", "x_min": 69, "x_max": 87, "y_min": 165, "y_max": 243},
    {"name": "116_short", "x_min": 87, "x_max": 99, "y_min": 165, "y_max": 183},
    {"name": "118", "x_min": 138, "x_max": 156, "y_min": 165, "y_max": 183},
    {"name": "120_long", "x_min": 192, "x_max": 210, "y_min": 165, "y_max": 243},
    {"name": "120_short", "x_min": 210, "x_max": 222, "y_min": 165, "y_max": 183},
    {"name": "122", "x_min": 261, "x_max": 279, "y_min": 165, "y_max": 183},
    {"name": "124", "x_min": 69, "x_max": 87, "y_min": 264, "y_max": 282},
    {"name": "126_long", "x_min": 138, "x_max": 156, "y_min": 204, "y_max": 282},
    {"name": "126_short", "x_min": 126, "x_max": 138, "y_min": 264, "y_max": 282},
    {"name": "128", "x_min": 192, "x_max": 210, "y_min": 264, "y_max": 282},
    {"name": "130_long", "x_min": 261, "x_max": 279, "y_min": 204, "y_max": 282},
    {"name": "130_short", "x_min": 249, "x_max": 261, "y_min": 264, "y_max": 282},
]


def read_input():
    with open("input.txt", "r", encoding="utf-8") as f:
        data = f.read().split()
    if len(data) < 7:
        raise ValueError("input.txt must contain: M eps g1 lambda1 lambda2 kappa T")
    return (
        int(data[0]), float(data[1]), float(data[2]), float(data[3]),
        float(data[4]), float(data[5]), float(data[6]),
    )


def build_solid_mask(M, h):
    mask_xy = np.zeros((M, M), dtype=bool)
    for building in BUILDINGS:
        i0 = max(0, min(M - 1, int(round(building["x_min"] / h))))
        i1 = max(0, min(M - 1, int(round(building["x_max"] / h))))
        j0 = max(0, min(M - 1, int(round(building["y_min"] / h))))
        j1 = max(0, min(M - 1, int(round(building["y_max"] / h))))
        mask_xy[i0:i1 + 1, j0:j1 + 1] = True
    return mask_xy.T


def apply_bc(u, g1):
    u[:, 0] = g1
    u[:, -1] = u[:, -2]
    u[0, :] = u[1, :]
    u[-1, :] = u[-2, :]
    u[:, 0] = g1


def enforce_physics(u, solid, g1):
    u[solid] = 0.0
    apply_bc(u, g1)


def precompute_segments(solid_2d, dirichlet_at_left):
    groups = {}
    m = solid_2d.shape[0]
    for line_idx in range(m):
        solid_line = solid_2d[line_idx]
        i = 0
        while i < m:
            if solid_line[i]:
                i += 1
                continue
            start = i
            while i < m and not solid_line[i]:
                i += 1
            end = i
            left_dirichlet = dirichlet_at_left and start == 0
            key = (end - start, left_dirichlet)
            if key not in groups:
                groups[key] = []
            groups[key].append((line_idx, start))
    result = {}
    for key, values in groups.items():
        arr = np.asarray(values, dtype=np.int64)
        result[key] = (arr[:, 0], arr[:, 1])
    return result


def make_tridiagonal(n, r, left_dirichlet):
    a = np.zeros(n, dtype=np.float64)
    b = np.full(n, 1.0 + 2.0 * r, dtype=np.float64)
    c = np.zeros(n, dtype=np.float64)
    if n == 1:
        b[0] = 1.0
        return a, b, c
    a[1:] = -r
    c[:-1] = -r
    if left_dirichlet:
        b[0] = 1.0
        c[0] = 0.0
    else:
        b[0] = 1.0 + r
    b[-1] = 1.0 + r
    c[-1] = 0.0
    return a, b, c


def thomas_batch(a, b, c, d):
    batch, n = d.shape
    cp = np.zeros((batch, n), dtype=np.float64)
    dp = np.zeros((batch, n), dtype=np.float64)
    cp[:, 0] = c[0] / b[0]
    dp[:, 0] = d[:, 0] / b[0]
    for i in range(1, n):
        denom = b[i] - a[i] * cp[:, i - 1]
        cp[:, i] = c[i] / denom if i + 1 < n else 0.0
        dp[:, i] = (d[:, i] - a[i] * dp[:, i - 1]) / denom
    x = np.empty_like(d)
    x[:, -1] = dp[:, -1]
    for i in range(n - 2, -1, -1):
        x[:, i] = dp[:, i] - cp[:, i] * x[:, i + 1]
    return x


def adi_sweep(rhs, segment_groups, r, g1):
    result = np.zeros_like(rhs)
    for (n, left_dirichlet), (lines, starts) in segment_groups.items():
        offsets = np.arange(n)
        cols = starts[:, None] + offsets[None, :]
        rows = lines[:, None]
        d = rhs[rows, cols].copy()
        if left_dirichlet:
            d[:, 0] = g1
        a, b, c = make_tridiagonal(n, r, left_dirichlet)
        x = thomas_batch(a, b, c, d)
        result[rows, cols] = x
    return result


def convection_upwind_formula(u, tau, h, lambda1, lambda2, solid, g1):
    ax = abs(lambda1) * tau / h
    ay = abs(lambda2) * tau / h
    if ax + ay > 1.0 + 1e-12:
        raise ValueError("CFL condition failed")

    x_up = np.empty_like(u)
    x_up_solid = np.zeros_like(solid)
    if lambda1 >= 0.0:
        x_up[:, 1:] = u[:, :-1]; x_up[:, 0] = g1
        x_up_solid[:, 1:] = solid[:, :-1]
    else:
        x_up[:, :-1] = u[:, 1:]; x_up[:, -1] = u[:, -1]
        x_up_solid[:, :-1] = solid[:, 1:]

    y_up = np.empty_like(u)
    y_up_solid = np.zeros_like(solid)
    if lambda2 >= 0.0:
        y_up[1:, :] = u[:-1, :]; y_up[0, :] = u[0, :]
        y_up_solid[1:, :] = solid[:-1, :]
    else:
        y_up[:-1, :] = u[1:, :]; y_up[-1, :] = u[-1, :]
        y_up_solid[:-1, :] = solid[1:, :]

    diagonal = np.empty_like(u)
    diagonal_solid = np.zeros_like(solid)
    if lambda1 >= 0.0:
        if lambda2 >= 0.0:
            diagonal[:-1, 1:] = u[1:, :-1]
            diagonal[-1:, :] = u[-1:, :]
            diagonal[:, :1] = u[:, :1]
            diagonal_solid[:-1, 1:] = solid[1:, :-1]
        else:
            diagonal[1:, 1:] = u[:-1, :-1]
            diagonal[:1, :] = u[:1, :]
            diagonal[:, :1] = u[:, :1]
            diagonal_solid[1:, 1:] = solid[:-1, :-1]
    else:
        if lambda2 >= 0.0:
            diagonal[:-1, :-1] = u[1:, 1:]
            diagonal[-1:, :] = u[-1:, :]
            diagonal[:, -1:] = u[:, -1:]
            diagonal_solid[:-1, :-1] = solid[1:, 1:]
        else:
            diagonal[1:, :-1] = u[:-1, 1:]
            diagonal[:1, :] = u[:1, :]
            diagonal[:, -1:] = u[:, -1:]
            diagonal_solid[1:, :-1] = solid[:-1, 1:]

    x_up = np.where(x_up_solid & ~diagonal_solid, diagonal, x_up)
    x_up = np.where(x_up_solid & diagonal_solid, u, x_up)
    y_up = np.where(y_up_solid, u, y_up)

    u_new = (1.0 - ax - ay) * u + ax * x_up + ay * y_up
    enforce_physics(u_new, solid, g1)
    return u_new


def diffusion_adi(u, tau, h, kappa, solid, row_segments, col_segments, g1):
    r = kappa * tau / (h * h)
    if r == 0.0:
        out = u.copy()
        enforce_physics(out, solid, g1)
        return out

    u_down = np.empty_like(u); u_up = np.empty_like(u)
    u_down[1:, :] = u[:-1, :]; u_down[0, :] = u[0, :]
    u_up[:-1, :] = u[1:, :];   u_up[-1, :] = u[-1, :]
    down_solid = np.zeros_like(solid); up_solid = np.zeros_like(solid)
    down_solid[1:, :] = solid[:-1, :]; up_solid[:-1, :] = solid[1:, :]
    u_down = np.where(down_solid, u, u_down)
    u_up = np.where(up_solid, u, u_up)
    rhs1 = u + r * (u_down - 2.0 * u + u_up)

    u_star = adi_sweep(rhs1, row_segments, r, g1)
    enforce_physics(u_star, solid, g1)

    u_left = np.empty_like(u_star); u_right = np.empty_like(u_star)
    u_left[:, 1:] = u_star[:, :-1]; u_left[:, 0] = g1
    u_right[:, :-1] = u_star[:, 1:]; u_right[:, -1] = u_star[:, -1]
    left_solid = np.zeros_like(solid); right_solid = np.zeros_like(solid)
    left_solid[:, 1:] = solid[:, :-1]; right_solid[:, :-1] = solid[:, 1:]
    u_left = np.where(left_solid, u_star, u_left)
    u_right = np.where(right_solid, u_star, u_right)
    rhs2 = u_star + r * (u_left - 2.0 * u_star + u_right)

    u_new = adi_sweep(rhs2.T, col_segments, r, g1).T
    enforce_physics(u_new, solid, g1)
    return u_new


def output_norm(u, solid):
    air = ~solid
    n_air = int(np.count_nonzero(air))
    if n_air == 0:
        return 0.0, 0
    return float(math.sqrt(np.sum(u[air] ** 2)) / n_air), n_air


def format_number(value):
    rounded = round(value)
    if abs(value - rounded) < 1e-12:
        return str(int(rounded))
    return repr(float(value))


def format_output(u, solid, h):
    norm, n_air = output_norm(u, solid)
    m = u.shape[0]
    lines = [repr(float(norm)), str(n_air)]
    for i in range(m):
        x = i * h
        for j in range(m):
            if not solid[j, i]:
                lines.append(
                    f"{format_number(x)} {format_number(j * h)} {repr(float(u[j, i]))}"
                )
    return "\n".join(lines) + "\n"


def solve(M, eps, g1, lambda1, lambda2, kappa, T):
    h = L / (M - 1)
    solid = build_solid_mask(M, h)
    row_segments = precompute_segments(solid, dirichlet_at_left=True)
    col_segments = precompute_segments(solid.T, dirichlet_at_left=False)

    u = np.zeros((M, M), dtype=np.float64)
    apply_bc(u, g1)
    u[solid] = 0.0

    lambda_max = max(abs(lambda1), abs(lambda2))
    if lambda_max < 2.0 * math.pi:
        tau = h / (2.0 * math.pi)
    else:
        tau = h / lambda_max

    lambda_sum = abs(lambda1) + abs(lambda2)
    if lambda_sum * tau / h > 1.0 + 1e-12:
        tau = h / lambda_sum

    current_time = 0.0
    max_steps = int(math.ceil(T / tau)) + 10

    for _ in range(max_steps):
        if current_time >= T:
            break
        dt = min(tau, T - current_time)
        if dt <= 0.0:
            break
        u = diffusion_adi(u, dt, h, kappa, solid, row_segments, col_segments, g1)
        u = convection_upwind_formula(u, dt, h, lambda1, lambda2, solid, g1)
        current_time += dt

    return u, solid, h


def main():
    params = read_input()
    u, solid, h = solve(*params)
    answer = format_output(u, solid, h)
    with open("output.txt", "w", encoding="utf-8") as f:
        f.write(answer)

    try:
        import matplotlib.pyplot as plt
        masked = np.ma.masked_where(solid, u)
        plt.figure(figsize=(6, 6))
        plt.imshow(masked, origin="lower", cmap="viridis")
        plt.colorbar(label="u")
        plt.title("Solution u (buildings masked)")
        plt.savefig("picture.png", dpi=150, bbox_inches="tight")
        plt.close()
    except Exception:
        pass
main()
