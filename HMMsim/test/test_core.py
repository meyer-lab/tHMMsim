import pytest
import numpy as np
import numpy.typing as npt


def generate_lin(pi: npt.NDArray[np.float64], T: npt.NDArray[np.float64], ts, rng):
    Gs = np.zeros((ts.size, 2), dtype=int)

    # Generate lineage list
    cell_state = rng.choice(pi.size, p=pi)

    t_cur = 0.0

    # Handle generation 1 specially
    bern_obs = rng.binomial(1, p=0.01, size=1)  # 1 indicates death
    gamma_obs = rng.gamma(5.0, scale=3.0, size=1)  # gamma observations
    bern_obs = rng.binomial(1, p=0.01, size=1)  # 1 indicates death
    gamma_obs = rng.gamma(5.0, scale=3.0, size=1)  # gamma observations


    for gen in range(1, 20):
        for phase in range(2):
            bern_obs = rng.binomial(1, p=0.01, size=1)  # 1 indicates death
            gamma_obs = rng.gamma(5.0, scale=3.0, size=1)  # gamma observations

            idx = (ts > t_cur) & (ts < t_cur + gamma_obs)
            Gs[idx, phase] += 2 ** gen
            t_cur += gamma_obs

            if t_cur > ts[-1]:
                return Gs

            if bern_obs:
                return Gs

        cell_state = rng.choice(T.shape[0], p=T[cell_state, :])


def test_initialize():
    """Test initializeParams() work correctly"""
    rng = np.random.default_rng(1)
    ts = np.arange(96, step=1.0)

    pi = np.array([0.9, 0.1])

    T = np.array([[0.9, 0.1], [0.1, 0.9]])

    Gs = generate_lin(pi, T, ts, rng)

    for ii in range(300):
        Gs += generate_lin(pi, T, ts, rng)

    print(Gs)
