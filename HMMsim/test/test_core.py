import pytest
import numpy as np
import numpy.typing as npt


def generate_lin(pi: npt.NDArray[np.float64], T: npt.NDArray[np.float64], ts, rng):
    Gs = np.zeros((ts.size, 2), dtype=int)

    # Generate lineage list
    cell_state = rng.choice(pi.size, p=pi)

    t_cur = 0.0

    # Handle generation 1 specially
    # bern_obs = rng.binomial(1, p=0.01, size=1)  # 1 indicates death
    # gamma_obs = rng.gamma(5.0, scale=3.0, size=1)  # gamma observations

    bern_p = [0.99548085, 0.99539826]
    gamma_shape = [8.90923167, 28.69267082]
    gamma_scale = [2.29617727, 0.77380887]
    for gen in range(1, 20):
        for phase in range(2):
            bern_obs = rng.binomial(1, p=bern_p[phase], size=1)  # 1 indicates death
            gamma_obs = rng.gamma(gamma_shape[phase], scale=gamma_scale[phase], size=1)  # gamma observations

            idx = (ts > t_cur) & (ts < t_cur + gamma_obs)
            Gs[idx, phase] += 2 ** gen
            t_cur += gamma_obs

            if t_cur > ts[-1]:
                return Gs

            # if bern_obs:
            #     return Gs

        # cell_state = rng.choice(T.shape[0], p=T[cell_state, :])

    # return Gs

def test_initialize():
    """Test initializeParams() work correctly"""
    rng = np.random.default_rng(1)
    ts = np.arange(96, step=1.0)

    # pi = np.array([0.9, 0.1])
    pi = np.array([0.39569283, 0.151052, 0.1736583, 0.27959686])

    # T = np.array([[0.9, 0.1], [0.1, 0.9]])
    T =  np.array([[9.40936238e-01, 2.91397431e-02, 1.57051005e-02, 1.42189181e-02],
 [8.44410007e-02, 5.68472269e-04, 8.34060525e-01, 8.09300019e-02],
 [5.60224016e-03, 7.55455744e-01, 2.38507073e-01, 4.34942554e-04],
 [3.44898363e-02, 2.94873662e-02, 1.38300546e-04, 9.35884497e-01]])

    Gs = generate_lin(pi, T, ts, rng)

    for ii in range(300):
        Gs += generate_lin(pi, T, ts, rng)

    return Gs
