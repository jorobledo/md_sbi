from simulator import run_md
from sbi_workflow import plot_sim
import numpy as np

if __name__ == "__main__":
    params = (9.8985461e02, 2.6856201e-02, 2.7241945e00)
    t0 = 0.0
    dt = 0.1
    t_steps = 4000
    x = np.array([1, 5, 10])
    t = np.linspace(t0, t0 + (dt * t_steps), t_steps, endpoint=False)
    x_est = run_md(t[1] - t[0], len(t), x, *params)

    plot_sim(t, x_est, "estimated")
