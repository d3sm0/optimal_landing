from scipy.integrate import odeint
import numpy as np
import pandas
from deep_control import nn

def integrate_landing(dy, networks, compute_control, initial_state, final_check, dt=0.01, max_time=10,
                      stop_if_done=True, stop_if_crash=True, col_names=None):

    state_history = [initial_state]
    ts = [0]
    t = [0, dt]
    control = [[0]*len(networks)]
    check = None
    state = initial_state

    for i in range(int(max_time/dt)):
        state = odeint(dy, state, t, printmessg=False, rtol=10e-12, atol=10e-12, args=(networks, compute_control))[1, :]

        ts.append(ts[-1]+t[1])
        u = compute_control(state,networks)
        control.append(u)

        state_history.append(state)
        c = final_check(state)
        if 'crash' == c:
            if not check:
                check = 'crash'
            if stop_if_crash:
                break

        elif 'done' == c:
            if not check:
                check = 'done'
            if stop_if_done:
                break

    state_history = np.vstack(state_history)
    ts = np.asarray(ts).reshape(-1, 1)
    state_history = np.hstack((ts, state_history, control))
    state_history = pandas.DataFrame(state_history)

    if col_names:
        state_history.columns =  col_names
    return state_history, check

