"""#TODO."""

from scipy.integrate import odeint
import numpy as np
import pandas


def integrate_landing(dy, networks, compute_control, initial_state,
                      final_check, dt=0.01, max_time=10,
                      stop_if_done=True, stop_if_crash=True, col_names=None):
    """# TODO."""
    state_history = [initial_state]
    ts = [0]
    t = [0, dt]
    control = [[0]*len(networks)]
    check = None
    state = initial_state

    for i in range(int(max_time/dt)):
        state = odeint(dy, state, t, printmessg=False, rtol=10e-12,
                       atol=10e-12, args=(networks, compute_control))[1, :]

        ts.append(ts[-1]+t[1])
        u = compute_control(state, networks)
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
        state_history.columns = col_names
    return state_history, check


def evaluate_traj(nn_traj, opt_traj, value_err, targets=None,
                  norms=None, ):
    """Evaluate the NN-driven trajectory.

    Return:
        - minimum distance point
        - optimal difference to minimum distance point
    """
# TODO adapt for other models
    if not targets:
        targets = [0, 0, 0]
    if not norms:
        norms = [0, 0, 0, 0, 0]

    dist_to_goal = (norms[0]*(np.sqrt((nn_traj['x']-targets[0])**2 +
                                      (nn_traj['z']-targets[1])**2)) +
                    norms[1]*(np.sqrt((nn_traj['vx']-targets[2])**2 +
                                      (nn_traj['vz']-targets[3])**2))) 
    
    if len(norms) > 2:
        dist_to_goal += norms[2]*np.abs(nn_traj['theta']-targets[4])
    if len(norms) > 3:
        dist_to_goal += norms[3]*np.abs(nn_traj['dtheta']-targets[5])

    min_arg = dist_to_goal.argmin()
    min_st = nn_traj.iloc[min_arg]

    diff_dist = (np.sqrt((min_st['x']-targets[0])**2 +
                 (min_st['z']-targets[1])**2))
    diff_v = (np.sqrt((min_st['vx']-targets[2])**2 +
                      (min_st['vz']-targets[3])**2))

    diff = [diff_dist, diff_v]

    if len(norms) > 2:
        diff_theta = np.abs(min_st['theta'])
        diff.append(diff_theta)
    
    
    if len(norms) > 3:
        diff_dtheta = np.abs(min_st['dtheta'])
        dist_to_goal += diff_dtheta
        diff.append(diff_dtheta)

    # lims_st = np.sqrt((opt_traj['x']-targets[0])**2 +
    #                   (opt_traj['z']-targets[1])**2) >= diff_dist
    # lims_v = np.sqrt((opt_traj['vx']-targets[2])**2 +
    #                  (opt_traj['vz']-targets[3])**2) >= diff_v
    # lims_theta = np.abs(opt_traj['theta']-targets[4]) >= diff_theta
    # arr_and = np.logical_and
    # opt_end = arr_and(arr_and(lims_st, lims_v), lims_theta).nonzero()[0][-1]

    # discretization??

    closest = (norms[0]*(np.sqrt((opt_traj['x']-targets[0])**2 +
                                 (opt_traj['z']-targets[1])**2) -
                         diff_dist) +
               norms[1]*(np.sqrt((opt_traj['vx']-targets[2])**2 +
                                 (opt_traj['vz']-targets[3])**2) -
                         diff_v) )

    if len(norms) > 2:
        closest +=  norms[2]*(np.abs(opt_traj['theta']-targets[4]) - diff_theta)

    if len(norms) > 3:
        closest += norms[3]*(np.abs(opt_traj['dtheta']-targets[5]) -
                             diff_theta)

#    closest[closest <= 0] = np.Inf
    absclosest = np.abs(closest)
    opt_end = absclosest.argsort()[0]
    optimality_err = value_err(opt_traj.iloc[:opt_end+1],
                               nn_traj.iloc[:min_arg+1])
    a1 = absclosest.min()

    closest[closest <= 0] = np.Inf
    a2 = closest.min()

    opt_end = closest.argsort()[0]
    optimality_err2 = value_err(opt_traj.iloc[:opt_end+1],
                               nn_traj.iloc[:min_arg+1])


    optimality_err = optimality_err * a2/(a1+a2) + optimality_err2* a1/(a1+a2)

    return (diff, optimality_err, (opt_traj,
                               nn_traj))


def mass_optimal(nn_traj, opt_traj):
    """TODO."""
    nn_mass = np.diff(nn_traj['m'].iloc[[-1, 0]])
    opt_mass = np.diff(opt_traj['m'].iloc[[-1, 0]])
    return (nn_mass-opt_mass)/opt_mass
