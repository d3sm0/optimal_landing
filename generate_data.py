"""
Generate data using any of the landing models:
 - single trajectories
 - homotopy from quadratic control to mass optimal
 - random walks from an initial trajectory
 - parallel generation

@cesans 2016

"""

from random import random
import sys
import os
from multiprocessing import Process
import pickle

import numpy as np
from numpy.linalg import norm


from PyGMO import algorithm, population


def solve(problem, state0, homotopy=0, algo=None,  x=None, display=True):

    if not algo:
        # Use SNOPT if possible
        algo = algorithm.snopt(400, opt_tol=1e-3, feas_tol=1e-6)

        # Alternatively the scipy SQP solver can be used
        # algo = algorithm.scipy_slsqp(max_iter = 1000,acc = 1E-8,epsilon = 1.49e-08, screen_output = True)
        # algo.screen_output = True

    prob = problem(state0=state0, homotopy=homotopy, pinpoint=True)
    if x is None:
        pop = population(prob, 1)
    else:
        pop = population(prob)
        pop.push_back(x)

    pop = algo.evolve(pop)
    x = pop.champion.x
    feasible = prob.feasibility_x(x)

    if not feasible and (norm(x) < 1e-2):
        pop = algo.evolve(pop)
        pop = algo.evolve(pop)
        x = pop.champion.x
        feasible = prob.feasibility_x(x)

    if display:
        print( (u'\u2713' if feasible else u'\u2717') + ' (homotopy: {0})'.format(homotopy))

    return {'x': pop.champion.x, 'prob': prob, 'feasible': feasible}


def homotopy_path(problem, state0, algo=None, start=(0, None), h_min=1e-8, h_max=0.5, h=0.5, display=True):


    sol = solve(problem, state0, start[0], algo, x=start[1], display=display)
    if not sol['feasible']:
        if display:
            print('\t > The homotopy path could not be started')
        return sol
    else:
        if display:
            print('\t > Homotopy path started with h: {0}'.format(h))

    alpha = 0
    trial_alpha = h
    x = sol['x']
    while h > h_min:

        sol = solve(problem, state0, trial_alpha, algo, x=x, display=display)
        if sol['feasible']:
            x = sol['x']
            if trial_alpha == 1:

                # Correct result found
                return sol

            h = min(h * 2.0, h_max)
            (alpha, trial_alpha) = (trial_alpha, trial_alpha+h)
            trial_alpha = min(1, trial_alpha)

        else:
            h *= 0.5
            if display:
                print('\t > Decreasing h: {}'.format(h))
            trial_alpha = alpha + h

    if display:
        print('<< Fail = Homotopy path stopped')
    return sol


def random_state(ranges):

    state = []
    for r in ranges:
        var = random() * (r[1] - r[0]) + r[0]
        state.append(var)
    return state


def random_walk(problem, state0, bounds, walk_length=300, algo=None, walk_stop_when_fail=False, initial_x = 'homotopy',
                state_step=0.02, h_min=1e-8, h_max=0.5, h=0.5, display=True):
    '''

    :param problem:
    :param bounds:
    :param walk_length:
    :param algo:
    :param walk_stop_when_fail:
    :param initial_x: ['homotopy', None]
    :return:
    '''

    walk_trajs = []
    step_ranges = [(b[1]-b[0])*state_step for b in bounds]

    if initial_x is 'homotopy':
        sol = homotopy_path(problem, state0, algo=None, h_min=h_min, h_max=h_max, h=h, display=display)

    if initial_x is None:
        sol = solve(problem, state0, 1, algo=None, display=display)


    if not sol['feasible']:
        if display:
            print('\t> The random walk could not be started')
        return walk_trajs

    x = sol['x']
    state, control = sol['prob'].produce_data(x, 100)
    walk_trajs.append((state, control, x))

    while len(walk_trajs) < walk_length:

        state_tmp = [r*random()-r/2+x for r, x in zip(step_ranges, state0)]

        if not np.all([b[0] < s < b[1] for b, s in zip(bounds, state_tmp)]):
            break

        sol = solve(problem, state_tmp, 1, algo, x=x, display=display)

        if sol['feasible']:
            x = sol['x']
            state, control = sol['prob'].produce_data(x, 100)
            walk_trajs.append((state, control, x))
            state0 = state_tmp
        else:
            #   TODO  keep the same direction
            if walk_stop_when_fail:
                break

    return walk_trajs


def generate_random_walks(problem, trajs_n, bounds, th_id=0, dir='data', walk_length=300, algo=None,
                          state_step=0.02, h_min=1e-8, h_max=0.5, h=0.5, walk_stop_when_fail=False, 
                          display=True):

    curr_trajs = 0
    walk_id = 0
    if not os.path.exists(dir):
        os.makedirs(dir)

    while curr_trajs < trajs_n:

        if os.path.isfile(dir + '/random_walk_' + str(th_id) + '_' + str(walk_id) +'.pic'):
            ws = pickle.load(open(dir + '/random_walk_' + str(th_id) + '_' + str(walk_id) +'.pic','rb'))
            walk_id +=1
            curr_trajs += len(ws)
            continue

        state0 = random_state(bounds)

        walk_length = min(walk_length, trajs_n-curr_trajs)
        walk_trajs = random_walk(problem, state0, bounds, walk_length= walk_length, algo=algo,
                                 state_step= state_step, h_min = h_min, h_max = h_max, h=h,
                                 walk_stop_when_fail=walk_stop_when_fail, display=display)
        if(len(walk_trajs) > 0):
             curr_trajs += len(walk_trajs)
             pickle.dump(walk_trajs, open(dir + '/random_walk_' + str(th_id) + '_' + str(walk_id) +'.pic','wb'))
             walk_id += 1


def run_multithread(problem, n_trajs, n_threads, bounds, dir='data', walk_length=300, algo=None,
                          state_step=0.02, h_min=1e-8, h_max=0.5, h=0.5, walk_stop_when_fail=False,
                          display=True):

    samples_per_thread = n_trajs/n_threads

    ps = []
    for i in range(n_threads):
        p = Process(target=generate_random_walks, args=(problem, samples_per_thread, bounds, i,
                dir,walk_length,algo, state_step, h_min, h_max, h, walk_stop_when_fail, display))
        p.start()
        ps.append(p)

    for p in ps:
        p.join()


if __name__ == "__main__":

    x0b = (-200, 200)
    y0b = (500, 2000)
    vx0b = (-30, 30)
    vy0b = (-30, 10)
    m0b = (8000, 12000)
    th0b = (- np.pi/20, np.pi/20)

    if len(sys.argv) != 4 or sys.argv[3] == 'simple':
        from simple_landing import simple_landing as landing_problem
        initial_bounds = [x0b, y0b, vx0b, vy0b, m0b]
        filedir='simple'
    elif sys.argv[3] == 'rw':
        from rw_landing import rw_landing as landing_problem
        vx0b = (-10, 10)
        initial_bounds = [x0b, y0b, x0b, vy0b, th0b, m0b]
        filedir='rw'

    trajs = int(sys.argv[1])

    n_th = 1
    if len(sys.argv) > 2:
        n_th = int(sys.argv[2])

    # state = random_state(bounds)
    # sol = homotopy_path(landing_problem, state)

    run_multithread(landing_problem, trajs, n_th, initial_bounds, 'data/' + filedir, display=True)

