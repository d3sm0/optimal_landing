"""Generate data using any of the landing models.

 - single trajectories
 - homotopy from quadratic control to mass optimal
 - random walks from an initial trajectory
 - parallel generation

@cesans 2016

"""
from __future__ import print_function
from random import random
import sys
import os
from multiprocessing import Process
import pickle
from numpy import pi
import numpy as np
from numpy.linalg import norm

from PyGMO import algorithm, population


def solve(problem, state0, homotopy=0, algo=None,  x=None, display=True):
#    algo = algorithm.scipy_slsqp(max_iter=30, acc=1E-4, epsilon=1e-6,
 #                                    screen_output=False)
    if not algo:
        # Use SNOPT if possible
        algo = algorithm.snopt(400, opt_tol=1e-4, feas_tol=1e-4)

        # Alternatively the scipy SQP solver can be used
        # algo = algorithm.scipy_slsqp(max_iter = 1000,acc = 1E-8,
        #                             epsilon = 1.49e-08, screen_output = True)
        # algo.screen_output = True


    prob = problem(state0=state0, homotopy=homotopy, pinpoint=True,)



    if display:
        print('state: ', state0)
        print('homot: ', homotopy)
        print('algo : ', algo)
        print('x : ', x)

    if x is None:
        pop = population(prob, 1)
    else:
        pop = population(prob)
        pop.push_back(x)
   
    print(pop[0].cur_x)
    try:
        pop = algo.evolve(pop)
        x = pop[0].cur_x
        feasible = prob.feasibility_x(x)


        if not feasible and (norm(x) < 1e-2):
            pop = algo.evolve(pop)
            pop = algo.evolve(pop)
    #        x = pop.champion.x
            x = pop[0].cur_x
        feasible = prob.feasibility_x(x)
         
    except ValueError:
        feasible=False

    if display:
        print('evx    : ', x)
        print(feasible)
#   if display:
#       print( (u'\u2713' if feasible else u'\u2717') +
#                ' (homotopy: {0})'.format(homotopy))

    return {'x': pop.champion.x, 'prob': prob, 'feasible': feasible}


def homotopy_path(problem, state0, algo=None, start=(0, None), h_min=1e-4,
                  h_max=0.5, h=0.1, display=True):
    
    sol = solve(problem, state0, start[0], algo, x=start[1], display=display)
#    sol = solve(problem, state0, start[0], algo, x=ini_x, display=display)
    if not sol['feasible']:
        if display:
            print('\t > The homotopy path could not be started')
        return sol,0
    else:
        if display:
            print('\t > Homotopy path started with h: {0}'.format(h))

    alpha = 0
    trial_alpha = h
    x = sol['x']
    h_min = 1e-12
    its = 0
    h_max=0.2
    hpaths = []
    state, control = sol['prob'].produce_data(x, 1000)
    hpaths.append((state,control,x,0))
    increase_fater = 5
    inccount = 0
    while h > h_min and its < 1000:
        sol = solve(problem, state0, trial_alpha, algo, x=x, display=display)
        its +=1
        if sol['feasible']:
            x = sol['x']
            if trial_alpha == 1:
                return sol,1
            state, control = sol['prob'].produce_data(x, 1000)
            hpaths.append((state,control,x,0,trial_alpha))
            pickle.dump(hpaths,open('hpaths.pic','wb'))
            inccount +=1
            if inccount == 5:
                h *= 2
                inccount = 0
            h = min(min(h, h_max), 1-trial_alpha)
            print('Increasing h to: {}'.format(h))
            (alpha, trial_alpha) = (trial_alpha, trial_alpha+h)
            trial_alpha = min(1, trial_alpha)
        else:
            h *= 0.5
            if display:
                print('\t > Decreasing h: {}'.format(h))
            trial_alpha = alpha + h
            trial_alpha = min(1, trial_alpha)
            
        
    if display:
        print('<< Fail = Homotopy path stopped')

    if alpha == 1:
        sol = solve(problem, state0, alpha, algo, x=x, display=display)
    return sol, alpha


def random_state(ranges):
    state = []
    for r in ranges:
        var = random() * (r[1] - r[0]) + r[0]
        state.append(var)
    return state


def random_walk(problem, state0, bounds, walk_length=300, algo=None,
                walk_stop_when_fail=False, initial_x='homotopy',
                state_step=0.02, h_min=1e-4, h_max=0.5, h=0.1, display=True,
                ini_trials=1, walk_bounds=None):

    if not walk_bounds:
        walk_bounds = bounds
    walk_trajs = []
    step_ranges = [(b[1]-b[0])*state_step for b in walk_bounds]

    x = None
    if isinstance(initial_x, list) and not isinstance(initial_x, str):
        iniran_x = None
        if len(initial_x) == 3:
            iniran_x = initial_x[2]
        for i in range(ini_trials):
            sol = solve(problem, state0, algo=initial_x[0],
                        x=iniran_x, display=display)
            if sol['feasible']:
                break

        if not sol['feasible']:
            if display:
                print('\t> The random walk could not be started')
            return walk_trajs

        initial_x = initial_x[1]
        x = sol['x']

    if initial_x is 'homotopy':
        sol,alpha = homotopy_path(problem, state0, start=(0, x), algo=algo,
                            h_min=h_min, h_max=h_max, h=h, display=display)
    else:
        sol = solve(problem, state0, 1, x=x, algo=algo, display=display)
    if not sol['feasible']:
        if display:
            print('\t> The random walk could not be started')
        return walk_trajs

    x = sol['x']
    state, control = sol['prob'].produce_data(x, 1000)
    walk_trajs.append((state, control, x))

    while len(walk_trajs) < walk_length:
        state_tmp = [r*random()-r/2+x for r, x in zip(step_ranges, state0)]

        if not np.all([b[0] <= s <= b[1] for b, s
           in zip(walk_bounds, state_tmp)]):
            print(state_tmp)
            print(walk_bounds)
            print('out of bounds')
            break

        sol = solve(problem, state_tmp, alpha, algo, x=x, display=display)

        if sol['feasible']:
            x = sol['x']
            state, control = sol['prob'].produce_data(x, 1000)
            walk_trajs.append((state, control, x))
            state0 = state_tmp
        else:
        
            #   TODO  keep the same direction
            if walk_stop_when_fail:
                break
        print('walk_step', len(walk_trajs))
        pickle.dump(walk_trajs, open('current_walk.pic', 'wb'))

    return walk_trajs


def random_walk_h0(problem, state0, bounds, walk_length=300, algo=None,
                   walk_stop_when_fail=False, state_step=0.02, display=True,
                   ini_trials=1, walk_bounds=None, initial_random_walk=None):
    if not walk_bounds:
        walk_bounds = bounds

    walk_trajs = []
    step_ranges = [(b[1]-b[0])*state_step for b in walk_bounds]
    print('random walk ---> ini')
    x = None
    if initial_random_walk and initial_random_walk!='homotopy':
        x = initial_random_walk
    print('random walk ---> ini', x, walk_length)

    for _ in range(ini_trials):
        sol = solve(problem, state0, 0, x=x, algo=algo, display=display)
        if sol['feasible']:
            break
        else:
            x = None

    if not sol['feasible']:
        if display:
            print('\t> The random walk could not be started')
        return walk_trajs

    x = sol['x']
    state, control = sol['prob'].produce_data(x, 1000)
    walk_trajs.append((state, control, x))
    while len(walk_trajs) < walk_length:
        print(len(walk_trajs))
        state_tmp = [r*random()-r/2+x for r, x in zip(step_ranges, state0)]
        if not np.all([b[0] < s < b[1] for b, s
           in zip(walk_bounds, state_tmp)]):
            print(state_tmp, walk_bounds, 'outtt')
            
            break
        sol = solve(problem, state_tmp, 0, algo, x=x, display=display)

        if sol['feasible']:
            x = sol['x']
            state, control = sol['prob'].produce_data(x, 1000)
            walk_trajs.append((state, control, x))
            state0 = state_tmp
        else:
            #   TODO  keep the same direction
            if walk_stop_when_fail:
               break

    return walk_trajs


def generate_random_walks(problem, trajs_n, bounds, th_id=0, dir='data',
                          walk_length=300, algo=None, state_step=0.02,
                          h_min=1e-4, h_max=0.5, h=0.1,
                          stop_when_fail=False, display=True,
                          initial_random_walk='homotopy', walk_bounds=None,
                          qc=False):

    curr_trajs = 0
    walk_id = 0
    if not os.path.exists(dir):
        os.makedirs(dir)

    while curr_trajs < trajs_n:
        print(dir)

        if os.path.isfile(dir + '/random_walk_' + str(th_id) + '_' +
                          str(walk_id) + '.pic'):
            ws = pickle.load(open(dir + '/random_walk_' + str(th_id) + '_' +
                             str(walk_id) + '.pic', 'rb'))
            walk_id += 1
            curr_trajs += len(ws)
            continue

        state0 = random_state(bounds)

        walk_length = min(walk_length, trajs_n-curr_trajs)
        print(trajs_n, curr_trajs, walk_length)
        if qc:
            walk_trajs = random_walk_h0(problem, state0, bounds,
                                        walk_length=walk_length, algo=algo,
                                        state_step=state_step, display=display,
                                        walk_stop_when_fail=stop_when_fail,
                                        walk_bounds=walk_bounds,
                                        initial_random_walk=initial_random_walk)

        else:
            walk_trajs = random_walk(problem, state0, bounds,
                                     walk_length=walk_length, algo=algo,
                                     state_step=state_step, h_min=h_min,
                                     h_max=h_max, h=h,
                                     walk_stop_when_fail=stop_when_fail,
                                     display=display, walk_bounds=walk_bounds,
                                     initial_x=initial_random_walk)

        if(len(walk_trajs) > 0):
            curr_trajs += len(walk_trajs)
            pickle.dump(walk_trajs, open(dir + '/random_walk_' + str(th_id) +
                                         '_' + str(walk_id) + '.pic', 'wb'))
            walk_id += 1


def run_multithread(problem, n_trajs, n_threads, bounds, dir='data',
                    walk_length=300, state_step=0.02, h_min=1e-4, h_max=0.5,
                    h=0.1, walk_stop_when_fail=False, display=True,
                    initial_random_walk='homotopy', algo=None,
                    walk_bounds=None, qc=False):

    samples_per_thread = n_trajs/n_threads

    ps = []
    for i in range(n_threads):
        p = Process(target=generate_random_walks,
                    args=(problem, samples_per_thread, bounds, i, dir,
                          walk_length, algo, state_step, h_min, h_max, h,
                          walk_stop_when_fail, display,
                          initial_random_walk, walk_bounds, qc))
        p.start()
        ps.append(p)

    try:
       for p in ps:
           p.join()
    except:
       p.terminate()   


if __name__ == "__main__":

    x0b = (-200, 200)
    y0b = (500, 2000)
    vx0b = (-10, 10)
    vy0b = (-30, 10)
    m0b = (8000, 12000)
    th0b = (- np.pi/20, np.pi/20)

    if len(sys.argv) != 4 or sys.argv[3] == 'simple':
        from simple_landing import simple_landing as landing_problem
        initial_bounds = [x0b, y0b, vx0b, vy0b, m0b]
        filedir = 'simple_red'

    elif sys.argv[3] == 'simple_qc':
        from simple_landing import simple_landing as landing_problem
        initial_bounds = [x0b, y0b, vx0b, vy0b, m0b]
        filedir = 'simple_qc_red'

    elif sys.argv[3] == 'rw':
        from rw_landing import rw_landing as landing_problem
        vx0b = (-10, 10)
        initial_bounds = [x0b, y0b, vx0b, vy0b, th0b, m0b]
        filedir = 'rw2'

    elif sys.argv[3] == 'rw_qc':
        from rw_landing import rw_landing as landing_problem
        vx0b = (-10, 10)
        initial_bounds = [x0b, y0b, vx0b, vy0b, th0b, m0b]
        filedir = 'rw2_qc'

    elif sys.argv[3] == 'tv_qc':
        from tv_landing import tv_landing as landing_problem
        vx0b = (-10, 10)

        x0b = [-0.01, 0.01]
#        x0b = [-0.1, 0.1]
#        x0b = [-0.2, 0.2]
        y0b = [500, 2000]
        vx0b = [-0.01, 0.01]
        vx0b = [-0.001, 0.001]
        vx0b = [-0.001, 0.001]
        vy0b = [-40, 5]
        m0b = [8000,12000]
        theta0 = [0,0]
        omega0 = [0,0]
        filedir = 'tv_qc'
        initial_bounds = [x0b, y0b, vx0b, vy0b, theta0, omega0, m0b]
        x0b = [-100, 100]
        x0b = [-20, 20]
        y0b = [500, 2000]
        vx0b = [-2, 2]
        vx0b = [-1, 1]
        vy0b = [-40, 5]
        theta0 = [-0.1/360*2*pi, 0.1/360*2*pi]
        omega0 = [-0.01/360*2*pi, 0.01/360*2*pi]
        walk_bounds = [x0b, y0b, vx0b, vy0b, theta0, omega0, m0b]
    elif sys.argv[3] == 'tv':
        from tv_landing import tv_landing as landing_problem
        vx0b = (-10, 10)

        x0b = [-0.01, 0.01]
#        x0b = [-0.1, 0.1]
#        x0b = [-0.2, 0.2]
        y0b = [500, 2000]
        vx0b = [-0.01, 0.01]
        vx0b = [-0.001, 0.001]
        vx0b = [-0.001, 0.001]
        vy0b = [-40, 5]
        m0b = [8000,12000]
        theta0 = [0,0]
        omega0 = [0,0]
        filedir = 'tv'
        initial_bounds = [x0b, y0b, vx0b, vy0b, theta0, omega0, m0b]
        x0b = [-100, 100]
        x0b = [-20, 20]
        y0b = [500, 2000]
        vx0b = [-2, 2]
        vx0b = [-1, 1]
        vy0b = [-40, 5]
        theta0 = [-0.1/360*2*pi, 0.1/360*2*pi]
        omega0 = [-0.01/360*2*pi, 0.01/360*2*pi]
        walk_bounds = [x0b, y0b, vx0b, vy0b, theta0, omega0, m0b]

    elif sys.argv[3] == 'falcon_qc':
        from falcon_landing_boun import tv_landing as landing_problem
        x0b = [-20, 20]
        y0b = [500, 2000]
        vx0b = [-15, 15]
        vy0b = [-300, -100]
        m0b = [70000, 90000]

        theta0 = [-5/360*2*np.pi, 5/360*2*np.pi]
        omega0 = [-10/360*2*np.pi, 10/360*2*np.pi]

        initial_bounds = [[-5, 5], [1400, 1600], [-2, 2], [-150, -150],
                          [-5.0/360*2*np.pi, 5.0/360*2*np.pi],
                          [-2.0/360*2*np.pi, 2.0/360*2*np.pi], [45000, 45000]]
        walk_bounds = [x0b, y0b, vx0b, vy0b, theta0, omega0, m0b]
        filedir = 'falcon_qc'
        x = (2.7725139411905354e-10,
             0.005724760124809018,
             2.950847498026487e-10,
             -0.12013195562429928,
             2.996361244192239e-10,
             1.33450685205765e-10,
             0.3936895621144142,
             1.8687501772723236)

    elif sys.argv[3] == 'falcon':
        from falcon_landing_boun import tv_landing as landing_problem
        x0b = [-0.1,0.1]
        y0b = [500, 2000]
        vx0b = [-0.01, 0.01]
        vy0b = [-300, -100]
        m0b = [70000, 90000]

        theta0 = [-0.001/360*2*np.pi, 0.001/360*2*np.pi]
        omega0 = [-0.001/360*2*np.pi, 0.001/360*2*np.pi]

        initial_bounds = [[-5, 5], [1500, 1501], [-2, 2], [-199, -201],
                          [-5.0/360*2*np.pi, 5.0/360*2*np.pi],
                          [-2.0/360*2*np.pi, 2.0/360*2*np.pi], [60000, 80001]]
        initial_bounds = [[-0.01,0.01 ], [1495, 1505], [-0.01,0.01]  ,
                          [-198, -202],
                          [0, 0],
                          [-0    , 0], [60000, 62000]]
        walk_bounds = [x0b, y0b, vx0b, vy0b, theta0, omega0, m0b]
        filedir = 'falcon'
        x =  (0.0004063623657985068, -0.015166893186127163, 0.00047398741968363283, -0.0815609687198395, 0.0005701909839526759, 
            0.00020132959190827737, 0.35180899558417034, 1.6036558613069618)
        print(x)
        print('-'*50)
    trajs = int(sys.argv[1])

    n_th = 1
    if len(sys.argv) > 2:
        n_th = int(sys.argv[2])

    screen_output = False
    snopt_algo = algorithm.snopt(400, opt_tol=1e-3, feas_tol=1e-6,
                                 screen_output=screen_output)
    sci_algo = algorithm.scipy_slsqp(max_iter=30, acc=1E-8, epsilon=1.49e-08,
                                     screen_output=screen_output)

    if len(sys.argv) != 4 or sys.argv[3] == 'simple':
        run_multithread(landing_problem, trajs, n_th, initial_bounds, 'data/' +
                        filedir, display=True, algo=sci_algo)
    elif sys.argv[3] == 'simple_qc':
        run_multithread(landing_problem, trajs, n_th, initial_bounds, 'data/' +
                        filedir, display=True, qc=True, algo=sci_algo, state_step=0.02)
    elif sys.argv[3] == 'rw_qc':
        
        run_multithread(landing_problem, trajs, n_th, initial_bounds, 'data/' +
                        filedir, display=True, qc=True)
    elif sys.argv[3] == 'falcon_qc':
        print(initial_bounds)
        run_multithread(landing_problem, trajs, n_th, initial_bounds, 'data/' +
                        filedir, display=True, qc=True, initial_random_walk=x,
                        walk_bounds=walk_bounds, state_step=0.02)
    elif sys.argv[3] == 'falcon':
        print(initial_bounds)
        x =  (-0.0023167067667803445, -0.01614981303766957, -0.002975512465203157, 
                        -0.0868113621299725, -0.00435655164157381, -0.0015382588769658873, 0.362969715087788, 
                        1.6037360106105434)
        run_multithread(landing_problem, trajs, n_th, initial_bounds, 'data/' +
                        filedir, display=True, qc=False, 
                        initial_random_walk=[sci_algo,
                                             'homotopy',
                                              x],
                        walk_bounds=walk_bounds, state_step=0.02, algo=sci_algo)
    elif sys.argv[3] == 'rw':
        run_multithread(landing_problem, trajs, n_th, initial_bounds, 'data/' +
                        filedir, display=True,
                        initial_random_walk=[snopt_algo, 'homotopy'],
                        algo=sci_algo, h_max=0.1, h=0.1)
    elif sys.argv[3] == 'tv':
        print(trajs)
        x=(7.070046180388959e-06,
         0.0037478237502960154,
          2.3894990220812782e-05,
           -0.01548651498666569,
            6.578495539593591e-06,
             9.247044162669352e-06,
              0.029009363536818197,
               6.193734311785152)
        run_multithread(landing_problem, trajs, n_th, initial_bounds, 'data/' +
                        filedir, display=True,
                        initial_random_walk=[sci_algo,
                                             'homotopy',
#                                              x],
                                             [0, 0, 0, -0.015, 0, 0, 0, 5]],
                        algo=sci_algo, h_max=0.1, h=0.1,
                        walk_bounds=walk_bounds, state_step=0.02)
    elif sys.argv[3] == 'tv_qc':
        print(trajs)
        x=(7.070046180388959e-06,
         0.0037478237502960154,
          2.3894990220812782e-05,
           -0.01548651498666569,
            6.578495539593591e-06,
             9.247044162669352e-06,
              0.029009363536818197,
               6.193734311785152)
        x=     [0, 0, 0, -0.015, 0, 0, 0, 5]
        run_multithread(landing_problem, trajs, n_th, initial_bounds, 'data/' +
                        filedir, display=True, qc=True, initial_random_walk=x,
                        walk_bounds=walk_bounds, state_step=0.02, algo=sci_algo)
