"""
Implements an indirect method to solve the optimal control
problem of a varying mass spacecraft. No attitude is present, 
hence the name "simple"

Dario Izzo 2016

"""

from PyGMO.problem._base import base
from numpy.linalg import norm
from math import sqrt, sin, cos, atan2
from scipy.integrate import odeint
from numpy import linspace
from copy import deepcopy
import sys


class simple_landing(base):
    def __init__(
            self,
            state0 = [0., 1000., 20., -5., 10000.],
            statet = [0., 0., 0., 0, 9758.695805],
            c1=44000.,
            c2 = 311. * 9.81,
            g = 1.6229,
            homotopy = 0.,
            pinpoint = False
            ):
        """
        USAGE: reachable(self, start, end, Isp, Tmax, mu):

        * state0: initial state [x, y, vx, vy, m] in m,m,m/s,m/s,kg
        * statet: target state [x, y, vx, vy, m] in m,m,m/s,m/s,kg
        * c1: maximum thrusts for the main thruster (N)
        * c2: veff, Isp*g0 (m / s)
        * g: planet gravity [m/s**2]
        * homotopy: homotopy parameter, 0->QC, 1->MOC
        * pinpoint: if True toggles the final constraint on the landing x
        """

        super(simple_landing, self).__init__(6, 0, 1, 6, 0, 1e-5)

        # We store the raw inputs for convenience
        self.state0_input = state0
        self.statet_input = statet

        # We define the non dimensional units (will use these from here on)
        self.R = 1000.
        self.V = 100.
        self.M = 10000.
        self.A = (self.V * self.V) / self.R
        self.T = self.R / self.V
        self.F = self.M * self.A

        # We store the parameters
        self.c1 = c1 / self.F
        self.c2 = c2 / self.V
        self.g = g / self.A

        # We compute the initial and final state in the new units
        self.state0 = self._non_dim(self.state0_input)
        self.statet = self._non_dim(self.statet_input)

        # We set the bounds (these will only be used to initialize the population)
        self.set_bounds([-1.] * 5 + [1e-04], [1.] * 5 + [100. / self.T])

        # Activates a pinpoint landing
        self.pinpoint = pinpoint

        # Stores the homotopy parameter, 0->QC, 1->MOC
        self.homotopy = homotopy

    def _objfun_impl(self, x):
        return(1.,) # constraint satisfaction, no objfun

    def _compute_constraints_impl(self, x):
        # Perform one forward shooting
        xf, info = self._shoot(x)

        # Assembling the equality constraint vector
        ceq = list([0]*6)

        # Final conditions
        if self.pinpoint:
            #Pinpoint landing x is fixed lx is free
            ceq[0] = (xf[-1][0] - self.statet[0] )
        else:
            #Transversality condition: x is free lx is 0
            ceq[0] = xf[-1][5] ** 2

        ceq[1] = (xf[-1][1] - self.statet[1] )
        ceq[2] = (xf[-1][2] - self.statet[2] )
        ceq[3] = (xf[-1][3] - self.statet[3] )
        
        # Transversality condition on mass (free)
        ceq[4] = xf[-1][9] ** 2

        # Free time problem, Hamiltonian must be 0
        ceq[5] = self._hamiltonian(xf[-1]) ** 2

        return ceq

    def _hamiltonian(self, full_state):
        state = full_state[:5]
        costate = full_state[5:]

        # Applying Pontryagin minimum principle
        controls = self._pontryagin_minimum_principle(full_state)

        # Computing the R.H.S. of the state eom
        f_vett = self._eom_state(state, controls)

        # Assembling the Hamiltonian
        H = 0.
        for l, f in zip(costate, f_vett):
            H += l * f

        # Adding the integral cost function (WHY -)
        H += self._cost(state, controls)
        return H

    def _cost(self,state, controls):
        c1 = self.c1
        c2 = self.c2
        u, stheta, ctheta = controls

        retval = self.homotopy * c1 / c2 * u + (1 - self.homotopy) * c1**2 / c2 * u**2
        return retval

    def _eom_state(self, state, controls):
        # Renaming variables
        x,y,vx,vy,m = state
        c1 = self.c1
        c2 = self.c2
        g = self.g
        u, stheta, ctheta = controls

        # Equations for the state
        dx = vx
        dy = vy
        dvx = c1 * u / m * stheta
        dvy = c1 * u / m * ctheta - g
        dm = - c1 * u / c2
        return [dx, dy, dvx, dvy, dm]

    def _eom_costate(self, full_state, controls):
        # Renaming variables
        x,y,vx,vy,m,lx,ly,lvx,lvy,lm = full_state
        c1 = self.c1
        u, stheta, ctheta = controls

        # Equations for the costate
        lvdotitheta = lvx * stheta + lvy * ctheta
        dlx = 0.
        dly = 0.
        dlvx = - lx
        dlvy = - ly
        dlm =  c1 * u / m**2 * lvdotitheta
        
        return [dlx, dly, dlvx, dlvy, dlm]

    def _pontryagin_minimum_principle(self, full_state):
        # Renaming variables
        c1 = self.c1
        c2 = self.c2
        x,y,vx,vy,m,lx,ly,lvx,lvy,lm = full_state

        lv_norm = sqrt(lvx**2 + lvy**2)
        stheta = - lvx / lv_norm
        ctheta = - lvy / lv_norm

        if self.homotopy == 1:
            # Minimum mass
            S = 1. - lm - lv_norm / m * c2
            if S >= 0:
                u=0.
            if S < 0:
                u=1.
        else:
            u = 1. / 2. / c1 / (1 - self.homotopy) * (lm + lv_norm * c2 / m - self.homotopy)
            u = min(u,1.)
            u = max(u,0.)
        return [u, stheta, ctheta]

    def _eom(self, full_state, t):
        # Applying Pontryagin minimum principle
        state = full_state[:5]
        controls = self._pontryagin_minimum_principle(full_state)
        # Equations for the state
        dstate = self._eom_state(state, controls)
        # Equations for the co-states
        dcostate = self._eom_costate(full_state, controls)
        return dstate + dcostate

    def _shoot(self, x):
        # Numerical Integration
        xf, info = odeint(lambda a,b: self._eom(a,b), self.state0 + list(x[:-1]), linspace(0, x[-1],100), rtol=1e-13, atol=1e-13, full_output=1, mxstep=2000)
        return xf, info

    def _simulate(self, x, tspan):
        # Numerical Integration
        xf, info = odeint(lambda a,b: self._eom(a,b), self.state0 + list(x[:-1]), tspan, rtol=1e-13, atol=1e-13, full_output=1, mxstep=2000)
        return xf, info

    def _non_dim(self, state):
        xnd = deepcopy(state)
        xnd[0] /= self.R
        xnd[1] /= self.R
        xnd[2] /= self.V
        xnd[3] /= self.V
        xnd[4] /= self.M
        return xnd

    def _dim_back(self, state):
        xd = deepcopy(state)
        xd[0] *= self.R
        xd[1] *= self.R
        xd[2] *= self.V
        xd[3] *= self.V
        xd[4] *= self.M
        return xd

    def plot(self, x):
        import matplotlib as mpl
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        mpl.rcParams['legend.fontsize'] = 10

        # Producing the data
        tspan = linspace(0, x[-1], 100)
        full_state, info = self._simulate(x, tspan)
        # Putting dimensions back
        res = list()
        controls = list()
        ux = list(); uy=list()
        for line in full_state:
            res.append(self._dim_back(line[:7]))
            controls.append(self._pontryagin_minimum_principle(line))
            ux.append(controls[-1][0]*controls[-1][1])
            uy.append(controls[-1][0]*controls[-1][2])
        tspan = [it * self.T for it in tspan]

        x = list(); y=list()
        vx = list(); vy = list()
        m = list()
        for state in res:
            x.append(state[0])
            y.append(state[1])
            vx.append(state[2])
            vy.append(state[3])
            m.append(state[4])

        fig = plt.figure()
        ax = fig.gca()
        ax.plot(x, y, color='r', label='Trajectory')
        ax.quiver(x, y, ux, uy, label='Thrust', pivot='tail', width=0.001)
        ax.set_ylim(0,self.state0_input[1]+500)

        f, axarr = plt.subplots(3, 2)

        axarr[0,0].plot(x, y)
        axarr[0,0].set_xlabel('x'); axarr[0,0].set_ylabel('y'); 

        axarr[1,0].plot(vx, vy)
        axarr[1,0].set_xlabel('vx'); axarr[1,0].set_ylabel('vy');

        axarr[2,0].plot(tspan, m)

        axarr[0,1].plot(tspan, [controls[ix][0] for ix in range(len(controls))],'r')
        axarr[0,1].set_ylabel('u')
        axarr[0,1].set_xlabel('t')
        axarr[1,1].plot(tspan, [atan2(controls[ix][1], controls[ix][2]) for ix in range(len(controls))],'k')
        axarr[1,1].set_ylabel('theta')
        axarr[1,1].set_xlabel('t')
        axarr[2,1].plot(tspan, [controls[ix][2] for ix in range(len(controls))],'k')


        plt.ion()
        plt.show()
        return axarr

    def human_readable_extra(self):
        s = "\n\tDimensional inputs:\n"
        s = s + "\tStarting state: " + str(self.state0_input) + "\n"
        s = s + "\tTarget state: " + str(self.statet_input) + "\n"
        s = s + "\tThrusters maximum magnitude [N]: " + str(self.c1 * self.F) + "\n"
        s = s + "\tIsp * g0: " + str(self.c2 * self.V) + ", gravity: " + str(self.g * self.A) + "\n"

        s = s + "\n\tNon - Dimensional inputs:\n"
        s = s + "\tStarting state: " + str(self.state0) + "\n"
        s = s + "\tTarget state: " + str(self.statet) + "\n"
        s = s + "\tThrusters maximum magnitude [N]: " + str(self.c1) + "\n"
        s = s + "\tIsp * g0: " + str(self.c2) + ", gravity: " + str(self.g) + "\n\n"
        
        s = s + "\tHomotopy parameter: " + str(self.homotopy)
        s = s + "\tPinpoint?: " + str(self.pinpoint)

        return s

if __name__ == "__main__":
    from PyGMO import *
    from random import random

    # Use SNOPT if possible
    algo = algorithm.snopt(200, opt_tol=1e-5, feas_tol=1e-6)

    # Alternatively the scipy SQP solver can be used
    #algo = algorithm.scipy_slsqp(max_iter = 1000,acc = 1E-8,epsilon = 1.49e-08, screen_output = True)
    #algo.screen_output = True

    # Define the starting area (x0 will be irrelevanto if pinpoint is not True)
    x0b =  [-100, 100]
    y0b =  [500, 2000]
    vx0b = [-100, 100]
    vy0b = [-30, 10]
    m0b =  [8000, 12000]

    x0 = random() * (x0b[1] - x0b[0]) + x0b[0]
    y0 = random() * (y0b[1] - y0b[0]) + y0b[0]
    vx0 = random() * (vx0b[1] - vx0b[0]) + vx0b[0]
    vy0 = random() * (vy0b[1] - vy0b[0]) + vy0b[0]
    m0 = random() * (m0b[1] - m0b[0]) + m0b[0]
    state0 = [x0, y0, vx0, vy0, m0]

    # We start solving the Quadratic Control
    print("Trying I.C. {}".format(state0)),
    prob = simple_landing(state0 = state0, homotopy=0., pinpoint=True)
    count = 1
    for i in range(1, 20):
        print("Attempt # {}".format(i), end="")
        pop = population(prob,1)
        pop = algo.evolve(pop)
        pop = algo.evolve(pop)
        if (prob.feasibility_x(pop[0].cur_x)):
            print(" - Success, violation norm is: {0:.4g}".format(norm(pop[0].cur_c)))
            break
        else:
            print(" - Failed, violation norm is: {0:.4g}".format(norm(pop[0].cur_c)))

    print("PaGMO reports: ", end="")
    print(prob.feasibility_x(pop[0].cur_x))

    if not prob.feasibility_x(pop[0].cur_x):
        print("No QC solution! Ending here :(")
        sys.exit(0)
    else: 
        print("Found QC solution!! Starting Homotopy")
    print("from \t to\t step\t result")
    
    # We proceed to solve by homotopy the mass optimal control
    # Minimum and maximum step for the continuation
    h_min = 1e-8
    h_max = 0.3
    # Starting step
    h = 0.1

    trial_alpha = h
    alpha = 0
    x = pop[0].cur_x

    #algo.screen_output = True
    while True:
        if trial_alpha > 1:
            trial_alpha = 1.
        print("{0:.5g}, \t {1:.5g} \t".format(alpha, trial_alpha), end="")
        print("({0:.5g})\t".format(h), end="")
        prob = simple_landing(state0 = state0, pinpoint=True, homotopy=trial_alpha)

        pop = population(prob)
        pop.push_back(x)
        pop = algo.evolve(pop)

        if not (prob.feasibility_x(pop[0].cur_x)):
            pop = algo.evolve(pop)
            pop = algo.evolve(pop)
            pop = algo.evolve(pop)

        if (prob.feasibility_x(pop[0].cur_x)):
            x = pop.champion.x
            if trial_alpha == 1:
                print(" Success")
                break
            print(" Success")
            h = h * 2.
            h = min(h, h_max)
            alpha = trial_alpha
            trial_alpha = trial_alpha + h
        else:
            print(" - Failed, ", end="")
            print("norm c: {0:.4g}".format(norm(pop[0].cur_c)))
            h = h * 0.5
            if h < h_min:
                print("\nContinuation step too small aborting :(")
                sys.exit(0)
            trial_alpha = alpha + h