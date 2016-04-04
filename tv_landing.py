"""
Implements an indirect method to solve the optimal control
problem of a varying mass spacecraft controlled by one 
thruster capable of vectoring. 

Dario Izzo 2016

"""

from PyGMO.problem._base import base
from numpy.linalg import norm
from math import sqrt, sin, cos, atan2, pi
from scipy.integrate import odeint
from numpy import linspace
from copy import deepcopy
import sys


class tv_landing(base):
    def __init__(
            self,
            state0 = [0., 1000., 20., -5., 0., 0., 10000.],
            statet = [0., 0., 0., 0., 0., 0., 9758.695805],
            c1 = 44000.,
            c2 = 311. * 9.81,
            c3 = 300.,
            g = 1.6229,
            homotopy = 0.,
            pinpoint = False
            ):
        """
        USAGE: tv_landing(self, start, end, Isp, Tmax, mu):

        * state0: initial state [x, y, vx, vy, theta, omega, m] in m, m , m/s, m/s, rad, rad/s, kg
        * statet: target state [x, y, vx, vy, theta, omega, m] in m, m, m/s, m/s, rad, rad/s, kg
        * c1: maximum thrusts for the main thruster [N]
        * c2: veff, Isp*g0 (m / s)
        * c3: characteristic length (I / m / d) [m]
        * g: planet gravity [m/s**2]
        * homotopy: homotopy parameter, 0->QC, 1->MOC
        * pinpoint: if True toggles the final constraint on the landing x
        """

        super(tv_landing, self).__init__(8, 0, 1, 8, 0, 1e-4)

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
        self.c3 = c3 / self.R
        self.g = g / self.A

        # We compute the initial and final state in the new units
        self.state0 = self._non_dim(self.state0_input)
        self.statet = self._non_dim(self.statet_input)

        # We set the bounds (these will only be used to initialize the population)
        self.set_bounds([-1] * 7 + [1. / self.T], [1] * 7 + [200. / self.T])

        # Activates a pinpoint landing
        self.pinpoint = pinpoint

        # Selects the homotopy parameter, 0->QC, 1->MOC
        self.homotopy = homotopy

    def _objfun_impl(self, x):
        return(1.,) # constraint satisfaction, no objfun

    def _compute_constraints_impl(self, x):
        # Perform one forward shooting
        xf, info = self._shoot(x)

        # Assembling the equality constraint vector
        ceq = list([0]*8)

        # Final conditions
        if self.pinpoint:
            #Pinpoint landing x is fixed lx is free
            ceq[0] = (xf[-1][0] - self.statet[0] ) * 1
        else:
            #Transversality condition: x is free lx is 0
            ceq[0] = xf[-1][7] * 1

        ceq[1] = (xf[-1][1] - self.statet[1] ) * 1
        ceq[2] = (xf[-1][2] - self.statet[2] ) * 1
        ceq[3] = (xf[-1][3] - self.statet[3] ) * 1
        ceq[4] = (xf[-1][4] - self.statet[4] ) * 1

        
        # Transversality condition on omega and mass (free)
        ceq[5] = xf[-1][12] * 1
        ceq[6] = xf[-1][13] * 1

        # Free time problem, Hamiltonian must be 0
        ceq[7] = self._hamiltonian(xf[-1]) * 1

        return ceq

    def _hamiltonian(self, full_state):
        state = full_state[:7]
        costate = full_state[7:]

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
        c3 = self.c3
        u, ut = controls
        retval = self.homotopy * c1 / c2 * u + (1 - self.homotopy) * c1**2 / c2 * u**2
        return retval

    def _eom_state(self, state, controls):
        # Renaming variables
        x,y,vx,vy,theta,omega,m = state
        g = self.g
        c1 = self.c1
        c2 = self.c2
        c3 = self.c3
        u, ut = controls

        tdotit = ut[0] * cos(theta) - ut[1] * sin(theta)
        # Equations for the state
        dx = vx
        dy = vy
        dvx = c1 * u / m * ut[0]
        dvy = c1 * u / m * ut[1] - g
        dtheta = omega
        domega = - c1 / c3 * u / m * tdotit
        dm = - c1 / c2 * u
        return [dx, dy, dvx, dvy, dtheta, domega, dm]

    def _eom_costate(self, full_state, controls):
        # Renaming variables
        x,y,vx,vy,theta,omega,m,lx,ly,lvx,lvy,ltheta,lomega,lm = full_state
        c1 = self.c1
        c2 = self.c2
        c3 = self.c3
        u, ut = controls

        # Equations for the costate
        tdotit = ut[0] * cos(theta) - ut[1] * sin(theta)
        tdotitheta = ut[0] * sin(theta) + ut[1] * cos(theta)
        lvdott = lvx * ut[0] + lvy * ut[1]

        dlx = 0.
        dly = 0.
        dlvx = - lx
        dlvy = - ly
        dltheta = - lomega / c3 * c1 * u / m * tdotitheta
        dlomega = - ltheta
        dlm =  c1 / m**2 * u * (lvdott - lomega / c3 * tdotit)
        
        return [dlx, dly, dlvx, dlvy, dltheta, dlomega, dlm]

    def _pontryagin_minimum_principle(self, full_state):
        # Renaming variables
        x,y,vx,vy,theta,omega,m,lx,ly,lvx,lvy,ltheta,lomega,lm = full_state
        c1 = self.c1
        c2 = self.c2
        c3 = self.c3

        lauxx = lvx - lomega / c3 * cos(theta)
        lauxy = lvy + lomega / c3 * sin(theta)
        laux = sqrt(lauxx**2 + lauxy**2)

        # ut
        ut = [0]*2
        ut[0] = - lauxx / laux
        ut[1] = - lauxy / laux

        # u
        if self.homotopy==1:
            S = 1. - lm - laux * c2 / m
            if S >= 0:
                u=0.
            if S < 0:
                u=1.
        else:
            u = 1. / 2. / c1 / (1.-self.homotopy) * (lm + laux * c2 / m - self.homotopy) 
            u = min(u,1.) # NOTE: this can be increased to help convergence?
            u = max(u,0.)
        return u, ut

    def _eom(self, full_state, t):
        # Applying Pontryagin minimum principle
        state = full_state[:7]
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
        xf, info = odeint(lambda a,b: self._eom(a,b), self.state0 + list(x[:-1]), tspan, rtol=1e-12, atol=1e-12, full_output=1, mxstep=2000)
        return xf, info

    def _non_dim(self, state):
        xnd = deepcopy(state)
        xnd[0] /= self.R
        xnd[1] /= self.R
        xnd[2] /= self.V
        xnd[3] /= self.V
        xnd[4] /= 1.
        xnd[5] *= self.T
        xnd[6] /= self.M
        return xnd

    def _dim_back(self, state):
        xd = deepcopy(state)
        xd[0] *= self.R
        xd[1] *= self.R
        xd[2] *= self.V
        xd[3] *= self.V
        xd[4] *= 1.
        xd[5] /= self.T
        xd[6] *= self.M
        return xd

    def plot(self, x):
        import matplotlib as mpl
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        mpl.rcParams['legend.fontsize'] = 10

        # Producing the data
        tspan = linspace(0, x[-1], 300)
        full_state, info = self._simulate(x, tspan)
        # Putting dimensions back
        res = list()
        controls = list()
        ux = list(); uy=list()
        for line in full_state:
            res.append(self._dim_back(line[:7]))
            controls.append(self._pontryagin_minimum_principle(line))
            ux.append(controls[-1][0] * controls[-1][1][0])
            uy.append(controls[-1][0] * controls[-1][1][1])
        tspan = [it * self.T for it in tspan]

        x = list(); y=list()
        vx = list(); vy = list()
        theta = list()
        omega = list()
        m = list()
        for state in res:
            x.append(state[0])
            y.append(state[1])
            vx.append(state[2])
            vy.append(state[3])
            theta.append(state[4])
            omega.append(state[5])
            m.append(state[6])

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

        axarr[2,0].plot(tspan, theta)
        axarr[2,0].set_xlabel('t'); axarr[2,0].set_ylabel('theta');

        axarr[0,1].plot(tspan, [controls[ix][0] for ix in range(len(controls))],'r')
        axarr[0,1].set_ylabel('u')
        axarr[0,1].set_xlabel('t')
        axarr[1,1].plot(tspan, [controls[ix][1][0] for ix in range(len(controls))],'k')
        axarr[1,1].set_ylabel('sin(ut)')
        axarr[1,1].set_xlabel('t')

        axarr[2,1].plot(tspan, m)
        axarr[2,1].set_xlabel('t'); axarr[2,1].set_ylabel('m');


        plt.ion()
        plt.show()
        return axarr

    def human_readable_extra(self):
        s = "\n\tDimensional inputs:\n"
        s = s + "\tStarting state: " + str(self.state0_input) + "\n"
        s = s + "\tTarget state: " + str(self.statet_input) + "\n"
        s = s + "\tThrusters maximum magnitude [N]: " + str(self.c1 * self.F) + "\n"
        s = s + "\tIsp*g0: " + str(self.c2 * self.V) + ", gravity: " + str(self.g * self.A) + "\n"

        s = s + "\n\tNon-dimensional inputs:\n"
        s = s + "\tStarting state: " + str(self.state0) + "\n"
        s = s + "\tTarget state: " + str(self.statet) + "\n"
        s = s + "\tThrusters maximum magnitude [N]: " + str(self.c1) + "\n"
        s = s + "\tIsp*g0: " + str(self.c2) + ", gravity: " + str(self.g) + "\n\n"
        
        s = s + "\tHomotopy parameter: " + str(self.homotopy)
        s = s + "\n\tPinpoint?: " + str(self.pinpoint)

        return s

if __name__ == "__main__":
    from PyGMO import *
    from random import random
    algo = algorithm.snopt(200, opt_tol=1e-5, feas_tol=1e-5)
    #algo = algorithm.scipy_slsqp(max_iter = 1000,acc = 1E-8,epsilon = 1.49e-08, screen_output = True)
    algo.screen_output = False

    # Define the starting area (x0 will be irrelevant if pinpoint is not True)
    x0b = [-1, 1]
    y0b =  [500, 2000]
    vx0b = [-1, 1]
    vy0b = [5, -40]
    m0b =  [8000, 12000]

    x0 = random() * (x0b[1] - x0b[0]) + x0b[0]
    y0 = random() * (y0b[1] - y0b[0]) + y0b[0]
    vx0 = random() * (vx0b[1] - vx0b[0]) + vx0b[0]
    vy0 = random() * (vy0b[1] - vy0b[0]) + vy0b[0]
    m0 = random() * (m0b[1] - m0b[0]) + m0b[0]
    theta0 = 0.
    omega0 = 0.

    state0 = [x0, y0, vx0, vy0, theta0, omega0, m0]

    # Problem definition
    prob = tv_landing(state0 = state0, pinpoint=True, homotopy=0.)

    print("IC: {}".format(state0))
    
    # Attempting to solve the QC problem
    n_attempts = 1
    for i in range(1, n_attempts + 1):
        # Start with attempts
        print("Attempt # {}".format(i), end="")
        pop = population(prob)
        pop.push_back([0,0,0,-0.015,0,0,0,5])
        #pop.push_back(x0)
        pop = algo.evolve(pop)

        # Log constraints and chormosome
        print("\nc: ",end="")
        print(["{0:.2g}".format(it) for it in pop[0].cur_c])

        print("x: ",end="")
        print(["{0:.2g}".format(it) for it in pop[0].cur_x])

        # If succesfull proceed
        if (prob.feasibility_x(pop[0].cur_x)):
            break

    if not prob.feasibility_x(pop[0].cur_x):
        print("No QC solution! Ending here :(")
        sys.exit(0)
    else: 
        print("Found QC solution!! Starting Homotopy")
        x = pop[0].cur_x
        print("state0 = {}".format(state0))
        print("x = {}".format(x))
    
    #sys.exit(0)

    # We proceed to solve by homotopy the mass optimal control
    # Minimum and maximum step for the continuation
    h_min = 1e-4
    h_max = 0.2
    # Starting step
    h = 0.2

    trial_alpha = h
    alpha = 0
    x = pop[0].cur_x

    algo = algorithm.scipy_slsqp(max_iter = 40,acc = 1E-8,epsilon = 1.49e-08, screen_output = True)
    algo.screen_output = False
    while True:
        if trial_alpha > 1:
            trial_alpha = 1.
        print("{0:.5g}, \t {1:.5g} \t".format(alpha, trial_alpha), end="")
        print("({0:.5g})\t".format(h), end="")
        prob = tv_landing(state0 = state0, pinpoint=True, homotopy=trial_alpha)

        pop = population(prob)
        pop.push_back(x)
        pop = algo.evolve(pop)

        if (prob.feasibility_x(pop[0].cur_x)):
            x = pop[0].cur_x
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

