"""
Implements an indirect method to solve the optimal control problem of a 
varying mass spacecraft. The spacecraft is also equipped with a
reaction wheel to control its attitude

Dario Izzo 2016

"""

from PyGMO.problem._base import base
from numpy.linalg import norm
from math import sqrt, sin, cos, atan2, pi
from scipy.integrate import odeint
from numpy import linspace
from numpy.linalg import norm
from copy import deepcopy
import sys


class rw_landing(base):
    def __init__(
            self,
            state0 = [0., 1000., 20., -5., 0., 10000.],
            statet = [0., 0., 0., 0., 0., 9758.695805],
            c1 = 44000.,
            c2 = 311. * 9.81,
            c3 = 0.0698,
            g = 1.6229,
            homotopy = 0.,
            pinpoint = False
            ):
        """
        USAGE: rw_landing(self, start, end, Isp, Tmax, mu):

        * state0: initial state [x, y, vx, vy, theta, m] in m, m , m/s, m/s, rad, kg
        * statet: target state [x, y, vx, vy, theta, m] in m, m, m/s, m/s, rad, kg
        * c1: maximum thrusts for the main thruster (N)
        * c2: Isp g0 (m/s)
        * c3: maximum rate for the attitude change (theta dot) [rad / s]
        * g: planet gravity [m/s**2]
        * homotopy: homotopy parameter, 0->QC, 1->MOC
        * pinpoint: if True toggles the final constraint on the landing x
        """

        super(rw_landing, self).__init__(7, 0, 1, 7, 0, 1e-6)

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
        self.c3 = c3 * self.T
        self.g = g / self.A

        # We compute the initial and final state in the new units
        self.state0 = self._non_dim(self.state0_input)
        self.statet = self._non_dim(self.statet_input)

        # We set the bounds (these will only be used to initialize the population)
        self.set_bounds([-1] * 6 + [10. / self.T], [1] * 6 + [200. / self.T])

        # Activates a pinpoint landing
        self.pinpoint = pinpoint
        self.homotopy = homotopy
        self.alpha = 1./150.

    def _objfun_impl(self, x):
        return(1.,) # constraint satisfaction, no objfun

    def _compute_constraints_impl(self, x):
        # Perform one forward shooting
        xf, info = self._shoot(x)

        # Assembling the equality constraint vector
        ceq = list([0]*7)

        # Final conditions
        if self.pinpoint:
            #Pinpoint landing x is fixed lx is free
            ceq[0] = (xf[-1][0] - self.statet[0] ) * 100
        else:
            #Transversality condition: x is free lx is 0
            ceq[0] = xf[-1][6] * 100

        ceq[1] = (xf[-1][1] - self.statet[1] ) * 100
        ceq[2] = (xf[-1][2] - self.statet[2] ) * 100
        ceq[3] = (xf[-1][3] - self.statet[3] ) * 1000
        ceq[4] = (xf[-1][4] - self.statet[4] ) * 1000
        
        # Transversality condition on mass (free)
        ceq[5] = (xf[-1][11] * 10000) 

        # Free time problem, Hamiltonian must be 0
        ceq[6] = (self._hamiltonian(xf[-1]) * 10000) 

        return ceq

    def _hamiltonian(self, full_state):
        state = full_state[:6]
        costate = full_state[6:]

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
        u1, u2 = controls
        #if self.objfun_type=="MOC":
        #   retval = c1 / c2 * u1 + self.alpha * c3**2 * u2**2
        #elif self.objfun_type=="QC": 
        #   retval = c1**2 / c2 * u1**2 + self.alpha * c3**2 * u2**2
        retval = self.homotopy * c1 / c2 * u1 + (1 - self.homotopy) * c1**2 / c2 * u1**2 + self.alpha * c3**2 * u2**2

        return retval

    def _eom_state(self, state, controls):
        # Renaming variables
        x,y,vx,vy,theta,m = state
        g = self.g
        c1 = self.c1
        c2 = self.c2
        c3 = self.c3
        u1, u2 = controls

        # Equations for the state
        dx = vx
        dy = vy
        dvx = c1 * u1 / m * sin(theta)
        dvy = c1 * u1 / m * cos(theta) - g
        dtheta = c3 * u2
        dm = - c1 / c2 * u1
        return [dx, dy, dvx, dvy, dtheta, dm]

    def _eom_costate(self, full_state, controls):
        # Renaming variables
        x,y,vx,vy,theta,m,lx,ly,lvx,lvy,ltheta,lm = full_state
        c1 = self.c1
        c2 = self.c2
        c3 = self.c3
        u1, u2 = controls

        # Equations for the costate
        lvdotitheta = lvx * sin(theta) + lvy * cos(theta)
        lvdotitau = lvx * cos(theta) - lvy * sin(theta)

        dlx = 0.
        dly = 0.
        dlvx = - lx
        dlvy = - ly
        dltheta = - c1 / m * lvdotitau * u1
        dlm =  c1 / m**2 * lvdotitheta * u1
        
        return [dlx, dly, dlvx, dlvy, dltheta, dlm]

    def _pontryagin_minimum_principle(self, full_state):
        # Renaming variables
        x, y, vx, vy, theta, m, lx, ly, lvx, lvy, ltheta, lm = full_state
        c1 = self.c1
        c2 = self.c2
        c3 = self.c3

        # u1
        lvdotitheta = lvx * sin(theta) + lvy * cos(theta)
        if self.homotopy==1:
            S = 1. - lm + lvdotitheta * c2 / m
            if S >= 0:
                u1=0.
            if S < 0:
                u1=1.
        else:
            u1 = 1. / 2. / c1  * (lm - lvdotitheta * c2 / m - self.homotopy) / (1.-self.homotopy)
            u1 = min(u1,1.) # NOTE: this can be increased to help convergence?
            u1 = max(u1,0.)
        # u2
        u2 = -ltheta/2./c3/self.alpha
        u2 = max(-1, u2)
        u2 = min(1, u2)
        return u1, u2

    def _eom(self, full_state, t):
        # Applying Pontryagin minimum principle
        state = full_state[:6]
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
        xnd[4] /= 1.
        xnd[5] /= self.M
        return xnd

    def _dim_back(self, state):
        xd = deepcopy(state)
        xd[0] *= self.R
        xd[1] *= self.R
        xd[2] *= self.V
        xd[3] *= self.V
        xd[4] /= 1.
        xd[5] *= self.M
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
            res.append(self._dim_back(line[:6]))
            controls.append(self._pontryagin_minimum_principle(line))
            ux.append(controls[-1][0] * sin(line[4]))
            uy.append(controls[-1][0] * cos(line[4]))
        tspan = [it * self.T for it in tspan]

        x = list(); y=list()
        vx = list(); vy = list()
        theta = list()
        m = list()
        for state in res:
            x.append(state[0])
            y.append(state[1])
            vx.append(state[2])
            vy.append(state[3])
            theta.append(state[4])
            m.append(state[5])

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
        axarr[0,1].set_ylabel('u1')
        axarr[0,1].set_xlabel('t')
        axarr[1,1].plot(tspan, [controls[ix][1] for ix in range(len(controls))],'k')
        axarr[1,1].set_ylabel('u2')
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

        s = s + "\n\tNon - Dimensional inputs:\n"
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
    algo = algorithm.snopt(400, opt_tol=1e-3, feas_tol=1e-7)
    #algo = algorithm.scipy_slsqp(max_iter = 1000,acc = 1E-8,epsilon = 1.49e-08, screen_output = True)
    algo.screen_output = True


    vx0b = [-10, 10]
    vy0b = [-30, 10]
    x0b = [-100, 100]
    y0b = [500, 2000]

    # Pinpoint
    x0 = random() * (x0b[1] - x0b[0]) + x0b[0]
    y0 = random() * (y0b[1] - y0b[0]) + y0b[0]
    m0 = random() * (12000. - 8000.) + 8000.
    vx0 = random() * (vx0b[1] - vx0b[0]) + vx0b[0]
    vy0 = random() * (vy0b[1] - vy0b[0]) + vy0b[0]
    theta0 = random() * (pi/20 + pi/20) - pi/20

    state0 = [x0, y0, vx0, vy0, theta0, m0]

    prob = rw_landing(state0 = state0, pinpoint=True, homotopy=0.)
    
    print("IC: {}".format(state0))

    for i in range(1, 10):
        # Start with attempts
        print("Attempt # {}".format(i))
        pop = population(prob, 1)
        #pop.push_back([0,0,0,0,0,0,5.])
        pop = algo.evolve(pop)

        print("c: ",end="")
        print(["{0:.2g}".format(it) for it in pop[0].cur_c])

        print("x: ",end="")
        print(["{0:.2g}".format(it) for it in pop[0].cur_x])

        if (prob.feasibility_x(pop[0].cur_x)):
            break
        
    if not prob.feasibility_x(pop[0].cur_x):
        print("No QC solution! Ending here :(")
        sys.exit(0)
    else: 
        print("Found QC solution!! Starting Homotopy")



    
    # We proceed to solve by homotopy the mass optimal control
    print("from \t to\t step\t result")
    # Minimum and maximum step for the continuation
    h_min = 1e-4
    h_max = 0.2
    # Starting step
    h = 0.2

    #algo = algorithm.snopt(50, opt_tol=1e-3, feas_tol=1e-7)

    trial_alpha = h
    alpha = 0.
    x = pop[0].cur_x


    algo = algorithm.scipy_slsqp(max_iter = 30,acc = 1E-8,epsilon = 1.49e-08, screen_output = True)
    algo.screen_output = False

    while True:
        if trial_alpha > 1:
            trial_alpha = 1.
        print("{0:.5g}, \t {1:.5g} \t".format(alpha, trial_alpha), end="")
        print("({0:.5g})\t".format(h), end="")
        prob = rw_landing(state0 = state0, pinpoint=True, homotopy=trial_alpha)

        pop = population(prob)
        pop.push_back(x)
        pop = algo.evolve(pop)

        if (norm(pop[0].cur_c) < 1e-2):
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