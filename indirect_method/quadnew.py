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
from numpy import linspace, exp
from numpy.linalg import norm
from copy import deepcopy
import sys


class quad(base):
    def __init__(
            self,
            state0 = [-5, 0., 0., 0., 0],
            statet = [0., 0., 0., 0., 0.],
            c1 = 7.5,
            c2 = 2.,
            m = 0.365,
            g = 9.81
            ):

        super(quad, self).__init__(6, 0, 1, 6, 0, 1e-6)

        # We store the raw inputs for convenience
        self.state0 = state0
        self.statet = statet

        # We store the parameters
        self.c1 = c1 
        self.c2 = c2 
        self.g = g 
        self.m = m 

        # We set the bounds (these will only be used to initialize the population)
        self.set_bounds([-1] * 4 + [-10] + [0.1], [1] * 4 + [10] + [20.])

    def _objfun_impl(self, x):
        return(1.,) # constraint satisfaction, no objfun

    def _compute_constraints_impl(self, x):
        # Perform one forward shooting
        xf, info = self._shoot(x)

        # Assembling the equality constraint vector
        ceq = list([0]*6)

        # Final conditions, all the state is specified
        ceq[0] = (xf[-1][0] - self.statet[0] ) 
        ceq[1] = (xf[-1][1] - self.statet[1] )
        ceq[2] = (xf[-1][2] - self.statet[2] ) / 10
        ceq[3] = (xf[-1][3] - self.statet[3] ) / 10
        ceq[4] = (xf[-1][4] - self.statet[4] ) / 100
        
        # Free time problem, Hamiltonian must be 0
        ceq[5] = self._hamiltonian(xf[-1], x, x[-1]) / 1000000

        return ceq

    def _hamiltonian(self, state, dec_vector, t):
        lx0, ly0, lvx0, lvy0, ltheta0, T = dec_vector
        x,y,vx,vy,theta,ltheta = state
        g,m = [self.g, self.m]
        c1 = self.c1
        c2 = self.c2

        lvx = lvx0 - lx0 * t
        lvy = lvy0 - ly0 * t
        # Applying Pontryagin minimum principle
        controls = self._pontryagin_minimum_principle(state, dec_vector, t)
        u1, u2 = controls

        dstate = self._eom(state, controls, dec_vector, t)[:-1]
        costate = [lx0, ly0, lvx, lvy, ltheta]
        H=0.
        for i in range(len(dstate)):
            H += dstate[i] * costate[i]
        H = H + c1**2*u1**2 + c2**2*u2**2
        return H

    def _eom(self, state, controls, dec_vector, t):
        # Renaming variables
        x,y,vx,vy,theta,ltheta = state
        lx0, ly0, lvx0, lvy0, ltheta0, T = dec_vector
        lvx = lvx0 - lx0 * t
        lvy = lvy0 - ly0 * t
        u1, u2 = controls
        g,m = [self.g, self.m]
        c1 = self.c1
        c2 = self.c2

        # Equations 
        dx = vx
        dy = vy
        dvx = c1 * u1 / m * sin(theta)
        dvy = c1 * u1 / m * cos(theta) - g
        dtheta = c2 * u2
        dltheta = -c1*u1*(lvx*cos(theta) - lvy*sin(theta))
        return [dx, dy, dvx, dvy, dtheta, dltheta]

    def _pontryagin_minimum_principle(self, state, dec_vector, t):
        # Renaming variables
        x,y,vx,vy,theta,ltheta = state
        lx0, ly0, lvx0, lvy0, ltheta0, T = dec_vector
        lvx = lvx0 - lx0 * t
        lvy = lvy0 - ly0 * t
        S1 = lvx * sin(theta) + lvy * cos(theta)
        S2 = ltheta
        c1 = self.c1
        c2 = self.c2
        g,m = [self.g, self.m]

        if S1 >= 0:
            u1=0.
        if S1 < 0:
            u1=1.
        if S2 >= 0:
            u2=-1.
        if S2 < 0:
            u2=1.
        u1 = 1. / (1. + exp(-1*S1))  
        u2 = 2. / (1. + exp(-0.01*S2)) - 1. 

        u1 = -S1 / m / 2 / c1
        u2 = -S2 / 2 / c2

        return u1, u2

    def _eom_wrapper(self, state, dec_vector, t):
        # Applying Pontryagin minimum principle
        controls = self._pontryagin_minimum_principle(state, dec_vector, t)
        return self._eom(state, controls, dec_vector, t)

    def _shoot(self, dec_vector):
        # Numerical Integration
        xf, info = odeint(lambda y,t: self._eom_wrapper(y,dec_vector,t), self.state0 + list([dec_vector[-2]]), linspace(0, dec_vector[-1],100), rtol=1e-10, atol=1e-10, full_output=1, mxstep=2000)
        return xf, info

    def _simulate(self, dec_vector, tspan):
        # Numerical Integration
        xf, info = odeint(lambda y,t: self._eom_wrapper(y,dec_vector,t), self.state0 + list([dec_vector[-2]]), tspan, rtol=1e-13, atol=1e-13, full_output=1, mxstep=2000)
        return xf, info

    def plot(self, dec_vector):
        import matplotlib as mpl
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        mpl.rcParams['legend.fontsize'] = 10

        # Producing the data
        tspan = linspace(0, dec_vector[-1], 100)
        full_state, info = self._simulate(dec_vector, tspan)
        # Putting dimensions back
        res = list()
        controls = list()
        ux = list(); uy=list()
        for state, t in zip(full_state, tspan):
            res.append(state[:5])
            controls.append(self._pontryagin_minimum_principle(state, dec_vector, t))
            ux.append(controls[-1][0] * sin(state[4]))
            uy.append(controls[-1][0] * cos(state[4]))
        tspan = [it for it in tspan]

        x = list(); y=list()
        vx = list(); vy = list()
        theta = list()

        for state in res:
            x.append(state[0])
            y.append(state[1])
            vx.append(state[2])
            vy.append(state[3])
            theta.append(state[4])

        fig = plt.figure()
        ax = fig.gca()
        ax.plot(x, y, color='r', label='Trajectory')
        ax.quiver(x, y, ux, uy, label='Thrust', pivot='tail', width=0.001)
        #ax.set_ylim(0,self.state0_input[1]+500)

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

        plt.ion()
        plt.show()
        return axarr

if __name__ == "__main__":
    from PyGMO import *
    from random import random
    algo = algorithm.snopt(400, opt_tol=1e-3, feas_tol=1e-7)
    #algo = algorithm.scipy_slsqp(max_iter = 1000,acc = 1E-8,epsilon = 1.49e-08, screen_output = True)
    algo.screen_output = True

    prob = quad()
    pop = population(prob,1)
    pop = algo.evolve(pop)