import numpy as np
from scipy.integrate import odeint

import gym
from gym.spaces import Box


class Base(object):
    def __init__(self, c1=44000., c2=311. * 9.81, c3=0.0698, g=1.6229, m=10000., dt=0.1):
        self.r = 1000.
        self.v = 100.
        self.m = m
        self.t = self.r / self.v
        a = (self.v ** 2) / self.r
        f = self.m * a
        self.c1 = c1 / f
        self.c2 = c2 / self.v
        self.c3 = c3 * self.t
        self.g = g / a
        self.dt = dt

    def scale(self, state):
        raise NotImplementedError()

    def unscale(self, state):
        raise NotImplementedError()

    def shoot(self, action, start_state=None, tspan=None):
        tspan = [0, self.dt] if tspan is None else tspan
        next_state = odeint(self.eom_state, y0=start_state, t=tspan, args=(action,),
                            rtol=1e-13, atol = 1e-13, full_output=False, printmessg=False)[1:]
        return next_state

    def eom_costate(self):
        raise NotImplementedError()

    def eom_state(self, state, t, action):
        raise NotImplementedError()

    def pmp(self):
        raise NotImplementedError()

    def cost(self):
        raise NotImplementedError()

    def hamiltonian(self):
        raise NotImplementedError()

    @staticmethod
    def get_config():
        return dict(c1=44000., c2=311. * 9.81, c3=0.0698, g=1.6229, m=10000., dt=0.1)


class Simple(Base):
    def __init__(self,
                 s0=np.array([0., 1000., 20., -.5, 10000.]),
                 st=np.array([0., 0., 0., 0., 9758.69580]), config=None):
        super().__init__(**config)
        self.s0 = self.scale(s0)
        self.st = self.scale(st)
        self.control_dim = 3
        self.bound = dict(low=np.array((0., -1., -1.)), high=np.array((1., 1., 1.)))

    def eom_state(self, state, t, action):
        x, y, vx, vy, m = state
        u, stheta, ctheta = action

        dx = vx
        dy = vy
        dvx = self.c1 * u / m * stheta
        dvy = self.c1 * u / m * ctheta - self.g
        dm = - self.c1 * u / self.c2
        return [dx, dy, dvx, dvy, dm]

    def scale(self, state):
        s = state.copy()
        s[0] /= self.r
        s[1] /= self.r
        s[2] /= self.v
        s[3] /= self.v
        s[4] /= self.m
        return np.array(s)

    def unscale(self, state):
        s = state.copy()
        s[0] *= self.r
        s[1] *= self.r
        s[2] *= self.v
        s[3] *= self.v
        s[4] *= self.m
        return np.array(s)


class Quad(Base):
    def __init__(self,
                 s0=np.array([0., 1000., 20., -5., 0., 10000.]),
                 st=np.array([0., 0., 0., 0., 0., 9758.695805]), config=None):
        super().__init__(**config)
        self.s0 = self.scale(s0)
        self.st = self.scale(st)
        self.control_dim = 2
        self.bound = dict(low=np.array((0.5, -1.,)), high=np.array((1., 1.)))
        # self.bound = dict(u1=(0.05, 1.), u2=(-1, 1))  # thrust, pitch rate

    def scale(self, state):
        s = state.copy()
        s[0] /= self.r
        s[1] /= self.r
        s[2] /= self.v
        s[3] /= self.v
        s[4] /= 1.
        s[5] /= self.m
        return np.array(s)

    def unscale(self, state):
        s = state.copy()
        s[0] *= self.r
        s[1] *= self.r
        s[2] *= self.v
        s[3] *= self.v
        s[4] *= 1.
        s[5] *= self.m
        return np.array(s)

    def eom_state(self, state, t, action):
        # Renaming variables
        x, y, vx, vy, theta, m = state
        u1, u2 = action

        # Equations for the state
        dx = vx
        dy = vy
        dvx = self.c1 * u1 / m * np.sin(theta)
        dvy = self.c1 * u1 / m * np.cos(theta) - self.g
        dtheta = self.c3 * u2
        dm = - self.c1 / self.c2 * u1
        return [dx, dy, dvx, dvy, dtheta, dm]


class TV(Base):
    def __init__(self,
                 s0=np.array([0., 1000., 20., -5., 0., 0., 10000.]),
                 st=np.array([0., 0., 0., 0., 0., 0., 9758.695805]), config=None):
        super().__init__(**config)
        self.s0 = self.scale(s0)
        self.st = self.scale(st)
        self.control_dim = 3
        # self.phi = np.radians(10)
        self.bound = dict(low=np.array((0., -1., -1.)), high=np.array((1., 1., 1.)))
        # self.bound = dict(u1=(0., 1.), u2=(-self.phi, self.phi))  # thrust, thrust tilt

    def scale(self, state):
        s = state.copy()
        s[0] /= self.r
        s[1] /= self.r
        s[2] /= self.v
        s[3] /= self.v
        s[4] /= 1.  # maybe this should be radiant?
        s[5] *= self.t
        s[6] /= self.m
        return np.array(s)

    def unscale(self, state):
        s = state.copy()
        s[0] *= self.r
        s[1] *= self.r
        s[2] *= self.v
        s[3] *= self.v
        s[4] *= 1.
        s[5] /= self.t
        s[6] *= self.m
        return np.array(s)

    def eom_state(self, state, t, action):
        # Renaming variables
        x, y, vx, vy, theta, omega, m = state
        u, ut_0, ut_1 = action

        tdotit = ut_0 * np.cos(theta) - ut_1 * np.sin(theta)
        # Equations for the state
        dx = vx
        dy = vy
        dvx = self.c1 * u / m * ut_0
        dvy = self.c1 * u / m * ut_1 - self.g
        dtheta = omega
        domega = - self.c1 / self.c3 * u / m * tdotit
        dm = - self.c1 / self.c2 * u
        if m < 1e-4:
            dm = 0
        return [dx, dy, dvx, dvy, dtheta, domega, dm]


class Falcon(TV):
    def __init__(self, config):
        super().__init__(config=config)

    def eom_state(self, state, t, action):
        # Renaming variables
        x, y, vx, vy, theta, omega, m = state
        u, ut_0, ut_1 = action

        tdotit = ut_0 * np.cos(theta) - ut_1 * np.sin(theta)
        # Equations for the state
        dx = vx
        dy = vy
        dvx = self.c1 * u / m * ut_0
        dvy = self.c1 * u / m * ut_1 - self.g
        dtheta = omega
        domega = - self.c1 / self.c3 * u / m * tdotit
        dm = - self.c1 / self.c2 * u
        return [dx, dy, dvx, dvy, dtheta, domega, dm]

    def shoot(self, action, start_state=None, tspan=None):
        # TODO check numeric error
        tspan = [0, self.dt] if tspan is None else tspan
        next_state, info = odeint(self.eom_state, start_state, tspan, rtol=1e-6, atol=1e-6, args=(action,),
                                  mxstep=5000, hmax=0.01, hmin=1e-8, printmessg=True, full_output=True)  # [1:]

        return next_state[1:]


def select_problem(name):
    default = Base.get_config()
    if name == "simple":
        return Simple(config=default)
    elif name == "quad":
        return Quad(config=default)
    elif name == "rw":
        default["c2"] = 44000.
        return Quad(config=default)
    elif name == "tv":
        return TV(config=default)
    elif name == "falcon":
        default["c1"] = 5886000. * 0.3
        default["c3"] = 300.
        default["g"] = 9.81
        default["m"] = 80000.
        return Falcon(default)
    else:
        raise NotImplementedError()


class SimpleLanding(gym.Env):
    metadata = {
        "render.modes": ["human", "rgb_array"],
        "video.frames_per_second": "30",
    }

    def __init__(self, name="quad"):
        self.model = select_problem(name)
        self.state = self.model.s0.copy()
        self.max_steps = 400
        high = np.array([2] * len(self.state))
        self.observation_space = Box(-high, high, dtype=np.float32)
        self.action_space = Box(low=self.model.bound["low"], high=self.model.bound["high"], dtype=np.float32)
        self.viewer = None
        self.last_action = self.action_space.sample()

    def reset(self):
        self.state = self.model.s0.copy()
        self.done = False
        self.t = 0
        self.last_action = self.action_space.sample()
        return self.state

    def step(self, action):
        assert self.done == False
        assert self.action_space.contains(action)
        next_state = self.model.shoot(start_state=self.state, action=action).flatten()
        self.last_action = action
        assert self.observation_space.contains(next_state)
        r = self.did_explode(next_state)
        self.state = next_state
        self.t +=1
        if self.t > self.max_steps:
            self.done = True
        return next_state, r, self.done, None

    def render(self, mode='human'):
        # TODO
        pass

    # def render(self, mode='human'):
    #
    #     w, h = (1., 0.25)
    #     from gym.envs.classic_control import rendering
    #     if self.viewer is None:
    #         self.viewer = rendering.Viewer(500, 500)
    #         self.viewer.set_bounds(left=0, right=5., bottom=0., top=1.5)  # square bound
    #     # state = self.model.scale(self.state)
    #     x, y, vx, vy, theta, m = self.state
    #
    #     v = [(x - w / 2., y + h / 2.),
    #          (x + w / 2., y + h / 2.),
    #          (x + w / 2., y - h / 2.),
    #          (x - w / 2., y - h / 2.)]
    #
    #     w_0 = 10.
    #     v0 = [(- w_0 / 2., 0),
    #           (w_0 / 2., 0),
    #           (w_0 / 2., 0,),
    #           (- w_0 / 2., 0)]
    #     ground = self.viewer.draw_polyline(v0, linewidth=2)
    #     ground.set_color(.2, .2, .2)
    #     # u1, u2 = self.last_action # TODO this is wrong
    #     v = rotate_around(v, (x, y), theta)
    #
    #     spacecraft = self.viewer.draw_polygon(v=v)
    #     spacecraft.set_color(.8, .3, .3)
    #     # transform = rendering.Transform(rotation=theta, translation=(x, y))
    #     # transform = rendering.Transform(translation=(x, y))
    #     # transform.set_rotation(new=)
    #     # spacecraft.add_attr(transform)
    #
    #     # transform.set_rotation()
    #     # transform.set_translation()
    #
    #     # write polys
    #     return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()

    def did_explode(self, state):
        if state[1] < 0:  # if vx  is negative crashed
            self.done = True
            return -1
        elif state[1] < 10 / 1000. and abs(state[0]) < 1 and abs(state[3]) < 1:
            # if vx is small and x, y is 0, it landed
            self.done = True
            return 1
        else:
            return 0


if __name__ == "__main__":
    env = SimpleLanding(name="simple")
    s0 = env.reset()
    rw = 0
    t = 0
    for i in range(env.max_steps):
        action = env.action_space.sample()
        s1, r, done, _ = env.step(action)
        rw += r
        t += 1
        if done:
            break
    env.close()
    print(t, rw)
