import matplotlib.pyplot as plt
from matplotlib import animation
from math import sin, cos
import seaborn as sns
import numpy as np

sns.set_style("whitegrid")
sns.set_context("talk")
sns.set_style("ticks", {"xtick.major.size": 4, "ytick.major.size": 4})


def vis_control(traj, shadow_last=0):

    if 't' in traj:
        t = traj['t']
        xlabel = 't'
    else:
        t = list(range(len(traj)))
        xlabel = ''

    columns = [c for c in traj.columns if c != 't']

    plot_columns = 3
    plot_rows = int(len(columns)/3)

    if len(columns) % plot_columns > 0:
        plot_rows += 1

    plt.rcParams['figure.figsize'] = (10, 3*plot_rows)
    fig, axes = plt.subplots(nrows=plot_rows, ncols=plot_columns)
    fig.tight_layout()

    for i, c in enumerate(columns):
        bg_color = None
        if i > (len(columns)-shadow_last-1):
            bg_color = 'lightgray'
        plt.subplot(plot_rows, plot_columns, i+1, axisbg=bg_color)
        plt.locator_params(nbins=2)
        plt.xlabel(xlabel)
        plt.ylabel(c)
        plt.plot(t, traj[c])
        plt.xlim((t.iloc[0], t.iloc[-1]))
        plt.ylim((min(traj[c]), max(traj[c])))

    for i in range(len(columns),plot_rows*plot_columns):
        plt.subplot(plot_rows,plot_columns,i+1)
        plt.axis('off')

    plt.tight_layout()

def compare_control(traj, traj_comp, shadow_last=0,plot_columns = 4, order=None):

    xlabel = 't'
    if 't' in traj:
        t = traj['t']
    else:
        print('Time is needed for control comparison')
        return None

    if 't' in traj_comp:
        t_comp = traj_comp['t']
    else:
        print('Time is needed for control comparison')
        return None

    columns = [c for c in traj.columns if c != 't']
    if not order == None:
        columns = [columns[i] for i in order]

    plot_rows = int(len(columns)/plot_columns)

    if len(columns) % plot_columns > 0:
        plot_rows += 1

    sns.set(font_scale=1.8)
    sns.set_style("whitegrid")
    sns.set_style("ticks", {"xtick.major.size": 4, "ytick.major.size": 4})

    plt.rcParams['figure.figsize'] = (12, 5 *plot_rows)

    fig, axes = plt.subplots(nrows=plot_rows, ncols=plot_columns)
    sns.set_context(font_scale=2)


    for i, c in enumerate(columns):
        bg_color = None
        if i > (len(columns)-shadow_last-1):
            bg_color = 'lightgray'
        plt.subplot(plot_rows, plot_columns, i+1, axisbg=bg_color)
        plt.xlabel(xlabel)
        plt.ylabel(c)
        l1, = plt.plot(t, traj[c])
        l2, = plt.plot(t_comp, traj_comp[c], c=sns.color_palette()[2])
        plt.locator_params(nbins=4)
        r = (max(traj[c])) - (min(traj[c]))
        plt.ylim((min(traj[c])-0.1*r, max(traj[c])+0.1*r))
        plt.xlim((t.iloc[0], t.iloc[-1]))

    plt.figlegend([l1,l2],['Optimal control', 'DNN control'],loc = 'upper center', ncol=2,prop={'size':20}, bbox_to_anchor=(0.5, 1.01 ))

    for i in range(len(columns),plot_rows*plot_columns):
        plt.subplot(plot_rows,plot_columns,i+1)
        plt.axis('off')

    plt.tight_layout()
    return fig

def rotate_around(points, c, angle):
    output = []
    for p in points:
        p_translated = ((p[0] - c[0]),  (p[1] - c[1]))
        p_final = [0, 0]
        p_final[0] = p_translated[0] * cos(angle) + p_translated[1] * sin(angle) + c[0]
        p_final[1] = -p_translated[1] * cos(angle) + p_translated[0] * sin(angle) + c[1]
        output.append(p_final)
    return output


def vis_trajectory(traj, show_ground=False, angle_markers=False, x_lbl='x', y_lbl='z', angle_lbl='theta'):
    plt.rcParams['figure.figsize'] = (7,5)

    x = traj[x_lbl]
    z = traj[y_lbl]

    scale = np.abs(x.values).max()/20

    limx = max(abs(min(x)), abs(max(x))) + 2*scale
    limz = max(z) + 2*scale

    ax = plt.axes(xlim=(-limx, limx), ylim=(-3*scale, limz))

    if show_ground:
        ground = plt.Rectangle([-limx, -3*scale], 2*limx, 3*scale, color='gray')
        ax.add_patch(ground)

    plt.scatter(x[0], z[0], c='red', s=30)
    plt.scatter(x.iloc[-1], z.iloc[-1], c='green', s=30)

    plt.plot(x, z)

    if angle_markers:
        marker_lenght = 0.5*scale
        angle = traj[angle_lbl]

        for i in range(len(x)):
            a = angle[i]

            c = (x[i], z[i])

            c_ini = [c[0] - marker_lenght, c[1]]
            c_fin = [c[0] + marker_lenght, c[1]]
            [c_ini, c_fin] = rotate_around([c_ini, c_fin], c, a)

            plt.plot([c_ini[0], c_fin[0]], [c_ini[1], c_fin[1]], c='gray')

    plt.xlabel(x_lbl)
    plt.ylabel(y_lbl)


def get_animation(df, max_thrust=20.0, spacecraft_dims=(1, 0.25), craft_type='quadrotor', scale=1, interv=200):

    max_thrust /= scale
    x = df['x']
    z = df['z']
    dx = df['vx']
    dz = df['vz']

    theta = df['theta']
    theta = -theta

    thrust = df['thrust']

    fig = plt.figure()
    fig.set_dpi(100)
    fig.set_size_inches(7, 6.5)   

    limx = max(abs(min(x)), abs(max(x))) + 5
    limz = max(z) + 2

    if craft_type == 'spacecraft':
        limx += 100
        limz += 100
   
    ax = plt.axes(xlim=(-limx, limx), ylim=(-3, limz))
    ground = plt.Rectangle([-limx, -3], 2*limx, 3, color='gray')
    ax.add_patch(ground)

    plt.plot(x, z, color='gray', zorder=-1)

    w, h = spacecraft_dims
    w *= scale
    h *= scale
    
    null_polygon = [[0, 0], [0, 0], [0, 0], [0, 0]]
    spacecraft = plt.Polygon(null_polygon, fc='black')
    thruster = plt.Polygon(null_polygon, fc='silver')
    thrust_out = plt.Polygon(null_polygon, fc='orangered', edgecolor='none')

    vel = plt.Line2D([], [], color='dodgerblue', linewidth=2, marker='o', markevery=2)
    
    patches = [spacecraft, thruster, thrust_out, vel]

    def init():

        ax.add_patch(patches[0])
        ax.add_patch(patches[1])
        ax.add_patch(patches[2])
        ax.add_patch(patches[3])

        return patches,

    def animate(i):
        spacecraft_points = [[x[i] - w / 2.0, z[i] + h / 2.0],
                             [x[i] + w / 2.0, z[i] + h / 2.0],
                             [x[i] + w / 2.0, z[i] - h / 2.0],
                             [x[i] - w / 2.0, z[i] - h / 2.0]]
        spacecraft_points = rotate_around(spacecraft_points, (x[i], z[i]), theta[i])
        patches[0].set_xy(spacecraft_points)
        
        thrust_points = [[x[i] - w/2.0/4.0, z[i]+h],
                         [x[i] + w/2.0/4.0, z[i]+h],
                         [x[i] + w/2.0/4.0, z[i]],
                         [x[i] - w/2.0/4.0, z[i]]]

        thrust_points = rotate_around(thrust_points, (x[i], z[i]), theta[i])
        patches[1].set_xy(thrust_points)

        if craft_type == 'quadrotor':
                    thrust_points = [[x[i] - w/2.0/4.0 + w/2.0/4.0/4.0, z[i] - h/2 - thrust[i]*2.0/max_thrust],
                                     [x[i] + w/2.0/4.0 - w/2.0/4.0/4.0, z[i] - h/2 - thrust[i]*2.0/max_thrust],
                                     [x[i] + w/2.0/4.0 - w/2.0/4.0/4.0, z[i] - h/2],
                                     [x[i] - w/2.0/4.0 + w/2.0/4.0/4.0, z[i] - h/2]]

        if craft_type == 'spacecraft':
                    thrust_points = [[x[i] - w/2.0/4.0 + w/2.0/4.0/4.0, z[i] + h],
                                     [x[i] + w/2.0/4.0 - w/2.0/4.0/4.0, z[i] + h],
                                     [x[i] + w/2.0/4.0 - w/2.0/4.0/4.0, z[i] + h + thrust[i]*2.0/max_thrust],
                                     [x[i] - w/2.0/4.0 + w/2.0/4.0/4.0, z[i] + h + thrust[i]*2.0/max_thrust]]

        thrust_points = rotate_around(thrust_points, (x[i], z[i]), theta[i])
        patches[2].set_xy(thrust_points)
        
        vel_x = [x[i]+w, x[i]+dx[i]*scale*0.3+w]
        vel_z = [z[i], z[i]+dz[i]*0.3*scale]        
        patches[3].set_data(vel_x, vel_z)

        return patches

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(x), interval=interv)

    return anim


def get_animation_pendulum(df, interv, scale=1):

    x = df.values[:, 1]
    thetas = df.values[:, 3]

    fig = plt.figure()
    fig.set_dpi(100)
    fig.set_size_inches(7, 6.5)

    limx = max(abs(min(x)), abs(max(x))) + 5
    limz = limx

    ax = plt.axes(xlim=(-limx, limx), ylim=(-3, limz))
    ground = plt.Rectangle([-limx, -3], 2*limx, 3, color='gray')
    ax.add_patch(ground)

    l = 1*scale
    w, h = 2, 0.5
    w *= scale
    h *= scale
    center = (0, 0)
    pendulum = plt.Circle(xy=center, radius=0.3*scale, color='dodgerblue')
    null_polygon = [[0, 0], [0, 0]]
    pendulum_line = plt.Polygon(null_polygon, fc='black')
    cart = plt.Polygon(null_polygon, fc='black')

    patches = [cart, pendulum, pendulum_line]

    def init():

        ax.add_patch(patches[0])
        ax.add_patch(patches[1])
        ax.add_patch(patches[2])

        return patches,

    def animate(i):

        cart_points = [[x[i] - w/2.0, h],
                       [x[i] + w/2.0, h],
                       [x[i] + w/2.0, 0],
                       [x[i] - w/2.0,  0]]

        patches[0].set_xy(cart_points)
        theta = thetas[i]
        cx, cy = (l*sin(theta)+x[i], l*cos(theta)+h/2.0)
        patches[1].center = (cx, cy)

        patches[2].set_xy([[x[i], h/2], [cx, cy]])
        return patches,

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(x), interval=interv)

    return anim
