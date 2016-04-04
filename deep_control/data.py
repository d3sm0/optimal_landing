
from __future__ import print_function

import re
import numpy as np
import subprocess
import shutil
import os
import threading
import glob
import pandas
try:
    from tqdm import tqdm
except:
    def tqdm(*args, **kwargs):
        if args:
            return args[0]
        return kwargs.get('iterable', None)

amplfile = 'ampl'


def __computeampl(modelfile, outfile):
    """ Executes the ampl solver
    :param modelfile: filename of the ampl model description
    :param outfile: filename of the ampl results
    :return:
    """

    with open(outfile, 'w') as of:
        argampl = (amplfile, modelfile)
        popen = subprocess.Popen(argampl, stdout=of)
        popen.wait()

    with open(outfile, 'r') as of:
        outfile_srt = of.read()
    if -1 == outfile_srt.find('Optimal solution found'):
        return False
    return True


def __setconditions(conditions, modelfile):
    """ Updates modelfile with new initial conditions

    :param conditions: dictionary {parameter name : value}
    :param modelfile: ampl model filename
    :return: nothing
    """
    with open(modelfile, 'r') as f:
        results_str = f.read()

    for key, value in conditions.items():
        param_name = 'param ' + key + ':=[^;]*;'
        results_str = re.sub(param_name, 'param ' + key + ':=' + str(value) + ';', results_str)
        with open(modelfile, 'w') as f:
            f.write(results_str)


def _threaded_generate(modelfile, params, n_samples, th_idx=0, replace=False):
    """ Runs the ampl solver with random initial conditions

    :param modelfile: ampl model filename
    :param params: dictionary with the parameters that are randomly set with values [min,max]
    :param n_samples: number of trajectories to generate
    :param th_idx: id of the thread (in case of multithread execution)
    :return: nothing
    """

    modelname = os.path.basename(modelfile).split('.')[0]
    datadir = 'data/' + modelname

    outfile = "out/outfile_" + modelname + str(th_idx)
    resultfile = 'out/sol_' + modelname + '.out' + str(th_idx)

    th_modelfile = 'models/threaded/' + modelname + str(th_idx)

    shutil.copyfile(modelfile, th_modelfile)

    with open(th_modelfile, 'r') as f:
        filedata = f.read()

    newdata = filedata.replace("out/sol.out", resultfile)

    with open(th_modelfile, 'w') as f:
        f.write(newdata)

    trajs_found = th_idx*n_samples
    while trajs_found < (th_idx+1)*n_samples:
#        if 0 == ((trajs_found-th_idx*n_samples) % 1000):
#                print('{0}: {1} of {2}'.format(th_idx, trajs_found-th_idx*n_samples, n_samples))

        if os.path.isfile(datadir+'/{0:010}'.format(trajs_found)+'.data'):
            trajs_found +=1
            continue

        conditions = {}
        for key, value_range in params.items():
            conditions[key] = value_range[0] + (value_range[1]-value_range[0])*np.random.rand()

        __setconditions(conditions, th_modelfile)
        feasible = __computeampl(th_modelfile, outfile)
        if feasible:
            shutil.copyfile(resultfile, datadir+'/{0:010}'.format(trajs_found)+'.data')
            trajs_found += 1


def get_trajectory(modelfile, conditions, cols=None, col_names=None):

        outfile = 'outfile'
        if not os.path.exists('out'):
            os.makedirs('out')

        __setconditions(conditions, modelfile)

        feasible = __computeampl(modelfile, outfile)

        if feasible:
            traj = load_trajectory('out/sol.out', cols, col_names)
            return traj
        else:
#            print('Starting location infeasible')
            return None


class _TrajGenerator(threading.Thread):
    """threaded_generate class wrapper
    """
    def __init__(self, thread_id, modelfile, params, n_samples):
        threading.Thread.__init__(self)
        self.threadID = thread_id
        self.modelfile = modelfile
        self.params = params
        self.n_samples = n_samples

        modelname = os.path.basename(self.modelfile).split('.')[0]

        if not os.path.exists('out'):
            os.makedirs('out')
        if not os.path.exists('data'):
            os.makedirs('data')
        if not os.path.exists('models/threaded'):
            os.makedirs('models/threaded')
        datadir = 'data/' + modelname
        if not os.path.exists(datadir):
            os.makedirs(datadir)

    def run(self):
        _threaded_generate(self.modelfile, self.params, self.n_samples, self.threadID)


def generate_data(modelfile, params, n_samples, n_threads=1):

    """ Runs the ampl solver with random initial conditions (multithread)

    :param modelfile: ampl model filename
    :param params: dictionary with the parameters that are randomly set with values [min,max]
    :param n_samples: number of trajectories to generate
    :param n_threads: number of threads to use

    :return: nothing
    """
    samples_per_thread = n_samples/n_threads

    threads = []
    for i in range(n_threads):
        threadi = _TrajGenerator(i, modelfile, params, samples_per_thread)
        threads.append(threadi)

    for t in threads:
        t.start()


def load_trajectories(data_dir, cols=None, col_names=None, n=None):

    traj_files = glob.glob(data_dir+'/*.data')
    dfs = []
    if n:
        traj_files = traj_files[:n]
    for f in tqdm(traj_files, 'Loading trajectories', leave=True):
        df = load_trajectory(f, cols, col_names)
        if df is not None:
            dfs.append(df)
    return dfs


def load_trajectory(data_file, cols=None, col_names=None):

    if cols:
        df = pandas.read_csv(data_file, header=None, sep=' ', usecols=cols)
    else:
        df = pandas.read_csv(data_file, header=None, sep=' ')

    if col_names:
        df.columns = col_names

    return df


def create_training_data(trajs, train_p=0.7, n_outputs=1, first_node=True, last_node=True):

    nodes = trajs[0].shape[0]
    init = 0

    if not last_node:
        nodes -= 1

    if not first_node:
        nodes -= 1
        init = 1

    n_vars = trajs[0].shape[1] - n_outputs - 1

    n_trajs = len(trajs)

    trajs_train = trajs[:int(n_trajs*train_p)]
    trajs_test = trajs[int(n_trajs*train_p):]

    train_samples = nodes * len(trajs_train)
    test_samples = nodes * len(trajs_test)

    x_train = np.zeros((train_samples, n_vars))
    x_test = np.zeros((test_samples, n_vars))

    y_train = np.zeros((train_samples, n_outputs))
    y_test = np.zeros((test_samples, n_outputs))

    j = 0

    for t in tqdm(trajs_train, 'Creating training data', leave=True):
        # From 1 to 1+n_vars (0 is always time)
        data_in = t.values[init:init+nodes, 1:1+n_vars]
        data_out = t.values[init:init+nodes, -n_outputs:]
        for i in range(data_in.shape[0]):
            x_train[j, :] = data_in[i, :]
            y_train[j, :] = data_out[i, :]
            j += 1

    j = 0
    for t in tqdm(trajs_test, 'Creating test data', leave=True):
        # From 1 to 1+n_vars (0 is always time)
        data_in = t.values[init:init+nodes, 1:1+n_vars]
        data_out = t.values[init:init+nodes, -n_outputs:]
        for i in range(data_in.shape[0]):
            x_test[j, :] = data_in[i, :]
            y_test[j, :] = data_out[i, :]
            j += 1

    idx_train = list(range(x_train.shape[0]))
    np.random.shuffle(idx_train)

    return x_train, y_train, x_test, y_test, idx_train
