import numpy as np
from glob import glob
from tqdm import tqdm
import pickle


class Dataset(object):
    def __init__(self, data, batch_size=64, shuffle=True):
        self.data = data
        self.enable_shuffle = shuffle
        self.n = next(iter(data.values())).shape[0]
        self._next_id = 0
        self.batch_size = batch_size
        if self.enable_shuffle:
            self.shuffle()

    def shuffle(self):
        perm = np.arange(self.n)
        np.random.shuffle(perm)
        for key in self.data.keys():
            self.data[key] = self.data[key][perm]
        self._next_id = 0

    def next_batch(self):
        if self._next_id >= self.n and self.enable_shuffle:
            self.shuffle()

        curr_id = self._next_id
        curr_batch_size = min(self.batch_size, self.n - self._next_id)
        self._next_id += curr_batch_size
        data = dict()
        for key in self.data.keys():
            data[key] = self.data[key][curr_id:curr_id + curr_batch_size]
        return data

    def iterate_once(self):
        if self.enable_shuffle:
            self.shuffle()
        while self._next_id <= self.n - self.batch_size:
            yield self.next_batch()
        self._next_id = 0


def split_data(dataset, n_outputs):
    n_features = dataset.shape[1] - n_outputs
    # fitting time is not a bad idea tho
    x = dataset[:, 1:n_features]
    y = dataset[:, -n_outputs:]
    return x, y


def train_test_split(x,y,train_p=.8):
    x_train = x[:int(train_p * len(x))]
    y_train = y[:int(train_p * len(x))]
    x_test = x[int(train_p) * len(x):]
    y_test = y[int(train_p) * len(x):]
    return dict(x=x_train, y=y_train), dict(x=x_test, y=y_test)


def bound(y):
    y = ((y - y.min() / y.max() - y.min()) - .5) * 2
    return y, (y.min(), y.max())


def normalize(x, mu=None, std=None):
    mu = x.mean(axis=0) if mu is None else mu
    std = x.std(axis=0) if std is None else std
    return (x - mu) / std, (mu, std)


def preprocess(train_set, stats=(None, None)):
    x, x_stats = normalize(train_set['x'], *stats)
    y, y_stats = normalize(train_set['y'], *stats)
    y, bounds = bound(y)
    return dict(x=x, y=y), (x_stats, y_stats)


def postprocess(y, stats, bound):
    mu, std = stats
    _min, _max = bound
    y = (((y / 2) + .5) * (_max - _min) + _min)
    y = mu + y * std
    return y


def load_data(data_dir):
    try:
        with open(data_dir + '.pic', 'rb') as fin:
            dataset = pickle.load(fin)
            return dataset
    except IOError:
        print("Should load all files again")


def load_files(data_dir):
    files = glob(data_dir + '/*pic')
    trajs = []
    for f in tqdm(files, leave=True):
        with open(f, 'rb') as fin:
            rw = pickle.load(fin)
        for r in rw:
            traj = np.hstack((r[0], r[1]))  # features and actions
            # col_names = ['t', 'x', 'y', 'z', 'vz', 'm','theta' 'u1', 'u2'] # theta doesn't apply to simple landing
            trajs.append(traj)

    return trajs, np.concatenate(trajs)


def mass_optimal(nn_m, opt_m):
    goal = nn_m[0, -1] - nn_m[-1, -1]
    opt_goal = opt_m[0, -1] - opt_m[-1, -1]
    return abs((goal - opt_goal) / opt_goal)


def distance(s0, s1):
    # x,y, vx, vy, theta, m
    dist_goal = np.linalg.norm(s1[:2] - s0[:2]) + np.linalg.norm(s1[2:4] - s0[2:4])
    if len(s0) > 4:
        dist_goal += np.abs(s1[4] - s0[4])
    return dist_goal
