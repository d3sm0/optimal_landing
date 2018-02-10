from deep_control.tf_nn import Trainer
from deep_control.utils import *
from envs.simple import SimpleLanding
import random
import matplotlib.pyplot as plt
from pprint import pprint
from IPython import embed

config = {
    "batch_size": 32,
    # "act": tf.nn.relu,
    "keep_prob": .8,
    "units": [128, 64, 32],
    "lr": 1e-3,
    "seq_len": 4,
    "log_dir": "logs",
    "n_ep": 30,
    "train_p": .8,
    "source_files": "generate_data/data/simple_red_0",  # TODO fix path
    "data_dir": "traj/simple_red",
    "load_model": False,
    "topology": "fc",
    "shuffle": True,
    "seed": 123

}
random.seed(config["seed"])

env = SimpleLanding(name="simple")
trajs_list, trajs = load_files(data_dir=config["data_dir"] + "_1")
x, y = split_data(dataset=trajs, n_outputs=3)
x = np.apply_along_axis(env.model.scale, axis=1, arr=x)
# x = x.reshape((-1,10))
# y = y[:len(x)]
train_set, test_set = train_test_split(x, y, train_p=config["train_p"])
# train_set, stats = preprocess(train_set)
# test_set, _ = preprocess(test_set, stats)
train_set = Dataset(train_set, batch_size=config["batch_size"], shuffle=config["shuffle"])
test_set = Dataset(test_set, batch_size=config["batch_size"], shuffle=config["shuffle"])

trainer = Trainer(obs_dim=train_set.data["x"].shape[-1], output_dim=train_set.data['y'].shape[-1], config=config)

if trainer.load(config["log_dir"] + "/model") and config["load_model"] == True:
    print("Model loaded")
else:
    pprint(config)
    trainer.train(dataset=train_set)
    trainer.test(dataset=test_set)
    trainer.save(path=config["log_dir"] + "/model")

stats = dict(rw=[], ts=[], ds=[])
state = env.reset()
history = [state]
rw = 0
ts = 0
done = False
while not done:
    action = trainer.get_action(state)
    next_state, r, done, _ = env.step(action)  # TODO check integrals
    state = next_state.copy()
    history.append(state)
    rw += r
    ts += 1
    if done:
        state = env.reset()
        stats["rw"].append(rw)
        stats["ts"].append(ts)
        stats["ds"].append(distance(state, env.model.st))

pprint(stats)
history = np.array(history)
traj = random.choice(trajs_list)
x, y = split_data(traj, n_outputs=3)
x = np.apply_along_axis(env.model.scale, axis=1, arr=x)
mass_error = mass_optimal(history, x)
s = x[0].copy()
ds = []
thetas = []
us = []
history = []
actions = []
for s1_opt, action_opt in zip(x[1:], y[1:]):
    action = trainer.get_action(s)
    actions.append(action)
    s1 = env.model.shoot(action, start_state=s).flatten()
    d = distance(s1_opt, s1)
    u1, stheta, ctheta = action
    u1_opt, stheta_opt, ctheta_opt = action_opt
    # Trying to measure distance between stearing
    a_nn = np.arctan2(stheta, ctheta)
    a_opt = np.arctan2(stheta_opt, ctheta_opt)
    theta = a_nn - a_opt
    thetas.append(theta)
    ds.append(d)
    history.append(s)
    s = s1.copy()

mass_error = mass_optimal(np.array(history), x)
embed()
