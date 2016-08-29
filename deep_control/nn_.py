import pickle
import numpy as np
import theano
import theano.tensor as tensor
from lasagne import layers, objectives, nonlinearities, updates
try:
    from tqdm import tqdm
except:
    def tqdm(*args, **kwargs):
        if args:
            return args[0]
        return kwargs.get('iterable', None)

import os

import json

OUTPUT_LOG = 0
OUTPUT_NO = 1
OUTPUT_BOUNDED = 2
output_mode_names = ['outputLog', 'outputLin', 'outputBOUN']

THRUST = 0
THRUST_L = 1
THRUST_R = 2
DTHETA = 1
control_names = ['thrust', 'dtheta']


def get_name(model):
    name = 'nets/'

    name += model['data'] + '/'
    name += str(model['control']) + '/'
    name += model['hidden_nonlinearity']
    name += '_' + output_mode_names[model['output_mode']]
    name += '_' + str(model['nlayers'])
    name += '_' + str(model['units'])

    if model['dropout']:
        name += '_dropout'

    name += '.net'

    return name


def load_minibatch(x, y, index, batch_size, random_idx=None):

    if not random_idx:
        random_idx = range(y.shape[0])

    index %= (len(random_idx)/batch_size)
    index = int(index)
    xt = x[random_idx[batch_size*index:(index+1)*batch_size],:]
    yt = y[random_idx[batch_size*index:(index+1)*batch_size],:]
#    xt = x.take(random_idx[batch_size*index:(index+1)*batch_size], axis=0)
#    yt = y.take(random_idx[batch_size*index:(index+1)*batch_size], axis=0)

    return xt.reshape((batch_size, -1)), yt.reshape((batch_size, -1))


def create_norm(data):

    if len(data.shape) < 2:
        return data.mean(), data.std()

    xmeans = []
    xstds = []
    for i in range(data.shape[1]):
        xmeans.append(data[:, i].mean())
        xstds.append(data[:, i].std())

    return xmeans, xstds


def apply_norm(data, norm):

    if len(data.shape) < 2:
        return (data - norm[0])/norm[1]

    for i in range(data.shape[1]):
        data[:, i] = (data[:, i] - norm[0][i])/norm[1][i]

    return data


def apply_unnorm(data, norm):

    if len(data.shape) < 2:
        return (data * norm[1]) + norm[0]

    for i in range(data.shape[1]):
        data[:, i] = (data[:, i] * norm[1][i]) + norm[0][i]

    return data


def save_network(network, filename):
    params = layers.get_all_param_values(network)
    pickle.dump(params, open(filename, 'wb'))


def load_network_weights(network, filename):
    pas = pickle.load(open(filename, 'rb'))
    ls = layers.get_all_layers(network)
    for i, l in enumerate(ls[1:], 0):
        l.W.set_value(pas[(i*2)].astype(np.float32))
        l.b.set_value(pas[(i*2)+1].astype(np.float32))
    return network


def get_network(model):

    input_data = tensor.dmatrix('x')
    targets_var = tensor.dmatrix('y')

    network = layers.InputLayer((model['batch_size'], model['input_vars']), input_data)

    nonlin = nonlinearities.rectify
    if model['hidden_nonlinearity'] != 'ReLu':
        nonlin = nonlinearities.tanh

    prev_layer = network

    for l in range(model['nlayers']):
        fc = layers.DenseLayer(prev_layer, model['units'], nonlinearity=nonlin)
        if model['dropout']:
            fc = layers.DropoutLayer(fc, 0.5)
        prev_layer = fc

    output_lin = None
    if model['output_mode'] == OUTPUT_LOG:
        output_lin = nonlinearities.tanh
    output_layer = layers.DenseLayer(prev_layer, 1, nonlinearity=output_lin)

    predictions = layers.get_output(output_layer)

    if model['output_mode'] == OUTPUT_BOUNDED:
        (minth, maxth) = model['maxmin'][model['control']]
        maxt = theano.shared(np.ones((model['batch_size'], 1)) * maxth)
        mint = theano.shared(np.ones((model['batch_size'], 1)) * minth)
        predictions = tensor.min(tensor.concatenate([maxt, predictions], axis=1), axis=1)
        predictions = tensor.reshape(predictions, (model['batch_size'], 1))
        predictions = tensor.max(tensor.concatenate([mint, predictions], axis=1), axis=1)
        predictions = tensor.reshape(predictions, (model['batch_size'], 1))

    loss = objectives.squared_error(predictions, targets_var)
    loss = objectives.aggregate(loss, mode='mean')

    params = layers.get_all_params(output_layer)

    test_prediction = layers.get_output(output_layer, deterministic=True)
    test_loss = objectives.squared_error(test_prediction,  targets_var)
    test_loss = test_loss.mean()

    updates_sgd = updates.sgd(loss, params, learning_rate=model['lr'])
    ups = updates.apply_momentum(updates_sgd, params, momentum=0.9)

    train_fn = theano.function([input_data, targets_var], loss, updates=ups)
    pred_fn = theano.function([input_data], predictions)
    val_fn = theano.function([input_data, targets_var], test_loss)

    return {'train': train_fn, 'eval': val_fn, 'pred': pred_fn, 'layers': output_layer}


def preprocess(model, x, y=None):

    if 'X_norm' not in model:
        print('Preprocessing failed, missing normalization values')
        return

    if 'Y_norm' not in model:
        print('Preprocessing failed, missing normalization values or data')
        return

    x = apply_norm(x, model['X_norm'])
    if y is None:
        return x

    y = apply_norm(y, model['Y_norm'])

    if 'maxmin' not in model:
        print('Preprocessing failed, missing normalization values or data')
        return

    for i in range(y.shape[0]):
        data_min, data_max = model['maxmin'][i]
        if model['output_mode'] == OUTPUT_LOG:
            y_i = y[i]
            y_i = ((y_i - data_min)/(data_max - data_min) - 0.5) * 2
            y[i] = y_i
    return x, y


def preprocess_dataset(model, data=None):

    if 'X_norm' not in model:
        if not data:
            print('Preprocessing failed, missing normalization values or data')
            return
        xnorm = create_norm(data['X_train'])
        model['X_norm'] = xnorm

    if 'Y_norm' not in model:
        if not data:
            print('Preprocessing failed, missing normalization values or data')
            return

        ynorm = create_norm(data['Y_train'])
        model['Y_norm'] = ynorm

    data['X_train'] = apply_norm(data['X_train'], model['X_norm'])
    data['X_test'] = apply_norm(data['X_test'], model['X_norm'])

    data['Y_train'] = apply_norm(data['Y_train'], model['Y_norm'])
    data['Y_test'] = apply_norm(data['Y_test'], model['Y_norm'])

    if 'maxmin' not in model:
        if not data:
            print('Preprocessing failed, missing normalization values or data')
            return

        maxmins = []
        for i in range(data['Y_train'].shape[1]):
            data_min = data['Y_train'][:, i].min()
            data_max = data['Y_train'][:, i].max()
            maxmins.append((data_min, data_max))
        model['maxmin'] = maxmins

    for i in range(data['Y_train'].shape[1]):
        data_min, data_max = model['maxmin'][i]
        if model['output_mode'] == OUTPUT_LOG:
            y = data['Y_train'][:, i]
            y = ((y - data_min)/(data_max - data_min) - 0.5) * 2
            data['Y_train'][:, i] = y
            y = data['Y_test'][:, i]
            y = ((y - data_min)/(data_max - data_min) - 0.5) * 2
            data['Y_test'][:, i] = y

    return model, data


def postprocess(model, y):

    y = y.copy()
    
    if len(y.shape) < 2:
        y = y.reshape(1,-1)

    for i in range(y.shape[1]):
        data_min, data_max = model['maxmin'][i]
        if model['output_mode'] == OUTPUT_LOG:
            y[:,i] = (((y[:,i]/2.0)+0.5) * (data_max-data_min) + data_min)
    
    y = apply_unnorm(y, model['Y_norm'])

    return y


def save_training_data(data, dataname):

    if not os.path.exists('traj'):
        os.makedirs('traj')

    pickle.dump(data, open('traj/' + dataname + '.pic', 'wb'))


def load_model(modelfile):

    return pickle.load(open(modelfile, 'rb'))


def load_network(model, base_dir='./'):

    network = get_network(model)
    load_network_weights(network['layers'], base_dir+get_name(model))
    return network
    
    
def load_training_data(model, base_dir='./'):

    [x_train, y_train, x_test, y_test, idx_train] = pickle.load(open(base_dir +'traj/' + model['data'] + '.pic', 'rb'))

    data = {'X_train': x_train,
            'Y_train': y_train,
            'X_test': x_test,
            'Y_test': y_test,
            'idx_train': idx_train}

    model, data = preprocess_dataset(model, data)
    return data


def train(model):

    print('=====  MODEL ===========')
    for key, value in model.items():
        print('{0}:{1}'.format(key, value))

    print('========================')
    print('Loading data...')
    [x_train, y_train, x_test, y_test, idx_train] = pickle.load(open('traj/' + model['data'] + '.pic', 'rb'))

    data = {'X_train': x_train,
            'Y_train': y_train,
            'X_test': x_test,
            'Y_test': y_test,
            'idx_train': idx_train}

    print('=======================')
    print('Preprocessing...')
    model, data = preprocess_dataset(model, data)

    network = get_network(model)

    take_control = model['control']

    print('=======================')
    print('Training...')

    directory = os.path.dirname(get_name(model))
    if not os.path.exists(directory):
        os.makedirs(directory)

    pickle.dump(model, open(directory + '/' + os.path.basename(get_name(model)).split('.')[0] + '.model', 'wb'))
    json.dump(model, open(directory + '/' + os.path.basename(get_name(model)).split('.')[0] + '.modeltxt', 'w'))

    model['epochs_completed'] = 0
    minErr = np.Inf
    minEpoch = 0
    try:
        for epoch in range(model['epochs']):
            epoch_loss = []
            for i in tqdm(range(int(y_train.shape[0]/model['batch_size'])), 'Training, epoch ' + str(epoch), leave=True):
                xt, yt = load_minibatch(data['X_train'], data['Y_train'], i, model['batch_size'], idx_train)
                yt = yt.take([take_control], 1)
                loss = network['train'](xt, yt)
                epoch_loss.append(loss)
            epoch_loss = np.mean(epoch_loss)

            save_network(network['layers'], get_name(model))
            model['epochs_completed'] += 1
            print('epoch {0}, train loss: {1}'.format(epoch, epoch_loss))
            yt = y_test.take([take_control], 1)
            loss = network['eval'](data['X_test'], yt)
            print('epoch {0}, test loss: {1}'.format(epoch, loss))
            model['test_loss'] = float(loss)
            if loss < minErr:
                minErr = loss
                minEpoch = epoch
            if epoch - minEpoch > 5:
                print("Not learning, stopped")
                break
            pickle.dump(model, open(directory + '/' + os.path.basename(get_name(model)).split('.')[0] + '.model', 'wb'))
            json.dump(model, open(directory + '/' + os.path.basename(get_name(model)).split('.')[0] + '.modeltxt', 'w'))

    except KeyboardInterrupt:
        print('Training stopped')
        pass

