import tensorflow as tf

import numpy as np


class Model(object):
    def __init__(self, obs_dim, output_dim, config):
        self.name = "base"
        self.output_dim = output_dim
        self.obs_dim = obs_dim
        self.seq_len = config["seq_len"]
        self.gs = tf.train.get_or_create_global_step()
        self._init_ph()
        if config["topology"] == "conv":
            self._conv_model()
        elif config["topology"] == "lstm":
            self._lstm_model(units=32)
        else:
            self._build_graph(units=config["units"])  # config["act"])
        self._loss()
        self._params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        self._train(lr=config["lr"])
        self._summary_op(tensors=[self.loss, self.y_hat])

    def _init_ph(self):
        self.x = tf.placeholder(tf.float32, shape=[None, self.obs_dim], name="x")
        # self.x1 = tf.placeholder(tf.float32, shape=[None, self.obs_dim], name="x1")
        self.y = tf.placeholder(tf.float32, shape=[None, self.output_dim], name="y")
        self.keep_prob = tf.placeholder_with_default(1., (), name="dropout")
        # self.is_training = tf.placeholder_with_default(False, (), name="is_training")

    def _conv_model(self):
        with tf.variable_scope(self.name):
            h = tf.expand_dims(self.x, axis=1)
            h = conv1D(h, num_filters=32, kernel_size=2, stride=2, padding="SAME", act=tf.nn.relu, scope="conv_0")
            h = tf.nn.dropout(h, keep_prob=self.keep_prob)
            h = conv1D(h, num_filters=16, kernel_size=2, stride=2, padding="SAME", act=tf.nn.relu, scope="conv_1")
            h = tf.squeeze(h, axis=1)
            self.y_hat = tf.layers.dense(h, units=self.output_dim, activation=None)

    def _lstm_model(self, units=32):
        with tf.variable_scope(self.name):
            from tensorflow.contrib.rnn import BasicLSTMCell
            cell = BasicLSTMCell(num_units=units)
            h, _ = tf.nn.dynamic_rnn(cell, inputs=tf.expand_dims(self.x, axis=0), time_major=False,
                                     sequence_length=[self.seq_len], dtype=tf.float32)
            h = tf.squeeze(h, axis=0)
            self.y_hat = tf.layers.dense(inputs=h, units=self.output_dim, activation=None, name="dense_0")

    def _build_graph(self, units, act=tf.nn.relu):
        with tf.variable_scope(self.name, initializer=tf.orthogonal_initializer):
            h = self.x
            for idx, u in enumerate(units):
                h = tf.layers.dense(inputs=h, units=u, activation=act, name="dense_{}".format(idx),)
                h = tf.nn.dropout(h, keep_prob=self.keep_prob)
            self.y_hat = tf.layers.dense(h, units=self.output_dim, activation=None,)
            # self.logits = tf.layers.dense(h, units=1, activation=None)  # a little of a hack
            # self.u1 = tf.round(tf.nn.softmax(self.y_hat[:, 0]))
            # self.u1 = tf.reshape(self.u1, (-1, 1))
            # self.u2 = tf.layers.dense(h, units=self.output_dim - 1, activation=None)
            # self.actions = tf.concat((self.u1, tf.tanh(self.y_hat[:, 1:])), axis=-1)
            # self.s_hat = tf.layers.dense(h, units=self.obs_dim, activation=None)[:-1]  # remove the last

    def _loss(self):
        # u1 = self.y[:, 0]
        # u2 = self.y[:, 1:]
        # u1_hat = self.y_hat[:, 0]
        # u2_hat = self.y_hat[:, 1:]
        # l1 = tf.reduce_mean(tf.square(u1 - u1_hat))
        # l2 = tf.reduce_mean(tf.square(tf.atan2(u2_hat[:, 0], u2_hat[:, 1]) - tf.atan2(u2[:, 0], u2[:, 1])))
        # self.loss = l1 + l2
        # u1 = self.y[:, 0]
        # u1 = tf.reshape(u1, (-1, 1))
        # u2 = self.y[:, 1:]
        # l1 = tf.nn.sigmoid_cross_entropy_with_logits(labels=u1, logits=self.logits,
        #                                              name="l1")  # should use sigmoid instead
        # l2 = tf.square(self.u2 - u2)
        # l3 = tf.square(self.s_hat - self.x1)
        # l2 = tf.abs(tf.atan2(u2[:, 1], u2[:, 0]) - tf.atan2(self.u2[:, 1], self.u2[:, 0]), name="l2")
        # self.loss = tf.reduce_mean(l2) + tf.reduce_mean(l1)
        # self.loss = tf.reduce_mean(l2) + tf.reduce_mean(l1)# + tf.reduce_sum(l3)
        self.loss = tf.reduce_mean(tf.square(self.y - self.y_hat))

    def _train(self, lr):
        opt = tf.train.AdamOptimizer(learning_rate=lr)
        self.grads = tf.gradients(self.loss, self._params)
        # may apply clipping grads here or regularization
        self.train = opt.minimize(self.loss, global_step=self.gs)

    def _summary_op(self, tensors):
        ops = []
        for t in tensors:
            if t.get_shape().ndims >= 1:
                ops.append(tf.summary.histogram(name=t.name, values=t))
            elif t.get_shape().ndims < 1:
                ops.append(tf.summary.scalar(name=t.name, tensor=t))
            else:
                print("Shape not found")
        self.summarize = tf.summary.merge_all()


class Trainer(object):
    def __init__(self, obs_dim, output_dim, config):
        self.model = Model(obs_dim, output_dim, config)
        self.n_ep = config['n_ep']
        self.keep_prob = config["keep_prob"]
        self.saver = tf.train.Saver(self.model._params)
        self.writer = tf.summary.FileWriter(logdir=config["log_dir"])
        tf.set_random_seed(config["seed"])
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def train(self, dataset):
        print("Start training....\n")
        try:
            for ep in range(self.n_ep):
                losses = []
                for batch in dataset.iterate_once():
                    loss, _, summary, gs = self.sess.run(
                        [self.model.loss, self.model.train, self.model.summarize, self.model.gs],
                        feed_dict={self.model.x: batch['x'], self.model.y: batch['y'],
                                   self.model.keep_prob: self.keep_prob})
                    self.writer.add_summary(summary, gs)
                    self.writer.flush()
                    losses.append(loss)
                if ep % 2 == 0:
                    print("Ep {}, Loss {}".format(ep, np.mean(losses)))
        except KeyboardInterrupt:
            # should save here
            print("Training ended")

    def test(self, dataset):
        print("Start test ... \n")
        try:
            history = []
            losses = []
            for batch in dataset.iterate_once():
                loss, action, summary, gs = self.sess.run(
                    [self.model.loss, self.model.y_hat, self.model.summarize, self.model.gs],
                    feed_dict={self.model.x: batch["x"], self.model.y: batch["y"]})
                self.writer.add_summary(summary, gs)
                self.writer.flush()
                losses.append(loss)
                traj = np.hstack((batch["x"], action))
                history.append(traj)
            print("Test completed. Loss {}".format(np.mean(losses)))
            return np.concatenate(history)  # to be made as np array
        except KeyboardInterrupt:
            print("Test ended")

    def simulate(self, start_state, tspan, env):
        state = start_state.copy()
        history = []
        for t in tspan:
            action = self.get_action(state)
            state = env.shoot(action=action, start_state=state)  # [0, 0.1] dt
            history.append(np.hstack((state, action)))
        return np.array(history)  # check

    def get_action(self, state):
        # return self.sess.run(self.model.y_hat, feed_dict={self.model.x:[state]}).flatten()
        u1, u2, u3 = self.sess.run(self.model.y_hat, feed_dict={self.model.x: [state]}).flatten()
        return np.array([u1,np.tanh(u2), np.tanh(u3)])# np.clip(u2, -1., 1.), np.clip(u3, -1., .1)])

    def load(self, path):
        try:
            ckpt = tf.train.latest_checkpoint(path)
            self.saver.restore(self.sess, save_path=ckpt)
            return True
        except Exception as e:
            print(e)
            return False

    def save(self, path, gs=None):
        try:
            self.saver.save(self.sess, save_path=path + "/model.ckpt", global_step=gs)
        except Exception as e:
            print(e)


def conv1D(x, num_filters, kernel_size, stride, scope="conv", padding="SAME", act=tf.nn.relu):
    with tf.variable_scope(scope):
        channels = x.get_shape()[-1]
        x = tf.expand_dims(x, axis=1)
        w = tf.get_variable("w", [1, kernel_size, channels, num_filters], initializer=tf.orthogonal_initializer())
        b = tf.get_variable("b", [num_filters], initializer=tf.constant_initializer())
        z = tf.nn.conv2d(x, w, strides=[1, 1, stride, 1], padding=padding) + b
        h = act(z)
        h = tf.squeeze(h, axis=1)
        return h


if __name__ == '__main__':
    from utils import Dataset, train_test_split, load_data, preprocess

    train_set, test_set = train_test_split(load_data(config["data_dir"]), train_p=config["train_p"])
    train_set, stats = preprocess(train_set)  # you should only use stats from your training set to normalize the data
    test_set, _ = preprocess(test_set)
    train_set = Dataset(train_set, batch_size=config["batch_size"], shuffle=False)
    test_set = Dataset(test_set, batch_size=config["batch_size"], shuffle=False)
    trainer = Trainer(obs_dim=train_set.data["x"].shape[1], output_dim=train_set.data['y'].shape[1], config=config)
    trainer.train(dataset=train_set)
    trainer.train(dataset=test_set)
