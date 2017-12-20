"""."""

from collections import namedtuple
from model import Model
from six.moves import cPickle
from utils import TextLoader

import os
import tensorflow as tf
import time
import yaml


TrainConfiguration = namedtuple(
    'TrainConfiguration',
    [
        'data_dir',
        'input_encoding',
        'log_dir',
        'save_dir',
        'rnn_size',
        'num_layers',
        'model',
        'batch_size',
        'seq_length',
        'num_epochs',
        'save_every',
        'grad_clip',
        'learning_rate',
        'decay_rate',
        'gpu_mem',
        'init_from',
    ],
)


class Train:
    """."""

    def __init__(self):
        """."""
        with open('config.yml', 'r') as f:
            config = yaml.load(f)

        config = config['train']
        self.config = TrainConfiguration(
            data_dir=config['data_dir'],
            input_encoding=config['input_encoding'],
            log_dir=config['log_dir'],
            save_dir=config['save_dir'],
            rnn_size=config['rnn_size'],
            num_layers=config['num_layers'],
            model=config['model'],
            batch_size=config['batch_size'],
            seq_length=config['seq_length'],
            num_epochs=config['num_epochs'],
            save_every=config['save_every'],
            grad_clip=config['grad_clip'],
            learning_rate=config['learning_rate'],
            decay_rate=config['decay_rate'],
            gpu_mem=config['gpu_mem'],
            init_from=config['init_from'],
        )
        self.vocab_size = None

    def run(self):
        data_loader = TextLoader(
            self.config.data_dir,
            self.config.batch_size,
            self.config.seq_length,
            self.config.input_encoding,
        )
        self.vocab_size = data_loader.vocab_size

        # check compatibility if training is continued from previously
        # saved model
        if self.config.init_from is not None:
            # check if all necessary files exist
            assert os.path.isdir(self.config.init_from), (
                '{} must be a path'.format(self.config.init_from)
            )
            assert os.path.isfile(
                os.path.join(self.config.init_from, 'config.pkl')), (
                'config.pkl file does not exist in path {}'.format(
                    self.config.init_from)
            )
            assert os.path.isfile(os.path.join(
                self.config.init_from, 'words_vocab.pkl')
            ), 'words_vocab.pkl.pkl file does not exist in path {}'.format(
                self.config.init_from)
            ckpt = tf.train.get_checkpoint_state(self.config.init_from)
            assert ckpt, 'No checkpoint found'
            assert ckpt.model_checkpoint_path, (
                'No model path found in checkpoint')

            # open old config and check if models are compatible
            with open(
                os.path.join(self.config.init_from, 'config.pkl'), 'rb'
            ) as f:
                saved_model_args = cPickle.load(f)
            need_be_same = ['model', 'rnn_size', 'num_layers', 'seq_length']
            for checkme in need_be_same:
                assert vars(
                    saved_model_args)[checkme] == vars(self)[checkme], (
                    'Command line argument and saved model disagree '
                    'on "{}".'.format(checkme)
                )

            # open saved vocab/dict and check if vocabs/dicts are compatible
            with open(
                os.path.join(self.config.init_from, 'words_vocab.pkl'), 'rb'
            ) as f:
                saved_words, saved_vocab = cPickle.load(f)
            assert saved_words == data_loader.words, (
                'Data and loaded model disagree on word set!')
            assert saved_vocab == data_loader.vocab, (
                'Data and loaded model disagree on dictionary mappings!')

        with open(os.path.join(self.config.save_dir, 'config.pkl'), 'wb') as f:
            cPickle.dump(self, f)
        with open(
            os.path.join(self.config.save_dir, 'words_vocab.pkl'), 'wb'
        ) as f:
            cPickle.dump((data_loader.words, data_loader.vocab), f)

        model = Model(self.config, self.vocab_size)

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(self.config.log_dir)
        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=self.config.gpu_mem)

        with tf.Session(
            config=tf.ConfigProto(gpu_options=gpu_options)
        ) as sess:
            train_writer.add_graph(sess.graph)
            tf.global_variables_initializer().run()
            saver = tf.train.Saver(tf.global_variables())
            # restore model
            if self.config.init_from is not None:
                saver.restore(sess, ckpt.model_checkpoint_path)
            for e in range(model.epoch_pointer.eval(), self.config.num_epochs):
                sess.run(tf.assign(
                    model.lr,
                    self.config.learning_rate * (self.config.decay_rate ** e),
                ))
                data_loader.reset_batch_pointer()
                state = sess.run(model.initial_state)
                speed = 0
                if self.config.init_from is None:
                    assign_op = model.epoch_pointer.assign(e)
                    sess.run(assign_op)
                if self.config.init_from is not None:
                    data_loader.pointer = model.batch_pointer.eval()
                    self.config.init_from = None
                for b in range(data_loader.pointer, data_loader.num_batches):
                    start = time.time()
                    x, y = data_loader.next_batch()
                    feed = {
                        model.input_data: x,
                        model.targets: y,
                        model.initial_state: state,
                        model.batch_time: speed,
                    }
                    summary, train_loss, state, _, _ = sess.run([
                        merged,
                        model.cost,
                        model.final_state,
                        model.train_op,
                        model.inc_batch_pointer_op,
                    ], feed)
                    train_writer.add_summary(
                        summary, e * data_loader.num_batches + b)
                    speed = time.time() - start
                    if (
                        (e * data_loader.num_batches + b) %
                        self.config.batch_size == 0
                    ):
                        print(
                            '{}/{} (epoch {}), train_loss = {:.3f}, '
                            'time/batch = {:.3f}' .format(
                                e * data_loader.num_batches + b,
                                self.config.num_epochs *
                                data_loader.num_batches,
                                e,
                                train_loss, speed,
                            ),
                        )
                    # save for the last result
                    if (
                        (e * data_loader.num_batches + b) %
                        self.config.save_every == 0 or
                        (
                            e == self.config.num_epochs - 1 and
                            b == data_loader.num_batches - 1
                        )
                    ):
                        checkpoint_path = os.path.join(
                            self.config.save_dir, 'model.ckpt')
                        saver.save(
                            sess,
                            checkpoint_path,
                            global_step=e * data_loader.num_batches + b,
                        )
                        print('model saved to {}'.format(checkpoint_path))
            train_writer.close()

if __name__ == '__main__':
    train = Train()
    train.run()
