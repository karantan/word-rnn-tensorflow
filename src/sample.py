"""."""

from model import Model
# need for un-pickling `saved_args`
from train import TrainConfiguration  # noqa
from six.moves import cPickle

import os
import tensorflow as tf
import yaml


class SampleConfiguration:
    """."""
    def __init__(
        self,
        save_dir,
        sample_generated,
        prime,
        pick,
        width,
        sample,
    ):
        """."""
        self.save_dir = save_dir
        self.sample_generated = sample_generated
        self.prime = prime
        self.pick = pick
        self.width = width
        self.sample = sample


class Sample:
    """."""

    def __init__(self):
        """."""
        with open('config.yml', 'r') as f:
            config = yaml.load(f)

        config = config['sample']
        self.config = SampleConfiguration(
            save_dir=config['save_dir'],
            sample_generated=config['sample_generated'],
            prime=config['prime'],
            pick=config['pick'],
            width=config['width'],
            sample=config['sample'],
        )

    def run(self):
            with open(
                os.path.join(self.config.save_dir, 'config.pkl'), 'rb'
            ) as f:
                saved_args = cPickle.load(f)

            with open(
                os.path.join(self.config.save_dir, 'words_vocab.pkl'), 'rb'
            ) as f:
                words, vocab = cPickle.load(f)

            model = Model(saved_args, True)
            with tf.Session() as sess:
                tf.global_variables_initializer().run()
                saver = tf.train.Saver(tf.global_variables())
                ckpt = tf.train.get_checkpoint_state(self.config.save_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    print(model.sample(
                        sess,
                        words,
                        vocab,
                        self.config.sample_generated,
                        self.config.prime,
                        self.config.sample,
                        self.config.pick,
                        self.config.width
                    ))

if __name__ == '__main__':
    sample = Sample()
    sample.run()
