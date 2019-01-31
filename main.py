import tensorflow as tf
import os
import argparse
from models.wavegan import wavegan

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir_x', dest='dataset_dir', default='formatteddata', help='path of the dataset')
parser.add_argument('--output_dir_x', dest='output_dir', default='generatedaudio', help='path for generated output')
parser.add_argument('--epoch', dest='epoch', type=int, default=200001, help='# of epoch')
parser.add_argument('--epoch_step', dest='epoch_step', type=int, default=100, help='# of epoch to decay lr')
parser.add_argument('--lamda', dest='lamda', type=int, default=10, help='Wasserstein Distance Multiplier')
parser.add_argument('--generator_learning_rate', dest='glr', type=float, default=.0001,help=' generator learning rate')
parser.add_argument('--discriminator_learning_rate', dest='dlr', type=float, default=.0001,help=' generator learning rate')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=64, help='# images in batch')
parser.add_argument('--num_z_dims', dest='z_dims', type=int, default=5, help='dimensions in z')
parser.add_argument('--num_critic_steps', dest='num_critic_steps', type=int, default=10, help='number of discriminator steps per generator step')
parser.add_argument('--num_seconds', dest='num_seconds', type=int, default=4, help='amount of data in audio sample')
parser.add_argument('--len_audio_sample', dest='a_len', type=int, default=16384*4, help='amount of data in audio sample')
args = parser.parse_args()


def main(_):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    with tf.Session(config=tfconfig) as sess:
        model = wavegan(sess, args)
        model.train(args)
if __name__ == '__main__':
    tf.app.run()
