import argparse
import os
import tensorflow as tf
from model import Model



"""
This script defines hyperparameters.
"""



def configure(test_data_list_, out_dir_, test_step_, test_num_steps_, modeldir_, data_dir_):
	flags = tf.app.flags

	# training
	flags.DEFINE_integer('num_steps', 350000, 'maximum number of iterations')
	flags.DEFINE_integer('save_interval', 5000, 'number of iterations for saving and visualization')
	flags.DEFINE_integer('random_seed', 1234, 'random seed')
	flags.DEFINE_float('weight_decay', 0.0005, 'weight decay rate')
	flags.DEFINE_float('learning_rate', 2.5e-4, 'learning rate')
	flags.DEFINE_float('power', 0.9, 'hyperparameter for poly learning rate')
	flags.DEFINE_float('momentum', 0.9, 'momentum')
	flags.DEFINE_string('encoder_name', 'deeplab', 'name of pre-trained model, res101, res50 or deeplab')
	flags.DEFINE_string('pretrain_file', './modelAugment/model.ckpt-18000', 'pre-trained model filename corresponding to encoder_name')
	flags.DEFINE_string('data_list', './dataset/train.txt', 'training data list filename')

	# validation
	flags.DEFINE_integer('valid_step', 18000, 'checkpoint number for validation')
	flags.DEFINE_integer('valid_num_steps', 32659, '= number of validation samples')
	flags.DEFINE_string('valid_data_list', './dataset/val.txt', 'validation data list filename')

	# prediction / saving outputs for testing or validation
	flags.DEFINE_string('out_dir', out_dir_, 'directory for saving outputs')
	flags.DEFINE_integer('test_step', test_step_, 'checkpoint number for testing/validation')
	flags.DEFINE_integer('test_num_steps', test_num_steps_, '= number of testing/validation samples')
	flags.DEFINE_string('test_data_list', test_data_list_, 'testing/validation data list filename')
	flags.DEFINE_boolean('visual', True, 'whether to save predictions for visualization')

	# data
	flags.DEFINE_string('data_dir', data_dir_, 'data directory')
	flags.DEFINE_integer('batch_size', 15, 'training batch size')
	flags.DEFINE_integer('input_height', 256, 'input image height')
	flags.DEFINE_integer('input_width', 256, 'input image width')
	flags.DEFINE_integer('num_classes', 2, 'number of classes')
	flags.DEFINE_integer('ignore_label', 254, 'label pixel value that should be ignored')
	flags.DEFINE_boolean('random_scale', False, 'whether to perform random scaling data-augmentation')
	flags.DEFINE_boolean('random_mirror', False, 'whether to perform random left-right flipping data-augmentation')

	# log
	flags.DEFINE_string('modeldir', modeldir_, 'model directory')
	flags.DEFINE_string('logfile', 'log.txt', 'training log filename')
	flags.DEFINE_string('logdir', 'log', 'training log directory')

	flags.FLAGS.__dict__['__parsed'] = False
	return flags.FLAGS

def main(_):
	if args.option not in ['train', 'test', 'predict']:
		print('invalid option: ', args.option)
		print("Please input a option: train, test, or predict")
	else:
		# Set up tf session and initialize variables.
		# config = tf.ConfigProto()
		# config.gpu_options.allow_growth = True
		# sess = tf.Session(config=config)
		sess = tf.Session()
		# Run
		model = Model(sess, configure(test_data_list_=args.test_data_list, out_dir_=args.out_dir, test_step_=args.test_step, test_num_steps_=args.test_num_steps, modeldir_=args.modeldir, data_dir_=args.data_dir))
		getattr(model, args.option)()


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('--option', dest='option', type=str, default='predict',
		help='actions: train, test, or predict')
	parser.add_argument('--test_data_list', dest='test_data_list', type=str, default='./dataset/test.txt',
		help='testing/validation data list filename')
	parser.add_argument('--out_dir', dest='out_dir', type=str, default='outputAugmentTest',
		help='directory for saving testing outputs')
	parser.add_argument('--test_step', dest='test_step', type=int, default=18000,
		help='checkpoint number for testing/validation')
	parser.add_argument('--test_num_steps', dest='test_num_steps', type=int, default=32659,
		help='number of testing/validation samples')
	parser.add_argument('--modeldir', dest='modeldir', type=str, default='modelAugment',
		help='model directory')
	parser.add_argument('--data_dir', dest='data_dir', type=str, default='/hdd/wsi_fun/wsi_data/boundary_blocks/for_training',
		help='data directory')
	parser.add_argument('--gpu', dest='gpu', type=str, default='1',
		help='specify which GPU to use')

	args = parser.parse_args()

	# Choose which gpu or cpu to use
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
	tf.app.run()
