import argparse
import os
import tensorflow as tf
from model import Model



"""
This script defines hyperparameters.
"""



def configure(test_data_list_, out_dir_, test_step_, test_num_steps_, modeldir_, data_dir_, num_steps_, save_interval_, learning_rate_, pretrain_file_, data_list_, batch_size_, input_height_, input_width_, num_classes_, print_color_, log_dir_, log_file_):
	flags = tf.app.flags

	# training
	flags.DEFINE_integer('num_steps', num_steps_, 'maximum number of iterations')
	flags.DEFINE_integer('save_interval', save_interval_, 'number of iterations for saving and visualization')
	flags.DEFINE_integer('random_seed', 1234, 'random seed')
	flags.DEFINE_float('weight_decay', 0.0005, 'weight decay rate')
	flags.DEFINE_float('learning_rate', learning_rate_, 'learning rate')
	flags.DEFINE_float('power', 0.9, 'hyperparameter for poly learning rate')
	flags.DEFINE_float('momentum', 0.9, 'momentum')
	flags.DEFINE_string('encoder_name', 'deeplab', 'name of pre-trained model, res101, res50 or deeplab')
	flags.DEFINE_string('pretrain_file', pretrain_file_, 'pre-trained model filename corresponding to encoder_name')
	flags.DEFINE_string('data_list', data_list_, 'training data list filename')

	# validation
	flags.DEFINE_integer('valid_step', 217000, 'checkpoint number for validation')
	flags.DEFINE_integer('valid_num_steps', 81605, '= number of validation samples')
	flags.DEFINE_string('valid_data_list', './dataAugment/val.txt', 'validation data list filename')

	# prediction / saving outputs for testing or validation
	flags.DEFINE_string('out_dir', out_dir_, 'directory for saving outputs')
	flags.DEFINE_integer('test_step', test_step_, 'checkpoint number for testing/validation')
	flags.DEFINE_integer('test_num_steps', test_num_steps_, '= number of testing/validation samples')
	flags.DEFINE_string('test_data_list', test_data_list_, 'testing/validation data list filename')
	flags.DEFINE_boolean('visual', False, 'whether to save predictions for visualization')

	# data
	flags.DEFINE_string('data_dir', data_dir_, 'data directory')
	flags.DEFINE_integer('batch_size', batch_size_, 'training batch size')
	flags.DEFINE_integer('input_height', input_height_, 'input image height')
	flags.DEFINE_integer('input_width', input_width_, 'input image width')
	flags.DEFINE_integer('num_classes', num_classes_, 'number of classes')
	flags.DEFINE_integer('ignore_label', 255, 'label pixel value that should be ignored')
	flags.DEFINE_boolean('random_scale', False, 'whether to perform random scaling data-augmentation')
	flags.DEFINE_boolean('random_mirror', False, 'whether to perform random left-right flipping data-augmentation')

	# log
	flags.DEFINE_string('modeldir', modeldir_, 'model directory')
	flags.DEFINE_string('logfile', log_file_, 'training log filename')
	flags.DEFINE_string('logdir', log_dir_, 'training log directory')

	# text color
	flags.DEFINE_string('print_color', print_color_, 'color of printed outputs')



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
		model = Model(sess, configure(test_data_list_=args.test_data_list, out_dir_=args.out_dir, test_step_=args.test_step, test_num_steps_=args.test_num_steps, modeldir_=args.modeldir, data_dir_=args.data_dir, num_steps_=args.num_steps, save_interval_=args.save_interval, learning_rate_=args.learning_rate, pretrain_file_=args.pretrain_file, data_list_=args.data_list, batch_size_=args.batch_size, input_height_=args.input_height, input_width_=args.input_width, num_classes_=args.num_classes, print_color_=args.print_color, log_dir_=args.log_dir, log_file_=args.log_file))
		getattr(model, args.option)()


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('--option', dest='option', type=str, default='train',
		help='actions: train, test, or predict')
	parser.add_argument('--test_data_list', dest='test_data_list', type=str, default='./dataset/test.txt',
		help='testing/validation data list filename')
	parser.add_argument('--out_dir', dest='out_dir', type=str, default='output',
		help='directory for saving testing outputs')
	parser.add_argument('--test_step', dest='test_step', type=int, default=350000,
		help='checkpoint number for testing/validation')
	parser.add_argument('--test_num_steps', dest='test_num_steps', type=int, default=81605,
		help='number of testing/validation samples')
	parser.add_argument('--modeldir', dest='modeldir', type=str, default='modelAugment',
		help='model directory')
	parser.add_argument('--data_dir', dest='data_dir', type=str, default='/hdd/wsi_fun/ImageAugCustom/AugmentationOutput',
		help='data directory')
	parser.add_argument('--gpu', dest='gpu', type=str, default='0',
		help='specify which GPU to use')
	parser.add_argument('--num_steps', dest='num_steps', type=int, default=100000,
		help='maximum number of iterations')
	parser.add_argument('--save_interval', dest='save_interval', type=int, default=15000,
		help='number of iterations for saving and visualization')
	parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=2.5e-4,
		help='learning rate')
	parser.add_argument('--pretrain_file', dest='pretrain_file', type=str, default='deeplab_resnet.ckpt',
		help='pre-trained model filename corresponding to encoder_name')
	parser.add_argument('--data_list', dest='data_list', type=str, default='./dataAugment/train.txt',
		help='training data list filename')
	parser.add_argument('--batch_size', dest='batch_size', type=int, default=15,
		help='training batch size')
	parser.add_argument('--input_height', dest='input_height', type=int, default=256,
		help='input image height')
	parser.add_argument('--input_width', dest='input_width', type=int, default=256,
		help='input image width')
	parser.add_argument('--num_classes', dest='num_classes', type=int, default=2,
		help='number of classes in images')
	parser.add_argument('--log_dir', dest='log_dir', type=str, default="log",
		help='directory for saving log files')
	parser.add_argument('--log_file', dest='log_file', type=str, default="log.txt",
		help='Default logfile name')
	parser.add_argument('--print_color', dest='print_color', type=str, default="\033[0;37;40m",
		help='color of printed text')



	args = parser.parse_args()

	# Choose which gpu or cpu to use
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
	tf.app.run()
