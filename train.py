import argparse
import utility as ut
from custom_model import FlowerClassifier

choice_list = list(ut.supported_archs.keys())
parser = argparse.ArgumentParser(description='Train classifier')

#Add arguements
parser.add_argument('data_directory', help='''Root path to load images from. 
                    It should contain directory for 
                    training data named "train",
                    validation data named "validation"
                    and testing data named "test"
                    ''')
parser.add_argument('--save_dir', help='Location to save checkpoint (default: current directory)')
parser.add_argument('--arch', default=choice_list[0], choices=choice_list, help='Architecture to preload for training (default: densenet121)')
parser.add_argument('--learning_rate', default=0.002, type=float)
parser.add_argument('--hidden_units', default=500, type=int)
parser.add_argument('--epochs', default=1, type=int)
parser.add_argument('--gpu', help='Pass to train model on GPU (default: CPU)', action='store_true')

args = parser.parse_args()

data_dir = args.data_directory
epochs = args.epochs
learning_rate = args.learning_rate
hidden_units = args.hidden_units
save_dir = args.save_dir
gpu = args.gpu
arch = args.arch

classifier_data = ut.load_data(data_dir)  

classifier = FlowerClassifier(supported_archs = ut.supported_archs)
classifier.train(classifier_data = classifier_data, epochs= epochs, lr= learning_rate, hidden_unit= hidden_units, save_dir= save_dir, use_gpu= gpu, arch= arch)