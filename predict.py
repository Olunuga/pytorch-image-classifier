import argparse
import utility as ut
from custom_model import FlowerClassifier

parser = argparse.ArgumentParser(description='Predict image')

parser.add_argument('image_path', help='Path to image to predict')
parser.add_argument('checkpoint', help='Path to checkpoint')
parser.add_argument('--top_k', type=int, default=3, help='Number of top probability to return (default: 3)')
parser.add_argument('--category_names', help='Path to category map to real names json file')
parser.add_argument('--gpu', help='Pass to infer on GPU (default: CPU)', action='store_true')

args = parser.parse_args()

image_path = args.image_path
checkpoint_path = args.checkpoint
top_k = args.top_k
category_names_file_path = args.category_names
gpu = args.gpu

category_names = (ut.load_json_file(category_names_file_path) if category_names_file_path is not None else None )
image = ut.process_image(image_path)

classifier = FlowerClassifier(supported_archs = ut.supported_archs)
classifier.predict(image, checkpoint_path, gpu, topk = top_k, category_names = category_names)

