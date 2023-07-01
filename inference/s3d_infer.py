import os
import numpy as np
import cv2
import torch as th
from models.s3dg.s3dg import S3DG
import argparse

from config import reader
from ._utils import check_file_exists, check_dir_exists, clip_to_frames

cfg = reader()
num_classes = cfg["s3d"]["num_classes"]
file_weight = cfg["s3d"]["model_path"]
class_names_file = cfg["s3d"]["classes_names"]

def transform(snippet):
    """ stack & noralization """
    snippet = np.concatenate(snippet, axis=-1)
    snippet = th.from_numpy(snippet).permute(2, 0, 1).contiguous().float()
    snippet = snippet.mul_(2.).sub_(255).div(255)
    return snippet.view(1,-1,3,snippet.size(1),snippet.size(2)).permute(0,2,1,3,4)

def one_clip(clip_path):
    clip_filename = os.path.basename(clip_path)
    sample_dir = os.path.join(os.path.dirname(clip_path), f"{clip_filename.split('.')[0]}-frames")
    if check_dir_exists(sample_dir):
        print ('output dir exists? removing it ...')
        os.system('rm -rf ' + sample_dir)
    os.mkdir(sample_dir)
    # extract frames from the sample clip
    clip_to_frames(clip_path, sample_dir)
    return sample_dir


def prepare_model():
    model = S3DG(num_classes)
    # load the weight file and copy the parameters
    if check_file_exists(file_weight):
        print ('loading weight file')
        weight_dict = th.load(file_weight)
        model_dict = model.state_dict()
        for name, param in weight_dict.items():
            if 'module' in name:
                name = '.'.join(name.split('.')[1:])
            if name in model_dict:
                if param.size() == model_dict[name].size():
                    model_dict[name].copy_(param)
                else:
                    print (' size? ' + name, param.size(), model_dict[name].size())
            else:
                print (' name? ' + name)

        print (' loaded')
    else:
        print ('weight file?')
    return model


def infer(sample_dir, model, class_names):
    """ Output the top 5 Kinetics classes predicted by the model """
    model = model.cuda()
    th.backends.cudnn.benchmark = False
    model.eval()
    list_frames = [f for f in os.listdir(sample_dir) if os.path.isfile(os.path.join(sample_dir, f))]
    list_frames.sort()
    # read all the frames of sample clip
    snippet = []
    for frame in list_frames:
        img = cv2.imread(os.path.join(sample_dir, frame))
        # img = cv2.resize(img, (224, 224))
        img = img[...,::-1]
        snippet.append(img)

    clip_path = transform(snippet)
    with th.no_grad():
        logits = model(clip_path.cuda()).cpu().data[0]

    preds = th.softmax(logits, 0).numpy()
    sorted_indices = np.argsort(preds)[::-1][:5]
    print ('\nTop 5 classes ... with probability')
    for idx in sorted_indices:
        print (class_names[idx], '...', preds[idx])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip', type=str, help='path to the sample clip', required=True)
    args = parser.parse_args()

    if not check_file_exists(args.clip):
        print ('sample clip?')
        exit()

    class_names = [c.strip() for c in open(class_names_file)]
    print("extracting frames from the sample clip ...")
    sample_dir = one_clip(args.clip)
    model = prepare_model()
    infer(sample_dir, model, class_names)


