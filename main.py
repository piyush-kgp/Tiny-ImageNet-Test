
from torch.utils.data import DataLoader
from torchvision import models, transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.manifold import TSNE
import numpy as np
from PIL import Image
import pandas as pd
from itertools import permutations
import matplotlib.pyplot as plt
import argparse
import os, glob


class Identity(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x


class TinyImagnet_Val:
    def __init__(self, root_dir, transform, annotation_file, given_classes, samples_per_class):
        self.root_dir = root_dir
        self.annotations = pd.read_csv(annotation_file,sep='\t',header=None,
                            names=['filename','label','tl_x','tl_y','br_x','br_y'])
        self.annotations = self.annotations[self.annotations.label.isin(given_classes)]
        self.transform = transform
        self.given_classes = given_classes

    def __getitem__(self,index):
        filename, label, tl_x, tl_y, br_x, br_y =  self.annotations.iloc[index]
        fp = os.path.join(self.root_dir, "images", filename)
        return self.transform(Image.open(fp).convert('RGB')), label

    def __len__(self):
        return self.annotations.shape[0]


def test(val_dataloader, model_dict, writer, device):
    for name, model in model_dict.items():
        model.fc = Identity()
        model = model.to(device)
        model.eval()

        embeddings, labels = [], []
        for i, batch in enumerate(val_dataloader):
            imgs, lbls = batch
            imgs = imgs.float().to(device)
            emb = model(imgs)
            embeddings.append(emb.cpu().detach().numpy())
            labels.extend(lbls)

        embeddings = np.concatenate(embeddings)
        labels = np.asarray(labels)
        label_set = np.asarray(val_dataloader.dataset.given_classes)
        label_indices = np.asarray([np.where(label_set==l) for l in labels])[:,0,0]

        embeddings_tsne = TSNE(n_components=2).fit_transform(embeddings)

        plt.figure(figsize=(10,8))
        for i, label in enumerate(label_set):
            indices, = np.where(label_indices==i)
            xs, ys = embeddings_tsne[indices].T
            plt.scatter(xs, ys, label=label)
        plt.legend()
        plt.savefig("TIN_VAL_TSNE_plot_{}.png".format(name))
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Process args for Triplet Net Training')
    parser.add_argument("--val_dir", type=str, help="Val directory",
                        default="/home/delta_one/Metric_Learning/tiny-imagenet-200/val")
    parser.add_argument("--val_annotation_file", type=str, help="Val Annotation File",
                        default="/home/delta_one/Metric_Learning/tiny-imagenet-200/val/val_annotations.txt")
    parser.add_argument('--classes', type=str, nargs='+', default=\
                        ['n01443537', 'n01629819', 'n01641577', 'n01644900', 'n01698640', 'n01742172', 'n01768244', 'n01770393', 'n01774384', 'n01774750'],
                        help='Classes used for Train and Eval')
    parser.add_argument('--samples_per_class', type=int, default=50,
                        help='# Samples used for Test')
    parser.add_argument('--batch_size_test', type=int, default=30,
                        help='validation set input batch size')

    args = parser.parse_args()

    transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    val_dataset = TinyImagnet_Val(root_dir=args.val_dir, transform=transform_test,
                        annotation_file=args.val_annotation_file, given_classes=args.classes,
                        samples_per_class=args.samples_per_class)
    nw = 4 if torch.cuda.is_available() else 0
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size_test,
                                    num_workers=nw, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir="logs/")

    model_dict = {
        "AlexNet": models.alexnet(pretrained=True),
        "DenseNet": models.densenet121(pretrained=True),
        "MNASNet": models.mnasnet0_5(pretrained=True),
        "MobileNetV2": models.mobilenet_v2(pretrained=True),
        "ResNet18": models.resnet18(pretrained=True),
        "ResNet101": models.resnet101(pretrained=True),
        "ShuffleNet": models.shufflenet_v2_x0_5(pretrained=True),
        "SqueezeNet": models.squeezenet1_0(pretrained=True),
        "VGG19": models.vgg19(pretrained=True)
        }

    test(val_dataloader, model_dict, writer, device)


if __name__=="__main__":
    main()
