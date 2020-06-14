
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


class TinyImagnet_Val:
    def __init__(self, root_dir, given_classes, samples_per_class, transform):
        self.root_dir = root_dir
        self.annotations = pd.read_csv(annotation_file,sep='\t',header=None,
                            names=['filename','label','tl_x','tl_y','br_x','br_y'])
        self.annotations = self.annotations[self.annotations.label.isin(given_classes)]
        self.transform = transform

    def __getitem__(self,index):
        filename, label, tl_x, tl_y, br_x, br_y =  self.annotations.iloc[index]
        fp = os.path.join(self.root_dir, "images", filename)
        return self.transform(Image.open(fp).convert('RGB')), label

    def __len__(self):
        return self.annotations.shape[0]


class Retrieval_Dataset:
    def __init__(self, root_dir, given_classes, db_samples_per_class, transform, db):
        self.root_dir = root_dir

        if db=='SEARCH'
            classes_list, file_paths_list = [], []
            for cls in given_classes:
                classes_list.extend([cls]*db_samples_per_class)
                all_imgs = sorted(glob.glob(os.path.join(root_dir, cls, "images", "*.JPEG")))
                file_paths_list.extend(all_imgs[:db_samples_per_class])

        elif db=='QUERY':
            self.annotations = pd.read_csv(annotation_file,sep='\t',header=None,
                    names=['filename','label','tl_x','tl_y','br_x','br_y'])

            classes_list, file_paths_list = [], []
            for cls in given_classes:
                classes_list.extend([cls]*db_samples_per_class)
                all_imgs = sorted(glob.glob(os.path.join(root_dir, cls, "images", "*.JPEG")))
                file_paths_list.extend(all_imgs[:db_samples_per_class])

        self.annotations = pd.DataFrame({'file_path': file_paths_list, 'label': classes_list})
        self.transform = transform

    def __getitem__(self,index):
        fp, label=  self.annotations.iloc[index]
        return self.transform(Image.open(fp).convert('RGB')), label

    def __len__(self):
        return self.annotations.shape[0]


def create_database(root_dir, given_classes, db_samples_per_class, transform, test_batch_size):
    retrieval_ds = Retrieval_Dataset(root_dir, given_classes, db_samples_per_class, transform)
    nw = 4 if torch.cuda.is_available() else 0
    retrieval_dl = DataLoader(retrieval_ds, batch_size=test_batch_size, num_workers=nw, shuffle=False)
    embeddings = []
    for i, batch in enumerate(val_dataloader):
        imgs, lbls = batch
        imgs = imgs.float().to(device)
        emb = model(imgs)
        embeddings.append(emb.cpu().detach().numpy())

    embeddings = np.concatenate(embeddings)
    labels = np.asarray(labels)

    dictionary = {'batch': batch_num, 'db': embeddings, 'labels': labels}
    np.save("SearchDB/db_{}.npy".format(train_batch_num), dictionary)


def calculate_P_at_K(res, K):
    """
    Precision @ K assumes that there are at least K ground truth positives.
    Because otherwise it will never be equal to 1.
    """
    if not res.sum() > K:
        return ValueError("To calculate P@K, the requirement is that GTP>K")

    res = res[:K]
    weights = np.arange(K, 0, -1))
    P_at_K_weighted = weights*res./weights.sum()
    P_at_K_uniform = res.mean()
    P_at_K_at_least_one = 1. if res.sum()>0 else 0.
    return {
            "P_at_K_weighted": P_at_K_weighted,
            "P_at_K_uniform": P_at_K_uniform,
            "P_at_K_at_least_one": P_at_K_at_least_one,
           }

def calculate_AP(res):
    """
    AP means average precision at varying recall values (recall varies from 0 to 1)
    """
    gtp = np.sum(res)
    corrects = np.cumsum(res)*res
    totals = np.arange(1,len(res)+1)
    precisions = corrects/totals
    return  1./gtp*np.sum(precisions)

    # Method 2:
    # positions = np.sorted(np.where(res==1)+1)
    # AP = 1./gtp*np.sum((np.arange(gtp)+1)/positions)


def test_retrieval_metrics():
    res = np.asarray([1,1,1,0,0,0,0,0,1,1,0,1,0,1])
    P_at_K = calculate_P_at_K(res, 5)
    print("P@K metrics: {}".format(P_at_K))
    AP = calculate_AP(res)
    print("AP = {}".format(AP))
    print("Hand-calculated numbers = P_at_K_weighted: {}, P_at_K_uniform: {}, P_at_K_at_least_one: {}, AP: {}".format(
        (5+4+3)./(5+4+3+2+1),
        3/5,
        1,
        1./7*(1/1 + 2/2 + 3/3 + 4/9 + 5/10 + 6/12 + 7/14)
    ))


def calculate_retrieval_statistics(train_batch_num, val_labels, val_embeddings, classes_list, num_queries_per_class=10, K=5):
    dictionary = np.load("SearchDB/db_{}.npy".format(train_batch_num))
    db_embeddings = dictionary["db"]
    db_labels = dictionary["labels"]

    AP_array, P_at_K_uniform_array, P_at_K_weighted_array, P_at_K_at_least_one_array = [], [], [], []
    for cls in classes_list:
        print("Calculating statistics for class {}".format(cls), flush=True)
        indices = np.where(val_labels==cls)
        for i in range(num_queries_per_class):
            query_embedding = val_embeddings[indices[i]]
            distances = [np.linalg.norm(query_embedding, db_embedding) for db_embedding in db_embeddings]
            distances_order = distances.argsort()
            db_labels_retrieved = db_labels[distances_order]
            res = (db_labels_retrieved==cls).astype(int)
            P_at_K = calculate_P_at_K(res, K)
            AP = calculate_AP(res)
            print("Class {} Query {} P@K metrics\n{}\n".format(cls, i, P_at_K), flush=True)
            print("Class {} Query {} AP".format(cls, i, AP), flush=True)

            AP_array.append(AP)
            P_at_K_uniform_array.append(P_at_K_uniform)
            P_at_K_weighted_array.append(P_at_K_weighted)
            P_at_K_at_least_one_array.append(P_at_K_at_least_one)

    print("Overall Results at Batch {}: P@K Weighted {}, P@K Uniform {}, P@K ALO {}, MAP {}".format(
        np.mean(P_at_K_weighted_array),
        np.mean(P_at_K_uniform_array),
        np.mean(P_at_K_at_least_one_array),
        np.mean(AP_array)
    ), flush=True)


def main():
    parser = argparse.ArgumentParser(description='Process args for Triplet Net Training')
    parser.add_argument("--train_dir", type=str, help="Train directory",
                        default="/home/delta_one/Metric_Learning/tiny-imagenet-200/train")
    parser.add_argument("--val_dir", type=str, help="Val directory",
                        default="/home/delta_one/Metric_Learning/tiny-imagenet-200/val")
    parser.add_argument("--val_annotation_file", type=str, help="Val Annotation File",
                        default="/home/delta_one/Metric_Learning/tiny-imagenet-200/val/val_annotations.txt")
    parser.add_argument('--classes', type=str, nargs='+', default=\
                        ['n01443537', 'n01629819', 'n01641577', 'n01644900', 'n01698640', 'n01742172', 'n01768244', 'n01770393', 'n01774384', 'n01774750'],
                        help='Classes used for Train and Eval')
    parser.add_argument('--db_samples_per_class', type=int, default=100,
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
    model = models.resnet101(pretrained=True)
    model.fc = Identity()
    model = model.to(device)

    model.eval()
    create_database(args.train_dir, args.classes, args.db_samples_per_class, transform_test, args.batch_size_test)
    calculate_retrieval_statistics(batch_num, val_labels, val_embeddings, args.classes, args.num_queries_per_class, args.K_for_P_at_K)

if __name__=="__main__":
    main()
