import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from PIL import Image
import os


class dataset_manger(data.Dataset):
    def __init__(self, images_root, data_file, transforms=None, num_output_bins=8):
        self.images_root = images_root
        self.labels = []
        self.images_file = []
        self.transforms = transforms
        self.num_output_bins = num_output_bins
        with open(data_file) as fin:
            for line in fin:
                image_file, image_label = line.split()
                self.labels.append(int(image_label))
                self.images_file.append(image_file)

    def __getitem__(self, index):
        img_file, target = self.images_file[index], self.labels[index]
        full_file = os.path.join(self.images_root, img_file)
        img = Image.open(full_file)

        if img.mode == 'L':
            img = img.convert('RGB')

        if self.transforms:
            img = self.transforms(img)

        multi_hot_target = torch.zeros(self.num_output_bins).long()
        multi_hot_target[list(range(target))] = 1

        return img, target, multi_hot_target

    def __len__(self):
        return len(self.labels)


if __name__ == '__main__':
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    scaler = transforms.Resize((224, 224))
    preprocess = transforms.Compose([scaler, transforms.ToTensor(), normalize])
    t = dataset_manger('/home/share_data/age/CVPR19/datasets/MORPH', './data_list/ET_proto_val.txt', preprocess)
    train_loader = data.DataLoader(t, batch_size=2, shuffle=False)
    for data_, index, tar in train_loader:
        print(data_)
        print(index)
        print(tar)
        break
