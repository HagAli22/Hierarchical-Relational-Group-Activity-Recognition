import os
from PIL import __version__ as PILLOW_VERSION
from PIL import Image
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import pickle
root_dataset = 'D:/volleyball-datasets'


class get_B1_loaders(Dataset):
    def __init__(self, videos_path,annot_path, split, transform=None):
        self.samples = []
        self.transform = transform
        self.categories_dct = {
            'l-pass': 0,
            'r-pass': 1,
            'l-spike': 2,
            'r_spike': 3,
            'l_set': 4,
            'r_set': 5,
            'l_winpoint': 6,
            'r_winpoint': 7
        }
        # print("Available videos in data:", list(dataset_dict.keys()))
        # print("Available labels in labels_dict:", list(label_dict.keys()))

        self.data=[]
        with open(annot_path,'rb')as file:
            videos_annot=pickle.load(file)


        for idx in split:
            video_annot=videos_annot[str(idx)]

            for clip in video_annot.keys():
                frames_data = video_annot[str(clip)]['frame_boxes_dct']
                category = video_annot[str(clip)]['category']
                dir_frames = list(video_annot[str(clip)]['frame_boxes_dct'].items())

                for frame_id,boxes in dir_frames:
                    #if str(clip)==str(frame_id):
                    self.data.append(
                        {
                            'frame_path':f'{videos_path}/{str(idx)}/{str(clip)}/{frame_id}.jpg',
                            'category':category
                        }
                    )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample=self.data[idx]
        num_classes=len(self.categories_dct)
        labels=torch.zeros(num_classes)
        labels[self.categories_dct[sample['category']]]=1

        frame = Image.open(sample['frame_path']).convert('RGB')

        if self.transform:
            frame = self.transform(frame)

        return frame, labels

class get_B3_loaders(Dataset):
    def __init__(self, videos_path,annot_path, split, transform=None):
        self.samples = []
        self.transform = transform
        self.categories_dct = {
            'waiting': 0,
            'setting': 1,
            'digging': 2,
            'falling': 3,
            'spiking': 4,
            'blocking': 5,
            'jumping': 6,
            'moving': 7,
            'standing': 8
        }
        # print("Available videos in data:", list(dataset_dict.keys()))
        # print("Available labels in labels_dict:", list(label_dict.keys()))

        self.data=[]
        with open(annot_path,'rb')as file:
            videos_annot=pickle.load(file)


        for idx in split:
            video_annot=videos_annot[str(idx)]

            for clip in video_annot.keys():
                frames_data = video_annot[str(clip)]['frame_boxes_dct']

                dir_frames = list(video_annot[str(clip)]['frame_boxes_dct'].items())

                for frame_id,boxes in dir_frames:

                    #if str(clip) == str(frame_id):
                        #print("###",frame_id, str(clip))'
                    image_path=f'{videos_path}/{str(idx)}/{str(clip)}/{frame_id}.jpg'
                    image_path=os.path.join(image_path)
                    image=Image.open(image_path).convert('RGB')


                    for box_info in boxes:
                        x1,y1,x2,y2=box_info.box
                        category = box_info.category
                        #print("category",category)
                        cropred_image=image.crop((x1,y1,x2,y2))
                        cropred_image=self.transform(cropred_image)

                        self.data.append(
                            {
                                'frame_path': f'{videos_path}/{str(idx)}/{str(clip)}/{frame_id}.jpg',
                                'cropred_image': cropred_image,
                                'category': category
                            }
                        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample=self.data[idx]
        num_classes=len(self.categories_dct)
        labels=torch.zeros(num_classes)
        category=self.categories_dct[sample['category']]

        labels[self.categories_dct[sample['category']]]=1

        cropred_image=sample['cropred_image']

        return cropred_image, labels
