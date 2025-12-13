import os
import sys
import pickle
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random
import h5py
sys.path.append('D:\VSCODE\GNN')

from data.boxinfo import BoxInfo


def load_tracking_annot(path):
    with open(path, 'r') as file:
        player_boxes = {idx: [] for idx in range(12)}
        frame_boxes_dct = {}

        for idx, line in enumerate(file):
            box_info = BoxInfo(line)

            if box_info.player_ID > 11:
                continue
            player_boxes[box_info.player_ID].append(box_info)

        for player_ID, boxes_info in player_boxes.items():
            boxes_info = boxes_info[5:]
            boxes_info = boxes_info[:-6]

            for box_info in boxes_info:
                if box_info.frame_ID not in frame_boxes_dct:
                    frame_boxes_dct[box_info.frame_ID] = []
                frame_boxes_dct[box_info.frame_ID].append(box_info)

        return frame_boxes_dct   # 9 frames per clip   , each frame has 12 players


def load_video_annot(video_annot):
    with open(video_annot, 'r') as file:
        clip_category_dct = {}

        for line in file:
            items = line.strip().split(' ')[:2]
            clip_dir = items[0].replace('.jpg', '')
            clip_category_dct[clip_dir] = items[1]

        return clip_category_dct


def load_volleyball_dataset(videos_root, annot_root):
    videos_dirs = os.listdir(videos_root)
    videos_dirs.sort()

    videos_annot = {}

    for idx, video_dir in enumerate(videos_dirs):
        video_dir_path = os.path.join(videos_root, video_dir)

        if not os.path.isdir(video_dir_path):
            continue

        print(f'{idx}/{len(videos_dirs)} - Processing Dir {video_dir_path}')

        video_annot = os.path.join(video_dir_path, 'annotations.txt')
        clip_category_dct = load_video_annot(video_annot)

        clips_dir = os.listdir(video_dir_path)
        clips_dir.sort()

        clip_annot = {}

        for clip_dir in clips_dir:
            clip_dir_path = os.path.join(video_dir_path, clip_dir)

            if not os.path.isdir(clip_dir_path):
                continue

            assert clip_dir in clip_category_dct

            annot_file = os.path.join(annot_root, video_dir, clip_dir, f'{clip_dir}.txt')
            frame_boxes_dct = load_tracking_annot(annot_file)

            clip_annot[clip_dir] = {
                'category': clip_category_dct[clip_dir],
                'frame_boxes_dct': frame_boxes_dct
            }

        videos_annot[video_dir] = clip_annot

    return videos_annot


def create_data_loaders(dataset_root,
                        mode='B1',
                        batch_size=32,
                        num_workers=0,
                        sequence_length=9,
                        middle_frame_only=False,
                        team_split=False):
    """
    Create train, val, and test data loaders for specified baseline
    """

    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = VolleyballDataset(
        dataset_root=dataset_root,
        mode=mode,
        split='train',
        transform=train_transform,
        sequence_length=sequence_length,
        middle_frame_only=middle_frame_only,
        team_split=team_split
    )

    val_dataset = VolleyballDataset(
        dataset_root=dataset_root,
        mode=mode,
        split='val',
        transform=val_transform,
        sequence_length=sequence_length,
        middle_frame_only=middle_frame_only,
        team_split=team_split
    )

    test_dataset = VolleyballDataset(
        dataset_root=dataset_root,
        mode=mode,
        split='test',
        transform=val_transform,
        sequence_length=sequence_length,
        middle_frame_only=middle_frame_only,
        team_split=team_split
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader



class Person_classifier_loaders(Dataset):
    def __init__(self, videos_path, annot_path, split, transform=None, logger=None):
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

        self.data = []
        
        
        try:
            with open(annot_path, 'rb') as file:
                videos_annot = pickle.load(file)
        except Exception as e:
            raise

        box_count = 0
        for idx in split:
            video_annot = videos_annot[str(idx)]

            for clip in video_annot.keys():
                dir_frames = list(video_annot[str(clip)]['frame_boxes_dct'].items())

                for frame_id, boxes in dir_frames:
                    #print(type(frame_id), type(clip))
                    # framestr= str(frame_id)
                    # if framestr == clip:
                    #     print("s",frame_id, clip)
                    image_path = f'{videos_path}/{str(idx)}/{str(clip)}/{frame_id}.jpg'
                    image_path = os.path.join(image_path)

                    for box_info in boxes:
                        x1, y1, x2, y2 = box_info.box
                        category = box_info.category
                        box_count += 1

                        self.data.append(
                            {
                                'frame_path': f'{videos_path}/{str(idx)}/{str(clip)}/{frame_id}.jpg',
                                'box': (x1, y1, x2, y2),
                                'category': category
                            }
                        )
        
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        num_classes=len(self.categories_dct)
        category=sample['category']
        label_idx=torch.zeros(num_classes)
        label_idx[self.categories_dct[category]] = 1
        label_idx = self.categories_dct[sample['category']]
        # num_classes = len(self.categories_dct)
        # labels = torch.zeros(num_classes)
        #label_idx = self.categories_dct[sample['category']]

        # labels[self.categories_dct[sample['category']]] = 1

        image = Image.open(sample['frame_path']).convert('RGB')
        x1, y1, x2, y2 = sample['box']
        cropped_image = image.crop((x1, y1, x2, y2))
    
        if self.transform:
            cropped_image = self.transform(image=np.array(cropped_image))['image']

        return cropped_image, label_idx



class GroupActivityDataset(Dataset):
    def __init__(self, videos_path,annot_path, split,feature_path, sort = True,sec=False, transform=None):
        self.samples = []
        self.transform = transform
        self.sort = sort # If True, prepares data for the 2-group model
        self.sec = sec # If True, use sec of all frames
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

        self.data=[]
        self.clip={}
        
        with open(annot_path,'rb')as file:
            videos_annot=pickle.load(file)
        
        f = h5py.File(feature_path, 'r')

        self.features=f["fc7_features"]
        self.labels=f["labels"]
        self.meta=f["meta"][:]

        # Build lookup dict
        self.index_map = {}
        for i, m in enumerate(self.meta):
            vid, cid, fid, bid = m
            self.index_map[(vid, cid, fid, bid)] = i


        for idx in split:
            video_annot=videos_annot[str(idx)]

            for clip in video_annot.keys():

                category = video_annot[str(clip)]['category']
                dir_frames = list(video_annot[str(clip)]['frame_boxes_dct'].items()) # 9 frames per clip

                
                clip_data = []
                if self.sec == False: # all frames as separately for non-temporal model
                        frame_boxes=[]
                        for frame_id,boxes in dir_frames:
                            clip_data = []
                            frame_path=f'{videos_path}/{str(idx)}/{str(clip)}/{frame_id}.jpg'
                            frame_boxes=[]
    
                            for box_info in boxes: 
                                frame_boxes.append(box_info)

                            clip_data.append((idx,int(clip),int(clip),frame_boxes,frame_path))
                            self.data.append(
                                {
                                    'clip_data':clip_data,
                                    'category':category,
                                }
                            )

                
                else: # all frames as sequence for temporal model
                    for frame_id,boxes in dir_frames:
                        frame_path=f'{videos_path}/{str(idx)}/{str(clip)}/{frame_id}.jpg'
                        frame_boxes=[]
                        for box_info in boxes: 
                            frame_boxes.append(box_info)
                        
                        clip_data.append((idx,int(clip),frame_id,frame_boxes,frame_path))
                        
                    self.data.append(
                        {
                            'clip_data':clip_data,
                            'category':category,
                        }
                    )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample=self.data[idx]
        label_idx = self.categories_dct[sample['category']]

        frame_data=sample['frame_data']
        clip=[]

        for idx,clip,frame_id,frame_boxes in frame_data:
            seq_player=[]
            orders=[]
            
            for box_info in frame_boxes:

                x1, y1, x2, y2 = box_info.box

                x_center = (x1 + x2) // 2
                orders.append(x_center)

                box_id = box_info.box_ID

                index_in_meta = (idx, clip, frame_id, box_id)
                feature_idx = self.index_map[index_in_meta]
                box_feature = torch.tensor(self.features[feature_idx])
                seq_player.append(box_feature)

            if self.sort:
                # Sort seq_player based on orders
                orders_with_images = list(zip(orders, seq_player))
                orders_with_images.sort(key=lambda x: x[0])  # Sort by x_center
                seq_player = [img for _, img in orders_with_images]

            seq_player = torch.stack(seq_player)
            clip.append(seq_player)

        clip = torch.stack(clip).permute(1, 0, 2) # (num_people, num_frames, feature_dim)

        return clip, label_idx


