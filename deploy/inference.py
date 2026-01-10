"""
Group Activity Recognition - Inference Script
==============================================
Deploy the best model (RCRG_2R_11C_conc_Temp_GAT) with 91.85% accuracy.

Usage:
    # Predict on a specific clip
    python deploy/inference.py --video_id 7 --clip_id 38025
    
    # Predict on video folder with annotations
    python deploy/inference.py --video_folder data/videos_sample/7/38025
"""

import os
import sys
import argparse
import pickle
import torch
import cv2
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.person_classifer import Person_Classifer
from models.attention_model.RCRG_2R_11C_conc_Temp_GAT import RCRG_2R_11C_conc_Temp_GAT
from data.boxinfo import BoxInfo


# Activity classes
ACTIVITY_CLASSES = [
    'l-pass',      # Left pass
    'r-pass',      # Right pass  
    'l-spike',     # Left spike
    'r_spike',     # Right spike
    'l_set',       # Left set
    'r_set',       # Right set
    'l_winpoint',  # Left win point
    'r_winpoint'   # Right win point
]

# Image preprocessing (same as training)
TRANSFORM = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])


class CustomUnpickler(pickle.Unpickler):
    """Custom unpickler that handles BoxInfo class from different module paths."""
    def find_class(self, module, name):
        if name == 'BoxInfo':
            return BoxInfo
        return super().find_class(module, name)


def load_pickle_with_boxinfo(path):
    """Load pickle file with BoxInfo class compatibility."""
    with open(path, 'rb') as f:
        return CustomUnpickler(f).load()


class GroupActivityRecognizer:
    """
    Group Activity Recognition inference class.
    Uses RCRG_2R_11C_conc_Temp_GAT model (91.85% accuracy).
    """
    
    def __init__(self, model_path, person_classifier_path=None, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.model = self._load_model(model_path, person_classifier_path)
        self.model.eval()
        
    def _load_model(self, model_path, person_classifier_path):
        """Load the trained model."""
        person_model = Person_Classifer(num_classes=9)
        
        if person_classifier_path and os.path.exists(person_classifier_path):
            person_checkpoint = torch.load(person_classifier_path, map_location=self.device)
            if 'model_state_dict' in person_checkpoint:
                person_model.load_state_dict(person_checkpoint['model_state_dict'])
            else:
                person_model.load_state_dict(person_checkpoint)
            print(f"Loaded person classifier from: {person_classifier_path}")
        
        model = RCRG_2R_11C_conc_Temp_GAT(person_model, num_classes=8)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        print(f"Loaded model from: {model_path}")
        model = model.to(self.device)
        return model
    
    def preprocess_clip(self, frames, bboxes_per_frame, sort_by_x=True):
        """
        Preprocess a video clip for inference.
        
        Args:
            frames: List of 9 frames (numpy arrays BGR)
            bboxes_per_frame: List of 9 lists of BoxInfo objects or tuples
            sort_by_x: Sort players by x-center (left to right)
        
        Returns:
            Tensor of shape (1, 12, 9, 3, 224, 224)
        """
        max_players = 12
        clip_data = []
        
        for frame, bboxes in zip(frames, bboxes_per_frame):
            frame_players = []
            orders = []
            
            for box_info in bboxes:
                if hasattr(box_info, 'box'):
                    x1, y1, x2, y2 = box_info.box
                else:
                    x1, y1, x2, y2 = box_info
                
                x_center = (x1 + x2) // 2
                orders.append(x_center)
                
                # Crop person
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                person_crop = frame[y1:y2, x1:x2]
                
                if person_crop.size == 0:
                    person_crop = np.zeros((224, 224, 3), dtype=np.uint8)
                
                transformed = TRANSFORM(image=person_crop)
                frame_players.append(transformed['image'])
            
            # Sort by x-center if needed
            if sort_by_x and len(frame_players) > 0:
                orders_with_images = list(zip(orders, frame_players))
                orders_with_images.sort(key=lambda x: x[0])
                frame_players = [img for _, img in orders_with_images]
            
            # Pad to 12 players
            while len(frame_players) < max_players:
                frame_players.append(torch.zeros(3, 224, 224))
            
            frame_players = frame_players[:max_players]
            clip_data.append(torch.stack(frame_players))
        
        # Stack: (9, 12, 3, 224, 224) -> (12, 9, 3, 224, 224)
        clip_tensor = torch.stack(clip_data).permute(1, 0, 2, 3, 4)
        return clip_tensor.unsqueeze(0)  # (1, 12, 9, 3, 224, 224)
    
    @torch.no_grad()
    def predict(self, clip_tensor):
        """Run inference on preprocessed clip."""
        clip_tensor = clip_tensor.to(self.device)
        outputs = self.model(clip_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, dim=1)
        
        return {
            'class': predicted_class.item(),
            'class_name': ACTIVITY_CLASSES[predicted_class.item()],
            'confidence': confidence.item(),
            'probabilities': {
                name: prob.item() 
                for name, prob in zip(ACTIVITY_CLASSES, probabilities[0])
            }
        }
    
    def predict_from_clip_data(self, videos_path, video_id, clip_id, annot_data):
        """
        Predict from preprocessed annotation data.
        
        Args:
            videos_path: Root path to videos
            video_id: Video ID (e.g., '7')
            clip_id: Clip ID (e.g., '38025')
            annot_data: Loaded annotation dict from annot_all2.pkl
        """
        clip_data = annot_data[str(video_id)][str(clip_id)]
        frame_boxes_dct = clip_data['frame_boxes_dct']
        ground_truth = clip_data['category']
        
        frames = []
        bboxes_per_frame = []
        
        # Load 9 frames
        for frame_id in sorted(frame_boxes_dct.keys()):
            frame_path = os.path.join(videos_path, str(video_id), str(clip_id), f'{frame_id}.jpg')
            frame = cv2.imread(frame_path)
            
            if frame is None:
                raise FileNotFoundError(f"Frame not found: {frame_path}")
            
            frames.append(frame)
            bboxes_per_frame.append(frame_boxes_dct[frame_id])
        
        clip_tensor = self.preprocess_clip(frames, bboxes_per_frame)
        result = self.predict(clip_tensor)
        result['ground_truth'] = ground_truth
        result['correct'] = result['class_name'] == ground_truth
        
        return result


def visualize_prediction(videos_path, video_id, clip_id, annot_data, result, output_path=None):
    """Visualize prediction on middle frame."""
    clip_data = annot_data[str(video_id)][str(clip_id)]
    frame_boxes_dct = clip_data['frame_boxes_dct']
    
    # Get middle frame
    frame_ids = sorted(frame_boxes_dct.keys())
    middle_frame_id = frame_ids[len(frame_ids) // 2]
    
    frame_path = os.path.join(videos_path, str(video_id), str(clip_id), f'{middle_frame_id}.jpg')
    frame = cv2.imread(frame_path)
    
    # Draw bboxes
    for box_info in frame_boxes_dct[middle_frame_id]:
        x1, y1, x2, y2 = box_info.box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, box_info.category, (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Add prediction text
    pred_text = f"Predicted: {result['class_name']} ({result['confidence']*100:.1f}%)"
    gt_text = f"Ground Truth: {result['ground_truth']}"
    status = "CORRECT" if result['correct'] else "WRONG"
    color = (0, 255, 0) if result['correct'] else (0, 0, 255)
    
    cv2.putText(frame, pred_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, gt_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, status, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    if output_path:
        cv2.imwrite(output_path, frame)
        print(f"Saved visualization to: {output_path}")
    
    return frame


def main():
    parser = argparse.ArgumentParser(description='Group Activity Recognition Inference')
    parser.add_argument('--model_path', type=str, 
                        default='saved_best_model/RCRG_2R_11C_conc_Temp_GAT_best.pth',
                        help='Path to trained model')
    parser.add_argument('--person_classifier', type=str, default=None,
                        help='Path to person classifier (optional)')
    parser.add_argument('--annot_path', type=str, default='data/annot_all2.pkl',
                        help='Path to preprocessed annotations')
    parser.add_argument('--videos_path', type=str, default='data/videos_sample',
                        help='Path to videos folder')
    parser.add_argument('--video_id', type=str, required=True,
                        help='Video ID (e.g., 7)')
    parser.add_argument('--clip_id', type=str, required=True,
                        help='Clip ID (e.g., 38025)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for visualization')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Load annotations
    print(f"Loading annotations from: {args.annot_path}")
    annot_data = load_pickle_with_boxinfo(args.annot_path)
    
    # Check if video/clip exists
    if str(args.video_id) not in annot_data:
        print(f"Error: Video {args.video_id} not found. Available: {list(annot_data.keys())}")
        return
    
    if str(args.clip_id) not in annot_data[str(args.video_id)]:
        print(f"Error: Clip {args.clip_id} not found. Available: {list(annot_data[str(args.video_id)].keys())}")
        return
    
    # Initialize recognizer
    recognizer = GroupActivityRecognizer(
        model_path=args.model_path,
        person_classifier_path=args.person_classifier,
        device=args.device
    )
    
    # Run prediction
    result = recognizer.predict_from_clip_data(
        args.videos_path, args.video_id, args.clip_id, annot_data
    )
    
    # Print results
    print("\n" + "="*50)
    print("GROUP ACTIVITY RECOGNITION RESULT")
    print("="*50)
    print(f"Video: {args.video_id}, Clip: {args.clip_id}")
    print(f"Predicted: {result['class_name']}")
    print(f"Ground Truth: {result['ground_truth']}")
    print(f"Confidence: {result['confidence']*100:.2f}%")
    print(f"Status: {'✓ CORRECT' if result['correct'] else '✗ WRONG'}")
    print("\nAll Probabilities:")
    for name, prob in sorted(result['probabilities'].items(), key=lambda x: -x[1]):
        marker = " ←" if name == result['class_name'] else ""
        print(f"  {name}: {prob*100:.2f}%{marker}")
    
    # Visualize
    if args.output:
        visualize_prediction(args.videos_path, args.video_id, args.clip_id, 
                           annot_data, result, args.output)


if __name__ == '__main__':
    main()
