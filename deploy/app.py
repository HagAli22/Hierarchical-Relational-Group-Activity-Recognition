"""
Group Activity Recognition - FastAPI Web Service
=================================================
REST API for the best model (RCRG_2R_11C_conc_Temp_GAT) with 91.85% accuracy.

Run:
    uvicorn deploy.app:app --host 0.0.0.0 --port 8000
"""

import os
import sys
import io
import json
import tempfile
import shutil
import subprocess
from typing import Optional
from contextlib import asynccontextmanager

import torch
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import torchvision.transforms as T
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.person_classifer import Person_Classifer
from models.attention_model.RCRG_2R_11C_conc_Temp_GAT import RCRG_2R_11C_conc_Temp_GAT

ACTIVITY_CLASSES = [
    'l-pass', 'r-pass', 'l-spike', 'r_spike',
    'l_set', 'r_set', 'l_winpoint', 'r_winpoint'
]

TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

recognizer = None


class GroupActivityRecognizer:
    def __init__(self, model_path):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self._load_model(model_path)
        self.model.eval()
    
    def _load_model(self, model_path):
        person_model = Person_Classifer(num_classes=9)
        model = RCRG_2R_11C_conc_Temp_GAT(person_model, num_classes=8)
        checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model_keys = set(model.state_dict().keys())
        filtered = {k: v for k, v in state_dict.items() if k in model_keys}
        model.load_state_dict(filtered, strict=False)
        return model.to(self.device)
    
    def preprocess_clip(self, frames, bboxes_per_frame):
        max_players = 12
        clip_data = []
        
        for frame, bboxes in zip(frames, bboxes_per_frame):
            frame_players = []
            orders = []
            
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox['box'] if isinstance(bbox, dict) else bbox
                orders.append((x1 + x2) // 2)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                crop = frame[y1:y2, x1:x2]
                
                if crop.size == 0:
                    crop = np.zeros((224, 224, 3), dtype=np.uint8)
                
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                frame_players.append(TRANSFORM(Image.fromarray(crop_rgb)))
            
            if frame_players:
                pairs = sorted(zip(orders, frame_players), key=lambda x: x[0])
                frame_players = [img for _, img in pairs]
            
            while len(frame_players) < max_players:
                frame_players.append(torch.zeros(3, 224, 224))
            
            clip_data.append(torch.stack(frame_players[:max_players]))
        
        return torch.stack(clip_data).permute(1, 0, 2, 3, 4).contiguous().unsqueeze(0)
    
    @torch.no_grad()
    def predict(self, clip_tensor):
        clip_tensor = clip_tensor.to(self.device)
        outputs = self.model(clip_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, dim=1)
        return {
            'class_name': ACTIVITY_CLASSES[pred.item()],
            'confidence': conf.item(),
            'probabilities': {n: p.item() for n, p in zip(ACTIVITY_CLASSES, probs[0])}
        }


def extract_all_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def add_prediction_overlay(frame, prediction):
    result = frame.copy()
    pred_class = prediction['class_name']
    h, w = frame.shape[:2]
    text = pred_class.replace('_', '-')
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    thickness = 3
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = (w - text_w) // 2
    y = 50 + text_h
    cv2.rectangle(result, (x - 10, y - text_h - 10), (x + text_w + 10, y + 10), (0, 0, 0), -1)
    cv2.putText(result, text, (x, y), font, font_scale, (0, 255, 0), thickness)
    return result


def create_output_video(frames, annot, predictions, output_path, fps=10):
    h, w = frames[0].shape[:2]
    temp_path = output_path.replace('.mp4', '_temp.avi')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(temp_path, fourcc, fps, (w, h))
    
    for clip_info, prediction in zip(annot['clips'], predictions):
        for frame_info in clip_info['frames']:
            frame_idx = frame_info['frame_idx']
            frame = frames[frame_idx].copy()
            for bbox in frame_info['bboxes']:
                x1, y1, x2, y2 = bbox['box']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            frame = add_prediction_overlay(frame, prediction)
            out.write(frame)
    
    out.release()
    
    try:
        subprocess.run(
            ['ffmpeg', '-y', '-i', temp_path, '-c:v', 'libx264', 
             '-pix_fmt', 'yuv420p', '-preset', 'fast', '-crf', '22', output_path],
            capture_output=True, check=True
        )
        os.unlink(temp_path)
    except:
        shutil.move(temp_path, output_path)
    
    return output_path


@asynccontextmanager
async def lifespan(app: FastAPI):
    global recognizer
    model_path = 'saved_best_model/RCRG_2R_11C_conc_Temp_GAT_best.pth'
    print(f"Loading model from: {model_path}")
    recognizer = GroupActivityRecognizer(model_path=model_path)
    print("Ready!")
    yield
    recognizer = None


app = FastAPI(
    title="Group Activity Recognition API",
    description="Upload volleyball video to recognize group activity",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the main page."""
    html_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": recognizer is not None,
        "device": recognizer.device if recognizer else "not loaded"
    }


@app.get("/classes")
async def get_classes():
    return {"classes": ACTIVITY_CLASSES}


@app.post("/predict")
async def predict_video(
    video: UploadFile = File(...)
):
    """Upload video only - JSON loaded automatically from deploy/videos/video9.json."""
    if recognizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Load JSON automatically
        json_path = os.path.join(os.path.dirname(__file__), "videos", "video9.json")
        with open(json_path, "r") as f:
            annot = json.load(f)
        
        # Save video
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            content = await video.read()
            tmp.write(content)
            video_path = tmp.name
        
        # Extract frames
        frames = extract_all_frames(video_path)
        os.unlink(video_path)
        
        is_multi_clip = 'clips' in annot
        
        if is_multi_clip:
            # Multi-clip format
            predictions = []
            for clip_info in annot['clips']:
                clip_frames = []
                clip_bboxes = []
                for frame_info in clip_info['frames']:
                    clip_frames.append(frames[frame_info['frame_idx']])
                    clip_bboxes.append(frame_info['bboxes'])
                
                clip_tensor = recognizer.preprocess_clip(clip_frames, clip_bboxes)
                result = recognizer.predict(clip_tensor)
                predictions.append(result)
            
            # Create output video
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_out:
                output_path = tmp_out.name
            
            create_output_video(frames, annot, predictions, output_path)
            
            return FileResponse(
                output_path,
                media_type="video/mp4",
                filename="prediction_output.mp4"
            )
        
        else:
            # Single clip format
            bboxes = [f['bboxes'] for f in annot['frames']]
            clip_tensor = recognizer.preprocess_clip(frames, bboxes)
            result = recognizer.predict(clip_tensor)
            
            return {
                "class_name": result['class_name'],
                "confidence": result['confidence'],
                "probabilities": result['probabilities']
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
