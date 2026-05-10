"""
AI Junction Optimizer — FastAPI Backend
Extracted from Colab notebook for Railway deployment.
"""

import random
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ──────────────────────────────────────────────
# 1. YOLO DETECTION
# ──────────────────────────────────────────────
from ultralytics import YOLO
import base64

model = YOLO('yolov8n.pt')

VEHICLE_CLASSES  = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
VEHICLE_WEIGHTS  = {'car': 1.0, 'motorcycle': 0.5, 'bus': 2.5, 'truck': 2.0}

def decode_frame(b64_string: str) -> np.ndarray:
    img_bytes = base64.b64decode(b64_string)
    arr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def encode_frame(frame: np.ndarray) -> str:
    _, buf = cv2.imencode('.jpg', frame)
    return base64.b64encode(buf).decode('utf-8')

def detect_vehicles(frame: np.ndarray, lane_id: str = 'unknown') -> dict:
    results = model(
        frame,
        classes=list(VEHICLE_CLASSES.keys()),
        conf=0.4,
        verbose=False
    )[0]

    detections = []
    weighted_count = 0.0

    for box in results.boxes:
        cls_id   = int(box.cls[0])
        cls_name = VEHICLE_CLASSES.get(cls_id, 'vehicle')
        conf     = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        weight   = VEHICLE_WEIGHTS.get(cls_name, 1.0)
        weighted_count += weight

        detections.append({
            'type': cls_name,
            'confidence': round(conf, 2),
            'bbox': [x1, y1, x2, y2],
            'weight': weight
        })

        color = {
            'car': (0, 200, 100), 'motorcycle': (255, 165, 0),
            'bus': (0, 100, 255), 'truck': (180, 0, 255)
        }.get(cls_name, (200, 200, 200))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f'{cls_name} {conf:.0%}',
                    (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    cv2.rectangle(frame, (0, 0), (260, 36), (0, 0, 0), -1)
    cv2.putText(frame,
                f'Lane {lane_id}  |  vehicles: {len(detections)}  density: {weighted_count:.1f}',
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return {
        'lane_id': lane_id,
        'raw_count': len(detections),
        'weighted_density': round(weighted_count, 2),
        'detections': detections,
        'annotated_frame': encode_frame(frame)
    }


# ──────────────────────────────────────────────
# 2. SYNTHETIC TRAFFIC SIMULATOR
# ──────────────────────────────────────────────
LANE_IDS = ['North', 'South', 'East', 'West']

TRAFFIC_PROFILES = {
    'North': (8, 18),
    'South': (6, 14),
    'East':  (2, 7),
    'West':  (1, 5),
}

def make_synthetic_frame(vehicle_count: int, frame_h=480, frame_w=640) -> np.ndarray:
    frame = np.full((frame_h, frame_w, 3), (60, 60, 60), dtype=np.uint8)
    cv2.line(frame, (frame_w//2, 0), (frame_w//2, frame_h), (200, 200, 50), 2)
    for _ in range(vehicle_count):
        w, h = random.randint(40, 80), random.randint(25, 45)
        x = random.randint(10, frame_w - w - 10)
        y = random.randint(10, frame_h - h - 10)
        color = random.choice([(180,50,50),(50,130,200),(50,180,80),(200,160,50)])
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, -1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (230,230,230), 1)
    return frame

def simulate_junction_counts(use_yolo=False) -> dict:
    counts = {}
    for lane in LANE_IDS:
        lo, hi = TRAFFIC_PROFILES[lane]
        if use_yolo:
            n = random.randint(lo, hi)
            frame = make_synthetic_frame(n)
            result = detect_vehicles(frame, lane_id=lane)
            counts[lane] = result['weighted_density']
        else:
            counts[lane] = random.randint(lo, hi)
    return counts


# ──────────────────────────────────────────────
# 3. WEBSTER'S SIGNAL TIMING
# ──────────────────────────────────────────────
SATURATION_FLOW      = 1800
COUNT_TO_FLOW_FACTOR = 120

def compute_signal_timing(counts: dict,
                           min_green: int = 10,
                           max_green: int = 60,
                           lost_time_per_phase: int = 3) -> dict:
    n = len(counts)
    L = n * lost_time_per_phase

    flow_ratios = {
        lane: (count * COUNT_TO_FLOW_FACTOR) / SATURATION_FLOW
        for lane, count in counts.items()
    }

    Y = sum(flow_ratios.values())
    if Y >= 1.0:
        Y = 0.9

    C = (1.5 * L + 5) / (1 - Y)
    C = max(60, min(C, 180))

    effective_green = C - L
    y_total = sum(flow_ratios.values()) or 1.0

    green_times = {}
    for lane, y in flow_ratios.items():
        g = (y / y_total) * effective_green
        green_times[lane] = int(max(min_green, min(max_green, g)))

    fixed_green      = int(effective_green / n)
    total_wait_ai    = sum(max_green - g for g in green_times.values())
    total_wait_fixed = sum(max_green - fixed_green for _ in green_times)
    efficiency_gain  = round((1 - total_wait_ai / max(total_wait_fixed, 1)) * 100, 1)

    return {
        'cycle_length':    round(C),
        'green_times':     green_times,
        'fixed_baseline':  fixed_green,
        'efficiency_gain': efficiency_gain,
        'flow_ratios':     {k: round(v, 3) for k, v in flow_ratios.items()}
    }


# ──────────────────────────────────────────────
# 4. EMERGENCY VEHICLE DETECTION
# ──────────────────────────────────────────────
def detect_emergency_lights(frame: np.ndarray) -> dict:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    red_lo1   = cv2.inRange(hsv, np.array([0,   120, 120]), np.array([10,  255, 255]))
    red_lo2   = cv2.inRange(hsv, np.array([170, 120, 120]), np.array([180, 255, 255]))
    red_mask  = cv2.bitwise_or(red_lo1, red_lo2)
    blue_mask = cv2.inRange(hsv, np.array([105, 100, 100]), np.array([130, 255, 255]))

    red_pixels  = int(cv2.countNonZero(red_mask))
    blue_pixels = int(cv2.countNonZero(blue_mask))

    RED_THRESHOLD  = 800
    BLUE_THRESHOLD = 600

    detected     = red_pixels > RED_THRESHOLD or blue_pixels > BLUE_THRESHOLD
    vehicle_type = None
    if red_pixels  > RED_THRESHOLD:  vehicle_type = 'ambulance/fire'
    if blue_pixels > BLUE_THRESHOLD: vehicle_type = 'police'

    return {
        'emergency_detected': detected,
        'vehicle_type':       vehicle_type,
        'red_pixels':         red_pixels,
        'blue_pixels':        blue_pixels
    }

def apply_emergency_override(green_times: dict,
                              emergency_lane: str,
                              clear_duration: int = 60) -> dict:
    override = {lane: 0 for lane in green_times}
    override[emergency_lane] = clear_duration
    return {
        'mode':           'EMERGENCY_OVERRIDE',
        'priority_lane':  emergency_lane,
        'green_times':    override,
        'clear_duration': clear_duration,
        'message':        f'All lanes STOPPED — clearing {emergency_lane} for {clear_duration}s'
    }


# ──────────────────────────────────────────────
# 5. CONGESTION PREDICTOR
# ──────────────────────────────────────────────
class CongestionPredictor:
    LANE_CAPACITY = 20
    WINDOW        = 12
    ALERT_THRESH  = 0.75

    def __init__(self):
        self.history = {lane: deque(maxlen=self.WINDOW) for lane in LANE_IDS}

    def update(self, counts: dict) -> dict:
        predictions = {}
        for lane in LANE_IDS:
            count = counts.get(lane, 0)
            self.history[lane].append(count)

            density   = count / self.LANE_CAPACITY
            hist_norm = [c / self.LANE_CAPACITY for c in self.history[lane]]
            avg       = float(np.mean(hist_norm))

            if len(hist_norm) >= 2:
                xs    = np.arange(len(hist_norm))
                trend = float(np.polyfit(xs, hist_norm, 1)[0])
            else:
                trend = 0.0

            predicted = min(max(avg + trend * 5, 0.0), 1.0)

            status = 'clear'
            if predicted > self.ALERT_THRESH: status = 'warning'
            if predicted > 0.9:               status = 'critical'

            predictions[lane] = {
                'current_density':   round(density, 2),
                'predicted_density': round(predicted, 2),
                'trend':             round(trend, 4),
                'status':            status,
                'history':           [round(h, 2) for h in hist_norm]
            }
        return predictions


predictor = CongestionPredictor()


# ──────────────────────────────────────────────
# 6. FASTAPI APP
# ──────────────────────────────────────────────
app = FastAPI(title='AI Junction Optimizer', version='1.0')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

current_counts:  dict = {lane: 0 for lane in LANE_IDS}
emergency_state: dict = {'active': False, 'lane': None}
event_log:       list = []


def log_event(msg: str, level: str = 'info'):
    event_log.append({
        'timestamp': time.strftime('%H:%M:%S'),
        'message':   msg,
        'level':     level
    })
    if len(event_log) > 50:
        event_log.pop(0)


class DetectRequest(BaseModel):
    frame_b64: str
    lane_id:   str

class ManualCountsRequest(BaseModel):
    North: float = 0
    South: float = 0
    East:  float = 0
    West:  float = 0

class EmergencyRequest(BaseModel):
    active: bool
    lane:   Optional[str] = 'North'


@app.get('/')
def root():
    return {'status': 'online', 'system': 'AI Junction Optimizer'}


@app.post('/detect')
def detect(req: DetectRequest):
    try:
        frame  = decode_frame(req.frame_b64)
        result = detect_vehicles(frame, lane_id=req.lane_id)
        current_counts[req.lane_id] = result['weighted_density']
        log_event(f'Detected {result["raw_count"]} vehicles on {req.lane_id}')
        return result
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post('/counts')
def set_counts(req: ManualCountsRequest):
    global current_counts
    current_counts = req.model_dump()
    log_event(f'Manual counts updated: {current_counts}')
    return {'status': 'updated', 'counts': current_counts}


@app.get('/optimize')
def optimize():
    counts = simulate_junction_counts() if all(
        v == 0 for v in current_counts.values()
    ) else current_counts

    timing      = compute_signal_timing(counts)
    predictions = predictor.update(counts)

    if emergency_state['active'] and emergency_state['lane']:
        override = apply_emergency_override(
            timing['green_times'], emergency_state['lane']
        )
        log_event(f'EMERGENCY OVERRIDE — lane {emergency_state["lane"]}', 'emergency')
        return {
            'mode':        'EMERGENCY',
            'counts':      counts,
            'timing':      override,
            'predictions': predictions,
            'timestamp':   time.strftime('%H:%M:%S')
        }

    log_event(f'Cycle optimized — gain={timing["efficiency_gain"]}%')
    return {
        'mode':        'NORMAL',
        'counts':      counts,
        'timing':      timing,
        'predictions': predictions,
        'timestamp':   time.strftime('%H:%M:%S')
    }


@app.post('/emergency')
def set_emergency(req: EmergencyRequest):
    emergency_state['active'] = req.active
    emergency_state['lane']   = req.lane if req.active else None
    level = 'emergency' if req.active else 'info'
    msg   = (f'EMERGENCY ACTIVATED — {req.lane}' if req.active
             else 'Emergency cleared — resuming normal operation')
    log_event(msg, level)
    return {'status': 'ok', 'emergency': emergency_state, 'message': msg}


@app.get('/status')
def status():
    counts      = simulate_junction_counts() if all(
        v == 0 for v in current_counts.values()
    ) else current_counts
    timing      = compute_signal_timing(counts)
    predictions = predictor.update(counts)
    return {
        'counts':        counts,
        'timing':        timing,
        'predictions':   predictions,
        'emergency':     emergency_state,
        'recent_events': event_log[-10:],
        'timestamp':     time.strftime('%H:%M:%S')
    }


@app.get('/logs')
def get_logs():
    return {'events': event_log}


# ──────────────────────────────────────────────
# 7. RUN
# ──────────────────────────────────────────────
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
