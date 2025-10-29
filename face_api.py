# face_api.py
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import mediapipe as mp
import io
import os

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=True, max_num_faces=1,
                             refine_landmarks=True, min_detection_confidence=0.5)

def read_image_from_bytes(data: bytes):
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def landmark_list_to_np(landmarks, w, h):
    pts = []
    for lm in landmarks:
        pts.append((int(lm.x * w), int(lm.y * h)))
    return pts

def eye_aspect_ratio(eye_pts):
    # eye_pts is list of 6 points: compute simple EAR
    # using vertical / horizontal distances
    A = np.linalg.norm(np.array(eye_pts[1]) - np.array(eye_pts[5]))
    B = np.linalg.norm(np.array(eye_pts[2]) - np.array(eye_pts[4]))
    C = np.linalg.norm(np.array(eye_pts[0]) - np.array(eye_pts[3]))
    if C == 0:
        return 0.0
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth_pts):
    # use simple mouth opening ratio: vertical / horizontal
    A = np.linalg.norm(np.array(mouth_pts[2]) - np.array(mouth_pts[10]))  # upper inner lip to lower inner lip
    C = np.linalg.norm(np.array(mouth_pts[0]) - np.array(mouth_pts[6]))   # left to right
    if C == 0:
        return 0.0
    return A / C

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    Accepts an image (single frame). Returns JSON:
    {
      "sleep_score": float,
      "mouth_score": float,
      "state": "sleeping"|"talking"|"ok"|"no_face",
      "debug": {...}
    }
    """
    data = await file.read()
    img = read_image_from_bytes(data)
    if img is None:
        return JSONResponse({"error": "Could not decode image"}, status_code=400)

    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(img_rgb)
    if results.multi_face_landmarks is None:
        return {"state": "no_face", "faces_detected": 0}

    lm = results.multi_face_landmarks[0].landmark
    pts = landmark_list_to_np(lm, w, h)

    # MediaPipe face mesh landmark indices for eyes/mouth (common mapping)
    # left eye: use landmarks around left eye (approx indices)
    LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]   # remodelled for mp mesh
    RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]
    # mouth (outer/inner)
    MOUTH_INNER_IDX = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 191]  # we'll select subset
    # For simplicity pick a small subset for mouth vertical/horizontal
    MOUTH_VERT_IDX = [13, 14]  # top inner lip, bottom inner lip (approx)
    MOUTH_HORZ_IDX = [78, 308]  # left & right mouth corners (approx)

    try:
        left_eye_pts = [pts[i] for i in LEFT_EYE_IDX]
        right_eye_pts = [pts[i] for i in RIGHT_EYE_IDX]
    except Exception:
        # fallback indices if mapping mismatch
        return {"state": "no_face", "faces_detected": 1}

    # compute EAR using first 6 points (approx)
    l_ear = eye_aspect_ratio(left_eye_pts[:6])
    r_ear = eye_aspect_ratio(right_eye_pts[:6])
    ear = (l_ear + r_ear) / 2.0

    # compute mouth ratio
    try:
        mouth_pts = [pts[i] for i in MOUTH_INNER_IDX]
        # choose representative vertical and horizontal
        top = pts[13] if 13 < len(pts) else pts[MOUTH_INNER_IDX[0]]
        bottom = pts[14] if 14 < len(pts) else pts[MOUTH_INNER_IDX[1]]
        left = pts[78] if 78 < len(pts) else pts[MOUTH_INNER_IDX[2]]
        right = pts[308] if 308 < len(pts) else pts[MOUTH_INNER_IDX[3]]
        mar = np.linalg.norm(np.array(top) - np.array(bottom)) / (np.linalg.norm(np.array(left) - np.array(right)) + 1e-6)
    except Exception:
        mar = 0.0

    # Simple thresholds (tune for your setup)
    EAR_SLEEP_THRESHOLD = 0.18   # below this considered eyes closed in a frame
    MAR_TALK_THRESHOLD = 0.04    # above this considered mouth open
    # We cannot decide sleeping across frames here, so return the raw numbers.
    state = "ok"
    if ear < EAR_SLEEP_THRESHOLD:
        state = "sleeping"
    elif mar > MAR_TALK_THRESHOLD:
        state = "talking"

    # VERY simple ID detection heuristic:
    # detect large rectangle (card) in lower half with high edge density
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    # find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    id_found = False
    for c in contours:
        x,y,wc,hc = cv2.boundingRect(c)
        area = wc*hc
        if area < 1000:
            continue
        # check aspect ratio for card-like shape and located in center/lower region
        ar = wc / float(hc+1e-6)
        if 0.6 < ar < 2.0 and y > (h//4):  # located roughly in lower half
            id_found = True
            break

    if not id_found:
        # prefer to change state only if no face state already
        if state == "ok":
            state = "no_id"

    return {
        "state": state,
        "faces_detected": 1,
        "ear": float(ear),
        "mar": float(mar),
        "id_found": bool(id_found)
    }

# Render runner
if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("face_api:app", host="0.0.0.0", port=port)
