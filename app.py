from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import numpy as np
import base64
from rembg import remove

app = FastAPI(title="Pooja Enterprises CAD Contour Engine")

# Explicit CORS Settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImagePayload(BaseModel):
    image_base64: str
    grid_size_mm: float = 10.0

@app.get("/")
def health_check():
    return {"status": "CAD Engine Active", "version": "1.0.0"}

@app.post("/extract-contour")
def extract_contour(data: ImagePayload):
    try:
        # 1. Decode Base64 Image
        img_bytes = base64.b64decode(data.image_base64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image data")

        # 2. Background Removal via Rembg
        no_bg_img = remove(img)
        alpha_channel = no_bg_img[:, :, 3]

        # 3. Find Outer Contour Silhouette
        contours, _ = cv2.findContours(alpha_channel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            raise HTTPException(status_code=422, detail="No contour detected")

        largest_contour = max(contours, key=cv2.contourArea)

        # 4. Approximate Polygon Lines
        epsilon = 0.003 * cv2.arcLength(largest_contour, True)
        approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)

        # 5. Crop Bounding Box & Scale Points to Bounding Box
        x, y, w, h = cv2.boundingRect(approx_contour)
        
        normalized_points = []
        for pt in approx_contour:
            px, py = pt[0]
            # Normalize coordinates strictly relative to the article's own bounding box (0.0 to 1.0)
            norm_x = (float(px) - x) / float(w) if w > 0 else 0.5
            norm_y = (float(py) - y) / float(h) if h > 0 else 0.5
            normalized_points.append({
                "x": round(norm_x, 4),
                "y": round(norm_y, 4)
            })

        return {
            "status": "success",
            "contour_points": normalized_points,
            "aspect_ratio": round(float(w) / float(h), 4) if h > 0 else 1.0
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
