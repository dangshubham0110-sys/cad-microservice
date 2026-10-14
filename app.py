from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import numpy as np
import base64
from rembg import remove

app = FastAPI(title="Pooja Enterprises CAD Contour Engine")

# Enable CORS for Google Apps Script Web App requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImagePayload(BaseModel):
    image_base64: str
    grid_size_mm: float = 10.0  # 10mm cutting mat reference

@app.get("/")
def health_check():
    return {"status": "CAD Engine Active", "version": "1.0.0"}

@app.post("/extract-contour")
def extract_contour(data: ImagePayload):
    try:
        # 1. Decode Base64 Image string
        img_bytes = base64.b64decode(data.image_base64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image data")

        # 2. Background Removal via Rembg
        no_bg_img = remove(img)
        alpha_channel = no_bg_img[:, :, 3]

        # 3. Contour Detection on Isolated Silhouette
        contours, _ = cv2.findContours(alpha_channel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            raise HTTPException(status_code=422, detail="No article contour detected in image")

        # Find largest outer boundary contour
        largest_contour = max(contours, key=cv2.contourArea)

        # 4. Polygon Approximation (Vector Lines)
        epsilon = 0.003 * cv2.arcLength(largest_contour, True)
        approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)

        # 5. Extract Dimensions and Normalized Coordinates
        x, y, w, h = cv2.boundingRect(approx_contour)
        img_h, img_w = alpha_channel.shape
        
        normalized_points = []
        for pt in approx_contour:
            px, py = pt[0]
            normalized_points.append({
                "x": round(float(px) / img_w, 4),
                "y": round(float(py) / img_h, 4)
            })

        contour_area_px = cv2.contourArea(approx_contour)
        bounding_box_area_px = w * h
        solidity = float(contour_area_px) / bounding_box_area_px if bounding_box_area_px > 0 else 1.0

        return {
            "status": "success",
            "contour_points": normalized_points,
            "solidity_ratio": round(solidity, 4),
            "estimated_density_boost": round((1.0 - solidity) * 0.5, 4),
            "raw_aspect_ratio": round(float(w) / h, 4) if h > 0 else 1.0
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
