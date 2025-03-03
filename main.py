import os
import uuid
from pathlib import Path
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, File, UploadFile, Request, Form, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from loguru import logger
from pydantic import BaseModel

from pdf_image_extractor import extract_images_from_pdf, extract_and_segment_pdf_pages

# Check if YOLO is available
try:
    from yolo_detector import YOLO_AVAILABLE
except ImportError:
    YOLO_AVAILABLE = False

# Create directories
UPLOAD_DIR = Path("uploads")
STATIC_DIR = Path("static")
IMAGES_DIR = STATIC_DIR / "images"

UPLOAD_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)

app = FastAPI(title="PDF Image Extractor")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


class ImageInfo(BaseModel):
    page: int
    image_number: int
    filename: str
    path: str
    type: str = "unknown"  # New field to indicate the type of image
    confidence: Optional[float] = None  # Confidence score for YOLO detections


class PDFExtractResponse(BaseModel):
    images: List[ImageInfo]
    count: int
    yolo_used: bool = False


class SegmentationPageInfo(BaseModel):
    page: int
    original_page: str
    segmented_page: Optional[str] = None
    boxes_count: int = 0
    success: bool = False


class PDFSegmentationResponse(BaseModel):
    pages: List[SegmentationPageInfo]
    count: int
    success: bool = False


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "yolo_available": YOLO_AVAILABLE})


@app.get("/yolo-status")
async def yolo_status():
    return {"available": YOLO_AVAILABLE}


@app.post("/extract", response_model=PDFExtractResponse)
async def extract_images(
    request: Request, 
    pdf_file: UploadFile = File(...),
    use_yolo: bool = Form(False),
    yolo_confidence: float = Form(0.25)
):
    # Create unique session ID for this upload
    session_id = str(uuid.uuid4())
    session_dir = IMAGES_DIR / session_id
    session_dir.mkdir(exist_ok=True)
    
    # Save uploaded PDF temporarily
    pdf_path = UPLOAD_DIR / f"{session_id}.pdf"
    with open(pdf_path, "wb") as f:
        f.write(await pdf_file.read())
    
    # Check if YOLO is requested but not available
    if use_yolo and not YOLO_AVAILABLE:
        logger.warning("YOLO detection requested but not available")
        use_yolo = False
    
    # Extract images with optional YOLO detection
    images_data = extract_images_from_pdf(
        pdf_path, 
        session_dir,
        only_charts=True,  # We want charts, diagrams and info blocks only
        min_image_size=150,  # Ignore very small images
        use_yolo=use_yolo,
        yolo_confidence=yolo_confidence
    )
    
    # Format response
    images = []
    for img in images_data:
        # Create path relative to static directory for frontend
        rel_path = f"images/{session_id}/{Path(img['path']).name}"
        images.append(ImageInfo(
            page=img["page"],
            image_number=img["image_number"],
            filename=Path(img["path"]).name,
            path=rel_path,
            type=img.get("type", "unknown"),
            confidence=img.get("confidence")
        ))
    
    # Remove temporary PDF
    pdf_path.unlink()
    
    logger.info(f"Extracted {len(images)} charts/diagrams from {pdf_file.filename} (YOLO: {use_yolo})")
    return PDFExtractResponse(images=images, count=len(images), yolo_used=use_yolo)


@app.post("/segment-pdf", response_model=PDFSegmentationResponse)
async def segment_pdf(
    request: Request,
    pdf_file: UploadFile = File(...),
    confidence: float = Form(0.25),
    custom_model: bool = Form(False)
):
    # Create unique session ID for this upload
    session_id = f"session_{uuid.uuid4().hex[:8]}"
    logger.info(f"Starting PDF segmentation session: {session_id}")
    
    # Create session directory
    session_dir = Path("static") / "segmentation" / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    
    # Save uploaded file
    pdf_path = session_dir / f"{session_id}.pdf"
    with open(pdf_path, "wb") as f:
        f.write(await pdf_file.read())
    
    # Determine model to use
    model_path = None
    if custom_model:
        # Use project's custom model
        model_path = "runs/detect/train4/weights/best.pt"
        if not Path(model_path).exists():
            logger.warning(f"Custom model not found at {model_path}, falling back to default")
            model_path = None
    
    # Process PDF
    segmentation_results = extract_and_segment_pdf_pages(
        pdf_path=pdf_path,
        output_dir=session_dir,
        yolo_model_path=model_path,
        confidence=confidence
    )
    
    # Format response with paths relative to static directory
    pages = []
    for result in segmentation_results:
        # Process original page path
        original_page = result["original_page"]
        relative_original = None
        
        if original_page and Path(original_page).exists():
            # Make path relative to the root for browser URLs
            try:
                # If path is already in static directory, format it for the browser
                if "static/" in original_page:
                    relative_original = original_page
                else:
                    # Path might be absolute, try to make it relative to static
                    p = Path(original_page)
                    if "static" in p.parts:
                        # Find where "static" is in the path
                        static_index = p.parts.index("static")
                        # Reconstruct path from that point
                        relative_original = "/".join([""] + list(p.parts[static_index:]))
                    else:
                        # If not in static, just use the original
                        relative_original = f"/{original_page}"
            except Exception as e:
                logger.error(f"Error processing original page path: {e}")
                relative_original = f"/{original_page}"
        
        # Process segmented page path
        segmented_page = result.get("segmented_page")
        relative_segmented = None
        
        if segmented_page and Path(segmented_page).exists():
            try:
                # If path is already in static directory, format it for the browser
                if "static/" in segmented_page:
                    relative_segmented = segmented_page
                else:
                    # Path might be absolute, try to make it relative to static
                    p = Path(segmented_page)
                    if "static" in p.parts:
                        # Find where "static" is in the path
                        static_index = p.parts.index("static")
                        # Reconstruct path from that point
                        relative_segmented = "/".join([""] + list(p.parts[static_index:]))
                    else:
                        # If not in static, just use the original
                        relative_segmented = f"/{segmented_page}"
            except Exception as e:
                logger.error(f"Error processing segmented page path: {e}")
                relative_segmented = f"/{segmented_page}"
        
        # Log processed paths for debugging
        logger.info(f"Original path: {original_page} -> {relative_original}")
        logger.info(f"Segmented path: {segmented_page} -> {relative_segmented}")
        
        pages.append(SegmentationPageInfo(
            page=result["page"],
            original_page=relative_original,
            segmented_page=relative_segmented,
            boxes_count=result.get("boxes_count", 0),
            success=result.get("success", False)
        ))
    
    # Remove temporary PDF
    pdf_path.unlink()
    
    success = len(pages) > 0 and any(page.success for page in pages)
    logger.info(f"Segmented {len(pages)} pages from {pdf_file.filename} (success: {success})")
    
    return PDFSegmentationResponse(
        pages=pages,
        count=len(pages),
        success=success
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PDF Image Extractor Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind the server to. Use 127.0.0.1 for local access only, 0.0.0.0 for all interfaces.")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the server to")
    
    args = parser.parse_args()
    
    logger.info(f"Starting server on {args.host}:{args.port}")
    logger.info(f"You can access the server at http://{args.host if args.host != '0.0.0.0' else 'localhost'}:{args.port}")
    logger.info(f"YOLO detection {'available' if YOLO_AVAILABLE else 'not available'}")
    
    uvicorn.run("main:app", host=args.host, port=args.port, reload=True) 