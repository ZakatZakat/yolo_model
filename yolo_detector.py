from pathlib import Path
import os
from typing import List, Dict, Union, Optional, Tuple
import numpy as np
from loguru import logger
import io
from PIL import Image
import cv2

# Make YOLO dependencies optional
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    logger.warning("Ultralytics YOLO is not installed. YOLO chart detection will be disabled.")
    YOLO_AVAILABLE = False


class YOLOChartDetector:
    """Chart detector using YOLO model"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize YOLO chart detector
        
        Args:
            model_path: Path to the YOLO model file. If None, will use a pretrained model
                        or download one if it doesn't exist.
        """
        self.model = None
        self.available = YOLO_AVAILABLE
        
        if not self.available:
            logger.warning("YOLO detection not available. Install with 'pip install ultralytics'")
            return
            
        model_path = "runs/detect/train5/weights/best.pt"
        self.model = YOLO(model_path)
        self.classes = self.model.names
    
    def detect_charts(self, image_bytes: bytes, 
                     confidence: float = 0.25, 
                     relevant_classes: Optional[List[int]] = None) -> List[Dict]:
        """
        Detect charts in image using YOLO
        
        Args:
            image_bytes: Image data as bytes
            confidence: Confidence threshold for detections
            relevant_classes: List of class IDs to consider as charts/diagrams
                             If None, will use default classes that typically represent charts
                             
        Returns:
            List of dictionaries with detection information
        """
        if not self.available or self.model is None:
            logger.warning("YOLO model not available for detection")
            return []
            
        try:
            # Convert bytes to PIL Image
            img = Image.open(io.BytesIO(image_bytes))
            img_np = np.array(img)
            
            # Run inference
            results = self.model.predict(img_np, conf=confidence)
            
            # If no relevant classes specified, use these common object classes
            # that could represent charts or be part of charts
            if relevant_classes is None:
                # These classes in COCO dataset could be relevant:
                # book, laptop, tv, monitor, etc.
                relevant_classes = [0, 39, 41, 62, 63, 64, 65, 66, 67, 73]
            
            detections = []
            
            for r in results:
                boxes = r.boxes
                
                for box in boxes:
                    cls_id = int(box.cls.item())
                    
                    # Only keep relevant classes that could be charts
                    if relevant_classes is not None and cls_id not in relevant_classes:
                        continue
                        
                    conf = box.conf.item()
                    xyxy = box.xyxy.cpu().numpy()[0]  # Get coordinates
                    
                    detections.append({
                        "bbox": xyxy,
                        "confidence": conf,
                        "class_id": cls_id,
                        "class_name": self.classes[cls_id] if cls_id in self.classes else "unknown"
                    })
            
            return detections
            
        except Exception as e:
            logger.error(f"Error during YOLO detection: {e}")
            return []
    
    def extract_chart_regions(self, 
                             image_bytes: bytes, 
                             detections: List[Dict]) -> List[Dict[str, Union[bytes, str]]]:
        """
        Extract chart regions from image based on YOLO detections
        
        Args:
            image_bytes: Original image as bytes
            detections: List of detections from detect_charts
            
        Returns:
            List of dictionaries with extracted chart images
        """
        if not detections:
            return []
            
        extracted_charts = []
        
        try:
            # Open the image
            img = Image.open(io.BytesIO(image_bytes))
            
            for i, det in enumerate(detections):
                # Extract coordinates
                x1, y1, x2, y2 = det["bbox"]
                
                # Add a margin around the detection
                margin = min(img.width, img.height) * 0.05
                x1 = max(0, x1 - margin)
                y1 = max(0, y1 - margin)
                x2 = min(img.width, x2 + margin)
                y2 = min(img.height, y2 + margin)
                
                # Crop the image
                chart_img = img.crop((x1, y1, x2, y2))
                
                # Convert to bytes
                chart_buffer = io.BytesIO()
                chart_img.save(chart_buffer, format="PNG")
                chart_bytes = chart_buffer.getvalue()
                
                extracted_charts.append({
                    "image": chart_bytes,
                    "ext": "png",
                    "block_type": "yolo_chart",
                    "is_chart": True,
                    "confidence": det["confidence"],
                    "class_name": det["class_name"]
                })
                
        except Exception as e:
            logger.error(f"Error extracting chart regions: {e}")
            
        return extracted_charts
    
    def detect_and_extract_from_page(self, 
                                    page_image_bytes: bytes, 
                                    confidence: float = 0.25) -> List[Dict[str, Union[bytes, str]]]:
        """
        Full pipeline to detect and extract charts from a page image
        
        Args:
            page_image_bytes: Image of the page as bytes
            confidence: Confidence threshold for detections
            
        Returns:
            List of dictionaries with extracted chart images
        """
        # Skip if YOLO not available
        if not self.available:
            return []

        try:
            # Detect charts
            detections = self.detect_charts(page_image_bytes, confidence)
            
            # Extract chart regions
            charts = self.extract_chart_regions(page_image_bytes, detections)
            
            return charts
            
        except Exception as e:
            logger.error(f"Error in YOLO chart detection pipeline: {e}")
            return []
    
    def segment_image(self, 
                     image_bytes: bytes,
                     confidence: float = 0.25,
                     save_path: Optional[str] = None) -> Dict:
        """
        Segment image using YOLO model
        
        Args:
            image_bytes: Image data as bytes
            confidence: Confidence threshold for segmentation
            save_path: Optional path to save the segmented image
            
        Returns:
            Dictionary with segmentation results and path to segmented image
        """
        if not self.available or self.model is None:
            logger.warning("YOLO model not available for segmentation")
            return {"success": False, "error": "YOLO model not available"}
            
        try:
            # Convert bytes to PIL Image
            img = Image.open(io.BytesIO(image_bytes))
            img_np = np.array(img)
            
            # Create directory for save_path if needed
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                
                # Set the directory where YOLO will save the results
                save_dir = os.path.dirname(save_path)
                project_name = os.path.basename(os.path.dirname(save_dir))
                name = os.path.basename(save_dir)
            else:
                # Default save location
                save_dir = "static/segmentation/results"
                os.makedirs(save_dir, exist_ok=True)
                project_name = "static"
                name = "segmentation"
            
            # Run inference with visualization
            # Note: YOLO's save mechanism will create paths like project/name/run_x/image.jpg
            # We'll use the save=True but then manually save to our desired location
            results = self.model.predict(
                img_np, 
                conf=confidence, 
                save=True,
                project=project_name,
                name=name,
                exist_ok=True
            )
            
            # YOLO might have saved the image in its default location. Let's get that path.
            result_path = None
            yolo_saved_path = None
            
            if len(results) > 0:
                # Try to find the saved image path from results
                if hasattr(results[0], 'save_dir'):
                    save_dir_from_results = results[0].save_dir
                    if save_dir_from_results and os.path.exists(save_dir_from_results):
                        files = [f for f in os.listdir(save_dir_from_results) if f.endswith('.jpg') or f.endswith('.png')]
                        if files:
                            # Get the most recently modified file
                            files.sort(key=lambda x: os.path.getmtime(os.path.join(save_dir_from_results, x)), reverse=True)
                            yolo_saved_path = os.path.join(save_dir_from_results, files[0])
            
            # If YOLO saved the image and we have a custom save path, copy it there
            if yolo_saved_path and save_path:
                try:
                    # Copy the file to our desired location
                    import shutil
                    shutil.copy2(yolo_saved_path, save_path)
                    result_path = save_path
                    logger.info(f"Copied YOLO result from {yolo_saved_path} to {save_path}")
                except Exception as e:
                    logger.error(f"Error copying segmentation result: {e}")
                    # Fall back to the YOLO-generated path if copy fails
                    result_path = yolo_saved_path
            elif yolo_saved_path:
                # Use the YOLO-generated path if no custom path specified
                result_path = yolo_saved_path
            elif save_path:
                # If for some reason YOLO didn't save but we have a save path,
                # try to render the image ourselves
                try:
                    # Plot and save the results manually
                    if len(results) > 0:
                        from ultralytics.utils.plotting import Annotator
                        annotator = Annotator(img_np)
                        for r in results:
                            boxes = r.boxes
                            for box in boxes:
                                b = box.xyxy[0].tolist()  # get box coordinates in (top, left, bottom, right) format
                                c = int(box.cls)
                                annotator.box_label(b, f"{self.classes.get(c, c)} {box.conf[0]:.2f}")
                        
                        # Save the annotated image
                        result_img = Image.fromarray(annotator.result())[:, :, ::-1]
                        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                        result_img.save(save_path)
                        result_path = save_path
                except Exception as e:
                    logger.error(f"Error manually saving segmentation result: {e}")
            
            # Process box results
            boxes_data = []
            if len(results) > 0:
                # Access box information if available
                if hasattr(results[0], 'boxes') and results[0].boxes is not None:
                    for i, box in enumerate(results[0].boxes):
                        cls_id = int(box.cls.item()) if hasattr(box, 'cls') else -1
                        cls_name = self.classes.get(cls_id, "unknown") if hasattr(self, 'classes') else "unknown"
                        conf = float(box.conf.item()) if hasattr(box, 'conf') else 0.0
                        
                        if hasattr(box, 'xyxy'):
                            xyxy = box.xyxy.cpu().numpy()[0].tolist() if len(box.xyxy) > 0 else [0, 0, 0, 0]
                        else:
                            xyxy = [0, 0, 0, 0]
                        
                        boxes_data.append({
                            "id": i,
                            "class_id": cls_id,
                            "class_name": cls_name,
                            "confidence": conf,
                            "bbox": xyxy
                        })
            
            logger.info(f"Segmentation completed. Result path: {result_path}")
            return {
                "success": True,
                "boxes": boxes_data,
                "boxes_count": len(boxes_data),
                "result_path": result_path,
            }
            
        except Exception as e:
            logger.error(f"Error during YOLO segmentation: {e}")
            return {"success": False, "error": str(e)}


def initialize_yolo_detector(model_path: Optional[str] = None) -> YOLOChartDetector:
    """Initialize YOLO detector with optional custom model"""
    return YOLOChartDetector(model_path) 