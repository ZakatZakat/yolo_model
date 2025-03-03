from pathlib import Path
from typing import List, Dict, Union, Tuple, Optional
import fitz
import numpy as np
from loguru import logger
import io

# Try to import OpenCV, but make it optional
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    logger.warning("OpenCV (cv2) is not installed. Advanced chart detection will be disabled.")
    OPENCV_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    logger.warning("PIL is not installed. Advanced chart detection will be disabled.")
    PIL_AVAILABLE = False

# Import YOLO detector
try:
    from yolo_detector import initialize_yolo_detector, YOLO_AVAILABLE
    if YOLO_AVAILABLE:
        logger.info("YOLO chart detector initialized and available")
except ImportError:
    logger.warning("YOLO detector module not found. YOLO detection will be disabled.")
    YOLO_AVAILABLE = False


def is_chart_or_diagram(image_bytes: bytes) -> bool:
    """
    Analyze image to determine if it's likely a chart, diagram, or information block.
    
    Uses enhanced heuristics to detect charts like bar graphs, line charts, etc.
    """
    # If OpenCV or PIL is not available, consider all images as potential charts
    if not OPENCV_AVAILABLE or not PIL_AVAILABLE:
        return True
        
    try:
        # Convert to OpenCV format
        img_array = np.array(Image.open(io.BytesIO(image_bytes)))
        
        # Skip very small images
        if img_array.shape[0] < 100 or img_array.shape[1] < 100:
            return False
            
        if len(img_array.shape) == 3:  # Color image
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:  # Already grayscale
            gray = img_array
            
        # Edge detection - useful for finding grid lines and axes
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.count_nonzero(edges) / edges.size
        
        # Look for horizontal and vertical lines (common in charts)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, minLineLength=40, maxLineGap=10)
        
        # Analyze detected lines
        if lines is not None and len(lines) > 0:
            horizontal_lines = 0
            vertical_lines = 0
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(y2 - y1) < 10:  # Horizontal line
                    horizontal_lines += 1
                if abs(x2 - x1) < 10:  # Vertical line
                    vertical_lines += 1
                    
            # Charts typically have some horizontal and vertical lines for axes and gridlines
            has_axis_lines = horizontal_lines >= 2 or vertical_lines >= 2
        else:
            has_axis_lines = False
            
        # Color analysis - charts often have distinct color patterns
        if len(img_array.shape) == 3:
            # Resize for faster processing
            img_resized = cv2.resize(img_array, (100, 100))
            
            # Get unique colors
            colors = np.unique(img_resized.reshape(-1, img_resized.shape[2]), axis=0)
            unique_color_count = len(colors)
            
            # Charts usually have a moderate number of distinct colors
            has_chart_colors = 5 <= unique_color_count <= 20
        else:
            has_chart_colors = True  # Grayscale images could be charts too
        
        # Text detection - charts typically have some text for labels
        # This is a simplified approach since proper text detection requires more complex methods
        # Look for small connected components that might be text
        ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        small_contours = 0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if 5 <= w <= 50 and 5 <= h <= 50:  # Size range typical for text
                small_contours += 1
        
        has_potential_text = small_contours >= 5
        
        # Final decision based on combined criteria
        is_likely_chart = (
            (edge_density > 0.03 and edge_density < 0.2) and  # Some edges, but not too noisy
            (has_axis_lines or has_chart_colors) and  # Either has axis lines or chart-like colors
            has_potential_text  # Has potential text elements
        )
        
        return is_likely_chart
        
    except Exception as e:
        logger.warning(f"Error analyzing image: {e}")
        return True  # Include by default if analysis fails


def extract_text_blocks_as_images(
    page: fitz.Page, 
    min_block_size: int = 200
) -> List[Dict[str, Union[bytes, str]]]:
    """Extract text blocks with their surrounding regions as images"""
    blocks = []
    
    # Get text blocks
    text_blocks = page.get_text("blocks")
    
    for i, block in enumerate(text_blocks):
        x0, y0, x1, y1, text, block_no, block_type = block
        
        # Filter small blocks
        width, height = x1 - x0, y1 - y0
        if width < min_block_size or height < min_block_size:
            continue
            
        # Expand region slightly to capture any surrounding elements
        margin = 20
        rect = fitz.Rect(x0-margin, y0-margin, x1+margin, y1+margin)
        
        # Create image from this region
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=rect)
        img_data = pix.tobytes()
        
        blocks.append({
            "image": img_data,
            "ext": "png",
            "block_type": "text",
            "is_chart": True  # Mark text blocks as potential information blocks
        })
        
    return blocks


def detect_charts_and_diagrams(
    page: fitz.Page,
    diagram_detection_threshold: float = 0.1
) -> List[Dict[str, Union[bytes, str]]]:
    """Detect regions that might contain charts or diagrams based on vector graphics and content analysis"""
    result = []
    
    # Get the page as a drawing
    paths = page.get_drawings()
    
    # Also get text blocks to identify potential titles and legends
    text_blocks = page.get_text("blocks")
    
    if not paths:
        return result
        
    # Analyze paths to identify potential chart/diagram regions
    rects = []
    
    # First detect potential graphic elements that could be parts of charts
    chart_elements = []
    
    # Look for horizontal or vertical lines (typical for axes)
    for path in paths:
        for item in path["items"]:
            if item[0] == "l":  # line
                x0, y0 = item[1]
                x1, y1 = item[2]
                width = abs(x1 - x0)
                height = abs(y1 - y0)
                
                # Check if this is likely an axis line (horizontal or vertical)
                is_horizontal = height < 5 and width > 50
                is_vertical = width < 5 and height > 50
                
                if is_horizontal or is_vertical:
                    chart_elements.append((x0, y0, x1, y1))
    
    # If we have enough axis-like elements, this page likely contains charts
    if len(chart_elements) >= 2:
        # Extract all drawing elements to form chart areas
        for path in paths:
            # Extract bounding box of this drawing element
            points = []
            for item in path["items"]:
                if item[0] == "l":  # line
                    points.extend([item[1], item[2]])
                elif item[0] == "re":  # rectangle
                    x0, y0, x1, y1 = item[1]
                    points.extend([(x0, y0), (x1, y1)])
            
            if points:
                # Create bounding rectangle
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                rect = fitz.Rect(min(xs), min(ys), max(xs), max(ys))
                
                # Filter out very small elements
                if rect.width > 30 and rect.height > 30:
                    rects.append(rect)
    
    # Merge overlapping rectangles to form larger chart areas
    merged_rects = []
    while rects:
        current = rects.pop(0)
        overlaps = [i for i, r in enumerate(rects) if current.intersects(r)]
        
        while overlaps:
            idx = overlaps[0]
            current = current.include_rect(rects[idx])
            rects.pop(idx)
            overlaps = [i for i, r in enumerate(rects) if current.intersects(r)]
            
        merged_rects.append(current)
    
    # Now look for chart titles and legends to include in the bounding box
    final_chart_areas = []
    for rect in merged_rects:
        if rect.width < 100 or rect.height < 100:
            continue  # Skip small areas
        
        chart_area = rect.copy()
        
        # Look for titles above the chart
        for block in text_blocks:
            x0, y0, x1, y1, text, block_no, block_type = block
            block_rect = fitz.Rect(x0, y0, x1, y1)
            
            # If text is right above the chart or inside it at the top
            is_title = (y1 <= chart_area.y0 + 50 and  # Text is above or at top of chart
                       x0 >= chart_area.x0 - 100 and  # Text is not too far left
                       x1 <= chart_area.x1 + 100 and  # Text is not too far right 
                       y0 >= chart_area.y0 - 150)     # Text is not too far above
            
            # If text is to the right or inside on right (legend)
            is_legend = (x0 >= chart_area.x1 - 100 and  # Text is near right edge
                        y0 >= chart_area.y0 - 50 and   # Text is not too high above chart
                        y1 <= chart_area.y1 + 50)      # Text is not too low below chart
            
            # If text is below (axis labels, etc.)
            is_bottom_label = (y0 >= chart_area.y1 - 50 and  # Text is near bottom
                             y1 <= chart_area.y1 + 100 and  # Text is not too low
                             x0 >= chart_area.x0 - 50 and   # Text is not too far left
                             x1 <= chart_area.x1 + 50)      # Text is not too far right
            
            if is_title or is_legend or is_bottom_label:
                chart_area.include_rect(block_rect)
        
        # Expand a bit to ensure we capture everything
        expanded_rect = fitz.Rect(
            chart_area.x0 - 20, 
            chart_area.y0 - 20,
            chart_area.x1 + 20, 
            chart_area.y1 + 20
        )
        
        # Add this complete chart area
        final_chart_areas.append(expanded_rect)
    
    # Create images from these regions
    for i, rect in enumerate(final_chart_areas):
        try:
            # Create image with higher resolution for clarity
            pix = page.get_pixmap(matrix=fitz.Matrix(3, 3), clip=rect)
            img_data = pix.tobytes()
            
            result.append({
                "image": img_data,
                "ext": "png",
                "block_type": "chart",
                "is_chart": True
            })
        except Exception as e:
            logger.warning(f"Error extracting chart area: {e}")
    
    # If no charts found with vector analysis, try a more aggressive approach
    # Look for clusters of graphics and text that might form charts
    if not result:
        # Get all images
        img_list = page.get_images(full=True)
        imgs = []
        
        for img in img_list:
            xref = img[0]
            bbox = page.get_image_bbox(xref)
            if bbox and bbox.width > 100 and bbox.height > 100:
                imgs.append(bbox)
        
        # Combine image areas with nearby text blocks
        for img_bbox in imgs:
            chart_area = img_bbox.copy()
            
            # Look for nearby text that could be labels or legends
            for block in text_blocks:
                x0, y0, x1, y1, text, block_no, block_type = block
                block_rect = fitz.Rect(x0, y0, x1, y1)
                
                # If text is close to the image
                if (abs(block_rect.x0 - chart_area.x1) < 100 or  # right of image
                    abs(block_rect.x1 - chart_area.x0) < 100 or  # left of image
                    abs(block_rect.y0 - chart_area.y1) < 100 or  # below image
                    abs(block_rect.y1 - chart_area.y0) < 100):   # above image
                    chart_area.include_rect(block_rect)
            
            # Expand a bit
            expanded_rect = fitz.Rect(
                chart_area.x0 - 20, 
                chart_area.y0 - 20,
                chart_area.x1 + 20, 
                chart_area.y1 + 20
            )
            
            # Create image
            try:
                pix = page.get_pixmap(matrix=fitz.Matrix(3, 3), clip=expanded_rect)
                img_data = pix.tobytes()
                
                result.append({
                    "image": img_data,
                    "ext": "png",
                    "block_type": "chart_with_text",
                    "is_chart": True
                })
            except Exception as e:
                logger.warning(f"Error extracting chart with text: {e}")
    
    return result


def extract_images_from_pdf(
    pdf_path: Union[str, Path], 
    output_dir: Union[str, Path],
    only_charts: bool = True,
    min_image_size: int = 100,
    use_yolo: bool = False,
    yolo_confidence: float = 0.25,
    yolo_model_path: Optional[str] = None
) -> List[Dict[str, Union[str, int]]]:
    """Extract images, charts, and diagrams from PDF file
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save extracted images
        only_charts: Only extract charts/diagrams/info blocks (not regular images)
        min_image_size: Minimum size for images to extract
        use_yolo: Whether to use YOLO model for chart detection
        yolo_confidence: Confidence threshold for YOLO detections
        yolo_model_path: Path to custom YOLO model file, if None uses default
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    extracted_images = []
    
    # Initialize YOLO detector if enabled
    yolo_detector = None
    if use_yolo and YOLO_AVAILABLE:
        try:
            yolo_detector = initialize_yolo_detector(yolo_model_path)
            logger.info("YOLO detector initialized for chart detection")
        except Exception as e:
            logger.error(f"Failed to initialize YOLO detector: {e}")
    
    try:
        doc = fitz.open(pdf_path)
        
        for page_num, page in enumerate(doc):
            img_idx = 0
            
            # If YOLO is enabled, use it for chart detection
            if use_yolo and yolo_detector and yolo_detector.available:
                try:
                    # Render the whole page as an image with high resolution
                    pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))
                    page_image = pix.tobytes()
                    
                    # Detect and extract charts using YOLO
                    yolo_charts = yolo_detector.detect_and_extract_from_page(page_image, yolo_confidence)
                    
                    for chart in yolo_charts:
                        image_bytes = chart["image"]
                        image_ext = chart["ext"]
                        
                        image_name = f"page_{page_num + 1}_yolo_chart_{img_idx + 1}.{image_ext}"
                        image_path = output_dir / image_name
                        
                        with open(image_path, "wb") as img_file:
                            img_file.write(image_bytes)
                        
                        extracted_images.append({
                            "page": page_num + 1,
                            "image_number": img_idx + 1,
                            "path": str(image_path),
                            "extension": image_ext,
                            "type": "yolo_chart",
                            "confidence": chart.get("confidence", 0)
                        })
                        
                        img_idx += 1
                        
                    logger.info(f"YOLO detected {len(yolo_charts)} charts on page {page_num + 1}")
                    
                    # If we found charts with YOLO and only want charts, 
                    # we can skip traditional methods for this page
                    if len(yolo_charts) > 0 and only_charts:
                        continue  # Move to next page
                        
                except Exception as e:
                    logger.error(f"Error using YOLO detection on page {page_num + 1}: {e}")
            
            # Traditional detection methods
            # 1. Extract regular embedded images
            image_list = page.get_images()
            
            for img in image_list:
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    # Check image dimensions if PIL is available
                    image_too_small = False
                    if PIL_AVAILABLE:
                        img_obj = Image.open(io.BytesIO(image_bytes))
                        width, height = img_obj.size
                        
                        # Skip small images (likely to be icons, bullets, etc.)
                        if width < min_image_size or height < min_image_size:
                            image_too_small = True
                    
                    if image_too_small:
                        continue
                    
                    # Apply chart detection if required
                    if only_charts and not is_chart_or_diagram(image_bytes):
                        continue
                    
                    # Save image
                    image_name = f"page_{page_num + 1}_img_{img_idx + 1}.{image_ext}"
                    image_path = output_dir / image_name
                    
                    with open(image_path, "wb") as img_file:
                        img_file.write(image_bytes)
                    
                    extracted_images.append({
                        "page": page_num + 1,
                        "image_number": img_idx + 1,
                        "path": str(image_path),
                        "extension": image_ext,
                        "type": "embedded"
                    })
                    
                    img_idx += 1
                except Exception as e:
                    logger.warning(f"Error processing image on page {page_num + 1}: {e}")
            
            # 2. Detect charts and diagrams from vector graphics with improved algorithm
            try:
                charts = detect_charts_and_diagrams(page)
                
                for chart in charts:
                    image_bytes = chart["image"]
                    image_ext = chart["ext"]
                    
                    image_name = f"page_{page_num + 1}_chart_{img_idx + 1}.{image_ext}"
                    image_path = output_dir / image_name
                    
                    with open(image_path, "wb") as img_file:
                        img_file.write(image_bytes)
                    
                    extracted_images.append({
                        "page": page_num + 1,
                        "image_number": img_idx + 1,
                        "path": str(image_path),
                        "extension": image_ext,
                        "type": "chart"
                    })
                    
                    img_idx += 1
            except Exception as e:
                logger.warning(f"Error detecting charts on page {page_num + 1}: {e}")
            
            # 3. Extract text blocks as potential information blocks
            if only_charts:
                try:
                    text_blocks = extract_text_blocks_as_images(page)
                    
                    for block in text_blocks:
                        image_bytes = block["image"]
                        image_ext = block["ext"]
                        
                        image_name = f"page_{page_num + 1}_block_{img_idx + 1}.{image_ext}"
                        image_path = output_dir / image_name
                        
                        with open(image_path, "wb") as img_file:
                            img_file.write(image_bytes)
                        
                        extracted_images.append({
                            "page": page_num + 1,
                            "image_number": img_idx + 1,
                            "path": str(image_path),
                            "extension": image_ext,
                            "type": "info_block"
                        })
                        
                        img_idx += 1
                except Exception as e:
                    logger.warning(f"Error extracting text blocks on page {page_num + 1}: {e}")
        
        logger.info(f"Extracted {len(extracted_images)} charts/diagrams/blocks from {pdf_path}")
        return extracted_images
                
    except Exception as e:
        logger.error(f"Error extracting charts/diagrams from PDF: {e}")
        raise 


def extract_and_segment_pdf_pages(
    pdf_path: Union[str, Path],
    output_dir: Union[str, Path],
    yolo_model_path: Optional[str] = None,
    confidence: float = 0.25
) -> List[Dict[str, Union[str, int]]]:
    """
    Extract pages from PDF as images and perform segmentation on each page
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save extracted images and segmentation results
        yolo_model_path: Path to YOLO model weights (optional)
        confidence: Confidence threshold for YOLO
        
    Returns:
        List of dictionaries with page information and results
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    
    # Attempt to import YOLO detector
    try:
        from yolo_detector import initialize_yolo_detector
        detector = initialize_yolo_detector(yolo_model_path)
        if not detector.available:
            logger.warning("YOLO detector not initialized properly")
            return []
    except ImportError:
        logger.error("Failed to import YOLO detector")
        return []
    except Exception as e:
        logger.error(f"Error initializing YOLO detector: {e}")
        return []
    
    # Create directories for pages and segmentation
    pages_dir = output_dir / "pages"
    segmentation_dir = output_dir / "segmentation"
    pages_dir.mkdir(parents=True, exist_ok=True)
    segmentation_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    try:
        doc = fitz.open(pdf_path)
        
        for page_num, page in enumerate(doc):
            try:
                # Render page as high-resolution image
                pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))
                page_bytes = pix.pil_tobytes(format="PNG")
                
                # Save original page
                page_filename = f"page_{page_num + 1}.png"
                page_path = pages_dir / page_filename
                
                with open(page_path, "wb") as f:
                    f.write(page_bytes)
                
                # Perform segmentation
                seg_filename = f"page_{page_num + 1}_segmented.png"
                seg_path = segmentation_dir / seg_filename
                
                logger.info(f"Segmenting page {page_num + 1}, saving to {seg_path}")
                
                seg_result = detector.segment_image(
                    page_bytes, 
                    confidence=confidence,
                    save_path=str(seg_path)
                )
                
                # Get result path from segmentation
                segmented_page = seg_result.get("segmented_page")
                if not segmented_page and "result_path" in seg_result:
                    segmented_page = seg_result.get("result_path")
                
                # Verify the segmented image exists
                if segmented_page and Path(segmented_page).exists():
                    logger.info(f"Segmentation successful, saved to {segmented_page}")
                else:
                    logger.warning(f"Segmentation result path not found or invalid: {segmented_page}")
                
                # Log results
                boxes_count = seg_result.get("boxes_count", 0)
                success = seg_result.get("success", False)
                
                logger.info(f"Page {page_num + 1} segmentation: {boxes_count} boxes detected, success: {success}")
                
                results.append({
                    "page": page_num + 1,
                    "original_page": str(page_path),
                    "segmented_page": segmented_page,
                    "boxes_count": boxes_count,
                    "success": success
                })
            
            except Exception as e:
                logger.error(f"Error processing page {page_num + 1}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                
                # Still add the page to results, but mark as failed
                results.append({
                    "page": page_num + 1,
                    "original_page": str(page_path) if 'page_path' in locals() else None,
                    "segmented_page": None,
                    "boxes_count": 0,
                    "success": False,
                    "error": str(e)
                })
            
        doc.close()
        logger.info(f"PDF extraction and segmentation completed: {len(results)} pages processed")
        return results
        
    except Exception as e:
        logger.error(f"Error extracting and segmenting PDF: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return results 