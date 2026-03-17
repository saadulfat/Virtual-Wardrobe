import cv2
import numpy as np
import mediapipe as mp
from sklearn.cluster import KMeans
from collections import Counter
import webcolors
from colormath.color_objects import sRGBColor, LabColor, XYZColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
import logging
from typing import List, Tuple, Dict, Optional
import os

logger = logging.getLogger(__name__)

class ColorDetector:
    """Enhanced color detection using MediaPipe and advanced color analysis"""
    
    def __init__(self):
        # Initialize MediaPipe Selfie Segmentation
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.segment = self.mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
        
        # Extended color mapping for better recognition
        self.color_names = {
            'black': [(0, 0, 0), (40, 40, 40)],
            'white': [(240, 240, 240), (255, 255, 255)],
            'gray': [(80, 80, 80), (160, 160, 160)],
            'red': [(150, 0, 0), (255, 100, 100)],
            'blue': [(0, 0, 150), (100, 100, 255)],
            'green': [(0, 150, 0), (100, 255, 100)],
            'yellow': [(200, 200, 0), (255, 255, 100)],
            'orange': [(255, 140, 0), (255, 200, 100)],
            'purple': [(128, 0, 128), (200, 100, 200)],
            'pink': [(255, 192, 203), (255, 220, 220)],
            'brown': [(101, 67, 33), (160, 120, 80)],
            'beige': [(245, 245, 220), (255, 248, 235)],
            'navy': [(0, 0, 80), (50, 50, 128)],
            'maroon': [(128, 0, 0), (150, 50, 50)],
            'olive': [(128, 128, 0), (150, 150, 50)],
            'turquoise': [(64, 224, 208), (150, 255, 240)],
            'coral': [(255, 127, 80), (255, 160, 120)],
            'lime': [(0, 255, 0), (100, 255, 100)],
            'indigo': [(75, 0, 130), (120, 50, 180)],
            'violet': [(238, 130, 238), (255, 180, 255)],
            'gold': [(255, 215, 0), (255, 235, 50)],
            'silver': [(192, 192, 192), (220, 220, 220)],
            'khaki': [(240, 230, 140), (255, 245, 170)],
            'crimson': [(220, 20, 60), (255, 60, 100)],
            'teal': [(0, 128, 128), (50, 180, 180)],
        }
        
        # Color similarity thresholds
        self.similarity_threshold = 30
        
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Load and preprocess image"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize if too large (for faster processing)
            height, width = image_rgb.shape[:2]
            if height > 1000 or width > 1000:
                scale = min(1000/height, 1000/width)
                new_height, new_width = int(height * scale), int(width * scale)
                image_rgb = cv2.resize(image_rgb, (new_width, new_height))
            
            return image_rgb
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise
    
    def segment_clothing(self, image_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Use MediaPipe to segment clothing/person from background"""
        try:
            # Process with MediaPipe
            results = self.segment.process(image_rgb)
            
            if results.segmentation_mask is None:
                raise ValueError("MediaPipe failed to generate segmentation mask")
            
            # Create binary mask (threshold at 0.7 for better precision)
            mask = results.segmentation_mask
            clothing_mask = (mask > 0.7).astype(np.uint8)
            
            # Apply morphological operations to clean up mask
            kernel = np.ones((3, 3), np.uint8)
            clothing_mask = cv2.morphologyEx(clothing_mask, cv2.MORPH_CLOSE, kernel)
            clothing_mask = cv2.morphologyEx(clothing_mask, cv2.MORPH_OPEN, kernel)
            
            # Apply mask to image
            segmented_image = cv2.bitwise_and(image_rgb, image_rgb, mask=clothing_mask)
            
            return segmented_image, clothing_mask
            
        except Exception as e:
            logger.error(f"Error in clothing segmentation: {e}")
            raise
    
    def extract_dominant_colors(self, segmented_image: np.ndarray, mask: np.ndarray, k: int = 5) -> List[Tuple[np.ndarray, int]]:
        """Extract dominant colors using K-means clustering"""
        try:
            # Get pixels where mask is active
            cloth_pixels = segmented_image[mask == 1]
            
            if len(cloth_pixels) == 0:
                raise ValueError("No clothing pixels found in the image")
            
            # Remove near-black pixels (shadows, very dark areas)
            filtered_pixels = []
            for pixel in cloth_pixels:
                r, g, b = pixel
                # Skip very dark pixels (likely shadows) and very bright pixels (likely highlights)
                if not (r < 20 and g < 20 and b < 20) and not (r > 250 and g > 250 and b > 250):
                    filtered_pixels.append(pixel)
            
            if len(filtered_pixels) == 0:
                raise ValueError("No valid clothing pixels after filtering")
            
            filtered_pixels = np.array(filtered_pixels)
            
            # Sample pixels for faster processing if too many
            if len(filtered_pixels) > 10000:
                idx = np.random.choice(len(filtered_pixels), 10000, replace=False)
                filtered_pixels = filtered_pixels[idx]
            
            # Apply K-means clustering
            kmeans = KMeans(n_clusters=min(k, len(filtered_pixels)), random_state=42, n_init=10)
            kmeans.fit(filtered_pixels)
            
            colors = kmeans.cluster_centers_.astype(int)
            labels = kmeans.labels_
            label_counts = Counter(labels)
            
            # Return colors with their frequency
            color_frequency = []
            for idx, count in label_counts.items():
                color_frequency.append((colors[idx], count))
            
            # Sort by frequency (most common first)
            color_frequency.sort(key=lambda x: x[1], reverse=True)
            
            return color_frequency
            
        except Exception as e:
            logger.error(f"Error extracting dominant colors: {e}")
            raise
    
    def rgb_to_color_name(self, rgb: Tuple[int, int, int]) -> str:
        """Convert RGB values to human-readable color name"""
        try:
            r, g, b = rgb
            
            # First try exact CSS color match
            try:
                closest_name = webcolors.rgb_to_name((r, g, b))
                return closest_name.lower()
            except ValueError:
                pass
            
            # If exact match fails, find closest custom color
            min_distance = float('inf')
            closest_color = 'unknown'
            
            for color_name, (min_rgb, max_rgb) in self.color_names.items():
                # Check if RGB falls within the range
                if (min_rgb[0] <= r <= max_rgb[0] and 
                    min_rgb[1] <= g <= max_rgb[1] and 
                    min_rgb[2] <= b <= max_rgb[2]):
                    return color_name
                
                # Calculate distance to range center
                center_r = (min_rgb[0] + max_rgb[0]) / 2
                center_g = (min_rgb[1] + max_rgb[1]) / 2
                center_b = (min_rgb[2] + max_rgb[2]) / 2
                
                distance = np.sqrt((r - center_r)**2 + (g - center_g)**2 + (b - center_b)**2)
                
                if distance < min_distance:
                    min_distance = distance
                    closest_color = color_name
            
            # Use advanced color difference calculation for better accuracy
            try:
                target_color = sRGBColor(r/255.0, g/255.0, b/255.0)
                target_lab = convert_color(target_color, LabColor)
                
                min_delta = float('inf')
                best_match = closest_color
                
                # Check against CSS colors
                css_colors = {
                    'red': (255, 0, 0), 'blue': (0, 0, 255), 'green': (0, 128, 0),
                    'yellow': (255, 255, 0), 'orange': (255, 165, 0), 'purple': (128, 0, 128),
                    'pink': (255, 192, 203), 'brown': (165, 42, 42), 'black': (0, 0, 0),
                    'white': (255, 255, 255), 'gray': (128, 128, 128), 'navy': (0, 0, 128),
                    'maroon': (128, 0, 0), 'olive': (128, 128, 0), 'teal': (0, 128, 128)
                }
                
                for name, (cr, cg, cb) in css_colors.items():
                    css_color = sRGBColor(cr/255.0, cg/255.0, cb/255.0)
                    css_lab = convert_color(css_color, LabColor)
                    delta = delta_e_cie2000(target_lab, css_lab)
                    
                    if delta < min_delta and delta < 20:  # Reasonable threshold
                        min_delta = delta
                        best_match = name
                
                return best_match
                
            except Exception:
                # Fallback to simple distance
                return closest_color
            
        except Exception as e:
            logger.error(f"Error converting RGB to color name: {e}")
            return 'unknown'
    
    def detect_clothing_colors(self, image_path: str, max_colors: int = 3) -> List[Dict]:
        """Main function to detect clothing colors from image"""
        try:
            logger.info(f"Starting color detection for image: {image_path}")
            
            # Check if file exists
            if not os.path.exists(image_path):
                raise ValueError(f"Image file not found: {image_path}")
            
            # Step 1: Preprocess image
            image_rgb = self.preprocess_image(image_path)
            logger.info("Image preprocessed successfully")
            
            # Step 2: Segment clothing
            segmented_image, clothing_mask = self.segment_clothing(image_rgb)
            logger.info("Clothing segmentation completed")
            
            # Step 3: Extract dominant colors
            color_frequency = self.extract_dominant_colors(segmented_image, clothing_mask, k=max_colors+2)
            logger.info(f"Extracted {len(color_frequency)} color clusters")
            
            # Step 4: Convert to color names and filter
            detected_colors = []
            processed_colors = set()
            
            for i, (color_rgb, frequency) in enumerate(color_frequency[:max_colors*2]):  # Get extra colors to filter
                rgb_tuple = tuple(color_rgb)
                color_name = self.rgb_to_color_name(rgb_tuple)
                
                # Skip if we already have this color name
                if color_name in processed_colors:
                    continue
                
                # Calculate color confidence based on frequency and color validity
                total_pixels = sum(freq for _, freq in color_frequency)
                confidence = (frequency / total_pixels) * 100
                
                detected_colors.append({
                    'color_name': color_name,
                    'rgb': rgb_tuple,
                    'confidence': round(confidence, 2),
                    'frequency': frequency
                })
                
                processed_colors.add(color_name)
                
                if len(detected_colors) >= max_colors:
                    break
            
            # Sort by confidence
            detected_colors.sort(key=lambda x: x['confidence'], reverse=True)
            
            logger.info(f"Color detection completed. Found {len(detected_colors)} colors")
            return detected_colors
            
        except Exception as e:
            logger.error(f"Error in color detection: {e}")
            return [{'color_name': 'unknown', 'rgb': (128, 128, 128), 'confidence': 0, 'frequency': 0}]
    
    def get_primary_color(self, image_path: str) -> str:
        """Get the single most dominant color name"""
        try:
            colors = self.detect_clothing_colors(image_path, max_colors=1)
            if colors:
                return colors[0]['color_name']
            return 'unknown'
        except Exception as e:
            logger.error(f"Error getting primary color: {e}")
            return 'unknown'
    
    def cleanup(self):
        """Clean up MediaPipe resources"""
        try:
            if hasattr(self, 'segment'):
                self.segment.close()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

# Global color detector instance (lazy initialization)
_color_detector_instance = None

def get_color_detector():
    """Get or create the color detector instance"""
    global _color_detector_instance
    if _color_detector_instance is None:
        _color_detector_instance = ColorDetector()
    return _color_detector_instance

def detect_image_colors(image_path: str, max_colors: int = 3) -> List[Dict]:
    """Convenience function for color detection"""
    return get_color_detector().detect_clothing_colors(image_path, max_colors)

def get_primary_color_name(image_path: str) -> str:
    """Convenience function to get primary color name"""
    return get_color_detector().get_primary_color(image_path)