import cv2
import numpy as np
from scipy import ndimage
import logging

try:
    from skimage import exposure, segmentation, color
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    logging.warning("skimage not available. Some advanced image processing features will be limited.")

class ColposcopyPreprocessor:
    """Medical-specific preprocessing for colposcopy images"""
    
    def __init__(self):
        self.roi_detector = None  # Could be initialized with a pre-trained ROI model
        
    def preprocess_image(self, image):
        """Complete preprocessing pipeline for colposcopy images"""
        try:
            # Ensure image is in uint8 format
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            
            # Color normalization
            normalized = self._normalize_color(image)
            
            # Enhance acetowhite regions
            enhanced = self._enhance_acetowhite(normalized)
            
            # Detect cervical region
            roi_mask = self._detect_cervical_region(enhanced)
            
            # Apply ROI mask
            processed = self._apply_roi(enhanced, roi_mask)
            
            return processed, roi_mask
            
        except Exception as e:
            logging.error(f"Error in preprocessing image: {str(e)}")
            # Return original image and full mask if processing fails
            return image, np.ones_like(image[:,:,0])
    
    def _normalize_color(self, image):
        """Normalize color for consistent analysis"""
        try:
            if SKIMAGE_AVAILABLE:
                # Use skimage for advanced color normalization
                lab = color.rgb2lab(image)
                lab_normalized = exposure.rescale_intensity(lab)
                return (color.lab2rgb(lab_normalized) * 255).astype(np.uint8)
            else:
                # Fallback to simple normalization using cv2
                lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                l_norm = cv2.normalize(l, None, 0, 255, cv2.NORM_MINMAX)
                normalized = cv2.merge([l_norm, a, b])
                return cv2.cvtColor(normalized, cv2.COLOR_LAB2RGB)
        except Exception as e:
            logging.error(f"Error in color normalization: {str(e)}")
            return image
    
    def _enhance_acetowhite(self, image):
        """Enhance acetowhite regions for better visibility"""
        try:
            # Ensure image is in uint8 format
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Verify gray image is in correct format
            if gray.dtype != np.uint8:
                gray = gray.astype(np.uint8)
            
            # Apply CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Create enhanced color image
            enhanced_color = image.copy()
            
            # Enhance white regions
            _, white_mask = cv2.threshold(enhanced, 170, 255, cv2.THRESH_BINARY)
            
            # Apply enhancement only to white regions
            white_mask_3d = np.stack([white_mask] * 3, axis=-1) > 0
            enhanced_color[white_mask_3d] = np.minimum(
                enhanced_color[white_mask_3d] * 1.2,
                255
            ).astype(np.uint8)
            
            return enhanced_color
            
        except Exception as e:
            logging.error(f"Error in acetowhite enhancement: {str(e)}")
            return image
    
    def _detect_cervical_region(self, image):
        """Detect the cervical region in the image"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Apply Otsu's thresholding
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Morphological operations
            kernel = np.ones((5,5), np.uint8)
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return np.ones_like(gray)
                
            # Find largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Create mask
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [largest_contour], -1, 255, -1)
            
            return mask
            
        except Exception as e:
            logging.error(f"Error in cervical region detection: {str(e)}")
            return np.ones_like(image[:,:,0])
    
    def _apply_roi(self, image, mask):
        """Apply ROI mask to the image"""
        try:
            # Ensure mask is binary and same size as image
            mask_3d = np.stack([mask] * 3, axis=-1) > 0
            
            # Apply mask
            masked_image = image.copy()
            masked_image[~mask_3d] = 0
            
            return masked_image
            
        except Exception as e:
            logging.error(f"Error applying ROI: {str(e)}")
            return image
    
    def detect_lesions(self, image):
        """Detect potential lesion regions"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours based on size and shape
        lesions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100 and area < 10000:  # Size thresholds
                x, y, w, h = cv2.boundingRect(contour)
                lesions.append({
                    'bbox': (x, y, w, h),
                    'contour': contour,
                    'area': area
                })
        
        return lesions

class LesionAnalyzer:
    """Analyze detected lesions in colposcopy images"""
    
    def analyze_lesion(self, image, lesion):
        """Analyze characteristics of a detected lesion"""
        x, y, w, h = lesion['bbox']
        roi = image[y:y+h, x:x+w]
        
        features = {
            'size': lesion['area'],
            'color_features': self._analyze_color(roi),
            'texture_features': self._analyze_texture(roi),
            'shape_features': self._analyze_shape(lesion['contour']),
            'acetowhite_response': self._analyze_acetowhite(roi)
        }
        
        return features
    
    def _analyze_color(self, roi):
        """Analyze color characteristics of the lesion"""
        # Convert to different color spaces
        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(roi, cv2.COLOR_RGB2LAB)
        
        return {
            'mean_color': roi.mean(axis=(0,1)).tolist(),
            'std_color': roi.std(axis=(0,1)).tolist(),
            'hsv_features': hsv.mean(axis=(0,1)).tolist(),
            'lab_features': lab.mean(axis=(0,1)).tolist()
        }
    
    def _analyze_texture(self, roi):
        """Analyze texture patterns in the lesion"""
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        
        # GLCM features
        glcm = exposure.greycomatrix(gray, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4])
        contrast = exposure.greycoprops(glcm, 'contrast')
        dissimilarity = exposure.greycoprops(glcm, 'dissimilarity')
        homogeneity = exposure.greycoprops(glcm, 'homogeneity')
        
        return {
            'contrast': float(contrast.mean()),
            'dissimilarity': float(dissimilarity.mean()),
            'homogeneity': float(homogeneity.mean())
        }
    
    def _analyze_shape(self, contour):
        """Analyze shape characteristics of the lesion"""
        # Calculate shape metrics
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        return {
            'area': float(area),
            'perimeter': float(perimeter),
            'circularity': float(circularity)
        }
    
    def _analyze_acetowhite(self, roi):
        """Analyze acetowhite response in the lesion"""
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        
        # Calculate intensity metrics
        mean_intensity = gray.mean()
        std_intensity = gray.std()
        
        # Calculate white pixel ratio
        white_pixels = np.sum(gray > 170)  # Threshold for white
        total_pixels = gray.size
        white_ratio = white_pixels / total_pixels if total_pixels > 0 else 0
        
        return {
            'mean_intensity': float(mean_intensity),
            'std_intensity': float(std_intensity),
            'white_ratio': float(white_ratio)
        } 