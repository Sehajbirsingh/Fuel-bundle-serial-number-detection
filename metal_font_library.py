import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
import string

class MetalFont:
    """
    Professional Metal Font Generator
    Creates realistic engraved metal text with transparent backgrounds
    Perfect for fuel bundle serial numbers and industrial applications
    """
    
    def __init__(self):
        self.colors = {
            'metal_base': (120, 120, 115),
            'deep_shadow': (45, 45, 40),
            'highlight': (160, 160, 155),
            'bright_highlight': (200, 200, 195),
            'dark_edge': (30, 30, 25),
        }
    
    def _get_font(self, size=60):
        """Get system font"""
        fonts = ["C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/arial.ttf"]
        for font_path in fonts:
            if os.path.exists(font_path):
                try:
                    return ImageFont.truetype(font_path, size)
                except:
                    continue
        return ImageFont.load_default()
    
    def _create_mask(self, text, font, width, height):
        """Create high-quality text mask"""
        scale = 4
        high_res = Image.new('L', (width * scale, height * scale), 0)
        draw = ImageDraw.Draw(high_res)
        scaled_font = self._get_font(font.size * scale)
        
        bbox = draw.textbbox((0, 0), text, font=scaled_font)
        text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = (width * scale - text_width) // 2
        y = (height * scale - text_height) // 2
        
        draw.text((x, y), text, font=scaled_font, fill=255)
        return high_res.resize((width, height), Image.Resampling.LANCZOS)
    
    def _apply_lighting(self, base, mask):
        """Apply metal engraving lighting"""
        result = np.array(base)
        mask_array = np.array(mask).astype(float) / 255.0
        height, width = mask_array.shape
        
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if mask_array[y, x] > 0.1:
                    # Calculate gradients for lighting
                    grad_x = mask_array[y, x+1] - mask_array[y, x-1]
                    grad_y = mask_array[y+1, x] - mask_array[y-1, x]
                    
                    # Light intensity (from top-left)
                    light = max(0, -grad_x - grad_y + 0.5)
                    
                    # Apply engraved effect
                    if light > 0.7:
                        color = self.colors['bright_highlight']
                    elif light > 0.3:
                        color = self.colors['highlight']
                    else:
                        color = self.colors['deep_shadow']
                    
                    alpha = mask_array[y, x]
                    result[y, x] = result[y, x] * (1 - alpha) + np.array(color) * alpha
        
        return Image.fromarray(result.astype(np.uint8))
    
    def generate_text(self, text, width=400, height=100, font_size=50):
        """
        Generate metal engraved text with transparent background
        
        Args:
            text (str): Text to engrave
            width (int): Image width in pixels
            height (int): Image height in pixels  
            font_size (int): Font size
            
        Returns:
            PIL.Image: RGBA image with transparent background
        """
        # Create temporary base
        temp_base = Image.new('RGB', (width, height), self.colors['metal_base'])
        font = self._get_font(font_size)
        mask = self._create_mask(text, font, width, height)
        
        # Apply lighting
        lit_image = self._apply_lighting(temp_base, mask)
        
        # Convert to RGBA with transparency
        final = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        final_array = np.array(final)
        lit_array = np.array(lit_image)
        mask_array = np.array(mask).astype(float) / 255.0
        
        for y in range(height):
            for x in range(width):
                if mask_array[y, x] > 0.1:
                    alpha = int(mask_array[y, x] * 255)
                    final_array[y, x] = [*lit_array[y, x], alpha]
        
        return Image.fromarray(final_array)
    
    def generate_serial(self, pattern="AAAnnnA"):
        """
        Generate random serial number following pattern
        
        Args:
            pattern (str): Pattern where 'A' = letter, 'n' = number
            
        Returns:
            str: Generated serial number
        """
        result = ""
        for char in pattern:
            if char == 'A':
                result += random.choice(string.ascii_uppercase)
            elif char == 'n':
                result += str(random.randint(0, 9))
            else:
                result += char
        return result
    
    def save_text(self, text, filename, width=400, height=100, font_size=50):
        """
        Generate and save metal text to file
        
        Args:
            text (str): Text to engrave
            filename (str): Output filename (with .png extension)
            width (int): Image width
            height (int): Image height
            font_size (int): Font size
        """
        image = self.generate_text(text, width, height, font_size)
        image.save(filename)
        return image 