import os
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import numpy as np
import random
import string

class RealisticMetalEngraver:
    def __init__(self):
        pass
    
    def generate_serial_number(self, format_pattern="AAAnnnA"):
        """Generate a random serial number following the specified pattern"""
        serial = ""
        for char in format_pattern:
            if char == 'A':
                serial += random.choice(string.ascii_uppercase)
            elif char == 'n':
                serial += str(random.randint(0, 9))
            else:
                serial += char
        return serial
    
    def get_system_font(self, size=50):
        """Get a bold system font"""
        font_paths = [
            "C:/Windows/Fonts/arialbd.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/calibrib.ttf",
            "C:/Windows/Fonts/tahoma.ttf",
        ]
        
        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    return ImageFont.truetype(font_path, size)
                except:
                    continue
        
        try:
            return ImageFont.truetype("arial.ttf", size)
        except:
            return ImageFont.load_default()
    
    def sample_surface_color(self, image, pos, box_size):
        """Sample the average color around the target area"""
        x, y = pos
        w, h = box_size
        
        # Sample area around the target region
        sample_area = image.crop((x-20, y-20, x+w+20, y+h+20))
        
        # Convert to numpy array and get average color
        arr = np.array(sample_area)
        avg_color = np.mean(arr, axis=(0, 1))
        
        return tuple(avg_color.astype(int)[:3])
    
    def create_engraved_effect(self, image, text, pos, box_size, font_size=40):
        """Create realistic engraved metal text effect"""
        x, y = pos
        w, h = box_size
        
        # Sample the metal surface color
        base_color = self.sample_surface_color(image, pos, box_size)
        print(f"Sampled surface color: {base_color}")
        
        # Create darker and lighter variations
        darker = tuple(max(0, int(c * 0.4)) for c in base_color)  # Much darker for engraved area
        shadow_dark = tuple(max(0, int(c * 0.2)) for c in base_color)  # Very dark shadow
        highlight = tuple(min(255, int(c * 1.3)) for c in base_color)  # Lighter highlight
        
        # Get font
        font = self.get_system_font(font_size)
        
        # Create text mask
        text_mask = Image.new('L', (w, h), 0)
        mask_draw = ImageDraw.Draw(text_mask)
        
        # Calculate text position (centered)
        bbox = mask_draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        text_x = (w - text_width) // 2
        text_y = (h - text_height) // 2
        
        # Draw text mask
        mask_draw.text((text_x, text_y), text, font=font, fill=255)
        
        # Create the engraved effect
        result_image = image.copy()
        
        # Extract the region we're working on
        region = result_image.crop((x, y, x + w, y + h))
        region_array = np.array(region)
        mask_array = np.array(text_mask)
        
        # Create engraved effect
        for i in range(h):
            for j in range(w):
                if mask_array[i, j] > 0:  # Inside text
                    # Make it darker (engraved/recessed)
                    alpha = mask_array[i, j] / 255.0
                    current_color = region_array[i, j]
                    
                    # Blend with darker color
                    new_color = []
                    for k in range(3):
                        new_val = current_color[k] * (1 - alpha) + darker[k] * alpha
                        new_color.append(int(new_val))
                    
                    region_array[i, j] = new_color
        
        # Add shadow effect (offset shadow inside the text)
        shadow_mask = text_mask.filter(ImageFilter.GaussianBlur(1))
        shadow_array = np.array(shadow_mask)
        
        # Apply shadow with offset
        for i in range(1, h-1):
            for j in range(1, w-1):
                if mask_array[i, j] > 0:  # Inside text
                    # Add shadow from top-left
                    shadow_strength = shadow_array[i-1, j-1] / 255.0 * 0.6
                    if shadow_strength > 0:
                        current_color = region_array[i, j]
                        for k in range(3):
                            new_val = current_color[k] * (1 - shadow_strength) + shadow_dark[k] * shadow_strength
                            region_array[i, j][k] = int(new_val)
        
        # Add subtle highlight on the bottom-right edge of text
        for i in range(h-1):
            for j in range(w-1):
                if (mask_array[i, j] > 0 and 
                    (mask_array[i+1, j] == 0 or mask_array[i, j+1] == 0)):  # Edge of text
                    
                    highlight_strength = 0.3
                    current_color = region_array[i, j]
                    for k in range(3):
                        new_val = current_color[k] * (1 - highlight_strength) + highlight[k] * highlight_strength
                        region_array[i, j][k] = min(255, int(new_val))
        
        # Convert back to image and paste back
        modified_region = Image.fromarray(region_array.astype(np.uint8))
        result_image.paste(modified_region, (x, y))
        
        return result_image
    
    def apply_engraved_serial(self, image_path, serial_number=None, pos=(720, 1790), box_size=(265, 98), font_size=40):
        """Apply realistic engraved serial number to image"""
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Generate serial number if not provided
        if serial_number is None:
            serial_number = self.generate_serial_number()
        
        print(f"Engraving serial number: {serial_number}")
        
        # Apply engraved effect
        result_image = self.create_engraved_effect(
            image, serial_number, pos, box_size, font_size
        )
        
        return result_image, serial_number

def main():
    """Main function"""
    engraver = RealisticMetalEngraver()
    
    # Parameters
    image_path = "RAT074G.png"
    output_path = "RAT074G_engraved.png"
    position = (720, 1790)
    box_size = (265, 98)
    
    # You can specify a custom serial number or let it generate one
    custom_serial = None  # Will generate AAAnnnA format
    # custom_serial = "RAT074G"  # Uncomment to use specific serial
    
    try:
        # Apply engraved effect
        result_image, used_serial = engraver.apply_engraved_serial(
            image_path,
            serial_number=custom_serial,
            pos=position,
            box_size=box_size,
            font_size=38
        )
        
        # Save result
        result_image.save(output_path, quality=95)
        
        print(f"✓ Realistic engraved effect applied!")
        print(f"✓ Serial number: {used_serial}")
        print(f"✓ Output saved as: {output_path}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
