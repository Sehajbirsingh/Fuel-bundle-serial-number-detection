import cv2
import numpy as np
import random
import string
import os

# === CONFIG ===
base_image_path = "RAT074G.png"  # reference image
output_dir = "synthetic_dataset"
os.makedirs(output_dir, exist_ok=True)

# Serial format: AAAnnnA
def generate_serial():
    return ''.join(random.choices(string.ascii_uppercase, k=3) +
                   random.choices(string.digits, k=3) +
                   random.choices(string.ascii_uppercase, k=1))

# # Overlay serial number at detected position
# def overlay_serial(img, serial_text, pos=(720, 1765), box_size=(265, 98)):
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     font_scale = 2.1  # Adjust this to fit box height
#     thickness = 4
#     color = (30, 30, 30)  # dark gray / black
#     text_size = cv2.getTextSize(serial_text, font, font_scale, thickness)[0]
    
#     x = pos[0] + (box_size[0] - text_size[0]) // 2
#     y = pos[1] + (box_size[1] + text_size[1]) // 2
    
#     cv2.putText(img, serial_text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
#     return img

def overlay_serial(img, serial_text, pos=(720, 1790), box_size=(265, 98)):
    font = cv2.FONT_HERSHEY_COMPLEX
    font_scale = 1.7
    thickness = 3 # Increased thickness for a more substantial look
    
    # Colors adjusted for a brownish-gray feel (BGR format)
    main_text_color_bgr = (80, 90, 100)    # Medium-dark grayish-brown
    highlight_color_bgr = (100, 110, 120)  # Lighter version for highlight
    shadow_color_bgr = (60, 70, 80)        # Darker version for shadow
    
    # Increased offsets for a more pronounced 3D effect
    highlight_offset = (-2, -2) # dx, dy
    shadow_offset = (2, 2)    # dx, dy

    char_spacing_reduction = 2
    angle = -6 # As per your last update for alignment

    actual_text_width = 0
    max_ascent = 0
    max_descent = 0
    char_details_for_drawing = []

    for i, char_code in enumerate(serial_text):
        (char_w, char_h_above_baseline), baseline_offset = cv2.getTextSize(char_code, font, font_scale, thickness)
        current_advance = char_w
        if i < len(serial_text) - 1:
            current_advance -= char_spacing_reduction
        actual_text_width += current_advance
        if char_h_above_baseline > max_ascent:
            max_ascent = char_h_above_baseline
        if baseline_offset > max_descent:
            max_descent = baseline_offset
        char_details_for_drawing.append({
            'char': char_code, 'adv_w': current_advance, 'draw_w': char_w,
            'h_above_bl': char_h_above_baseline, 'bl_offset': baseline_offset
        })

    text_render_height_unrotated = max_ascent + max_descent
    text_render_width_unrotated = actual_text_width
    
    canvas_padding = 15 + max(abs(highlight_offset[0]), abs(shadow_offset[0]), abs(highlight_offset[1]), abs(shadow_offset[1]))
    drawing_baseline_y = canvas_padding + max_ascent
    
    unrotated_canvas_w = text_render_width_unrotated + 2 * canvas_padding
    unrotated_canvas_h = text_render_height_unrotated + 2 * canvas_padding
    
    text_canvas_unrotated = np.zeros((unrotated_canvas_h, unrotated_canvas_w, 4), dtype=np.uint8)

    current_x_on_canvas = canvas_padding
    for detail in char_details_for_drawing:
        cv2.putText(text_canvas_unrotated, detail['char'], 
                    (int(round(current_x_on_canvas + shadow_offset[0])), int(round(drawing_baseline_y + shadow_offset[1]))), 
                    font, font_scale, (*shadow_color_bgr, 255), thickness, cv2.LINE_AA)
        current_x_on_canvas += detail['adv_w']

    current_x_on_canvas = canvas_padding
    for detail in char_details_for_drawing:
        cv2.putText(text_canvas_unrotated, detail['char'], 
                    (int(round(current_x_on_canvas + highlight_offset[0])), int(round(drawing_baseline_y + highlight_offset[1]))), 
                    font, font_scale, (*highlight_color_bgr, 255), thickness, cv2.LINE_AA)
        current_x_on_canvas += detail['adv_w']

    current_x_on_canvas = canvas_padding
    for detail in char_details_for_drawing:
        cv2.putText(text_canvas_unrotated, detail['char'], 
                    (int(round(current_x_on_canvas)), int(round(drawing_baseline_y))), 
                    font, font_scale, (*main_text_color_bgr, 255), thickness, cv2.LINE_AA)
        current_x_on_canvas += detail['adv_w']

    (h_uc, w_uc) = text_canvas_unrotated.shape[:2]
    center_uc = (w_uc // 2, h_uc // 2)
    rot_mat = cv2.getRotationMatrix2D(center_uc, angle, 1.0)

    cos_abs = np.abs(rot_mat[0, 0])
    sin_abs = np.abs(rot_mat[0, 1])
    rotated_canvas_w = int((h_uc * sin_abs) + (w_uc * cos_abs))
    rotated_canvas_h = int((h_uc * cos_abs) + (w_uc * sin_abs))

    rot_mat[0, 2] += (rotated_canvas_w / 2) - center_uc[0]
    rot_mat[1, 2] += (rotated_canvas_h / 2) - center_uc[1]

    rotated_text_canvas = cv2.warpAffine(text_canvas_unrotated, rot_mat, (rotated_canvas_w, rotated_canvas_h), 
                                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))

    target_box_center_x = pos[0] + box_size[0] // 2
    target_box_center_y = pos[1] + box_size[1] // 2

    paste_origin_x = int(round(target_box_center_x - rotated_canvas_w / 2))
    paste_origin_y = int(round(target_box_center_y - rotated_canvas_h / 2))
    
    img_h, img_w = img.shape[:2]

    roi_y_start = paste_origin_y
    roi_y_end = paste_origin_y + rotated_canvas_h
    roi_x_start = paste_origin_x
    roi_x_end = paste_origin_x + rotated_canvas_w

    text_src_y_start, text_src_y_end = 0, rotated_canvas_h
    text_src_x_start, text_src_x_end = 0, rotated_canvas_w

    if roi_y_start < 0:
        text_src_y_start -= roi_y_start
        roi_y_start = 0
    if roi_y_end > img_h:
        text_src_y_end -= (roi_y_end - img_h)
        roi_y_end = img_h
    if roi_x_start < 0:
        text_src_x_start -= roi_x_start
        roi_x_start = 0
    if roi_x_end > img_w:
        text_src_x_end -= (roi_x_end - img_w)
        roi_x_end = img_w

    if not (roi_y_end > roi_y_start and roi_x_end > roi_x_start and \
            text_src_y_end > text_src_y_start and text_src_x_end > text_src_x_start):
        return img

    img_roi = img[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
    text_to_blend = rotated_text_canvas[text_src_y_start:text_src_y_end, text_src_x_start:text_src_x_end]
    
    if img_roi.shape[0] != text_to_blend.shape[0] or img_roi.shape[1] != text_to_blend.shape[1]:
         return img

    alpha_mask_float = text_to_blend[:, :, 3].astype(np.float32) / 255.0
    bgr_text_pixels_float = text_to_blend[:, :, :3].astype(np.float32)

    alpha_mask_3ch = cv2.cvtColor(alpha_mask_float, cv2.COLOR_GRAY2BGR)

    blended_roi = bgr_text_pixels_float * alpha_mask_3ch + img_roi.astype(np.float32) * (1.0 - alpha_mask_3ch)
    
    img[roi_y_start:roi_y_end, roi_x_start:roi_x_end] = blended_roi.astype(np.uint8)

    return img

# Generate N samples
def generate_images(n=1):
    base = cv2.imread(base_image_path)
    for i in range(n):
        serial = generate_serial()
        img = base.copy()
        img_with_text = overlay_serial(img, serial)
        cv2.imwrite(f"{output_dir}/{serial}.jpg", img_with_text)

generate_images(n=500)
