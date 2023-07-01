import os
import cv2

def check_file_exists(path:str):
    if not os.path.exists(path) or not os.path.isfile(path):
        return False
    return True

def check_dir_exists(path:str):
    if not os.path.exists(path) or not os.path.isdir(path):
        return False
    return True

def draw_boxes(img, box, label, _ind_to_class):
    """
    Draw bounding boxes on the image

    Args:
        img: image to draw on
        box: bounding box
        label: class label
        _ind_to_class: dictionary of class labels

    Returns:
        img: image with bounding boxes drawn on
    """
    cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
    text =  f"{_ind_to_class[int(label)]}"
    coord = (int(box[0])+3, int(box[1])+7+10)
    cv2.putText(img, text, coord, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return img
	
def set_text(img, text, text_pos):
    """
    Draw text on the image

    Args:
        img: image to draw on
        text: text to draw
        text_pos: position of text

    Returns:
        img: image with text drawn on
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    lineThickness = 1
    color = (0,0,0)
    font_size = 0.5
    (text_width, text_height) = cv2.getTextSize(text, font, font_size, lineThickness)[0]
    text_offset_x,text_offset_y = text_pos
    box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height - 2))
    cv2.rectangle(img, box_coords[0], box_coords[1], color, cv2.FILLED)
    cv2.putText(img, text, text_pos, font, font_size, (255,255,255), lineThickness, cv2.LINE_AA)
    return img


def clip_to_frames(clip_path, output_path, frames_rate=16):
    """ video 1minute -> 60 frame i.e 1 frame per second """
    vidcap = cv2.VideoCapture(clip_path)
    frame_count = 0
    success, frame = vidcap.read()
    success = True
    while success:
        cv2.imwrite(os.path.join(output_path, f"{frame_count}.jpg"), frame)
        vidcap.set(cv2.CAP_PROP_POS_MSEC,(frame_count*1000))    # added this line 
        frame_count = frame_count + 1
        success, frame = vidcap.read()
    vidcap.release()
    cv2.destroyAllWindows()
    



