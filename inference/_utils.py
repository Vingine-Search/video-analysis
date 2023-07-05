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


# start_time, end_time in seconds
def clip_to_frames(clip_path, output_path, fps=1, start_time=None, end_time=None):
    # TODO: check if start_time and end_time are "valid"
    """ video 1minute -> 60 frame i.e 1 frame per second """
    vidcap = cv2.VideoCapture(clip_path)
    frame_count = 0
    if start_time:
        vidcap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
        frame_count = start_time
    success, frame = vidcap.read()
    while success:
        # frame = cv2.resize(frame, (224, 224))
        if end_time is not None and frame_count >= end_time:
            break
        # this is for s3d
        cv2.imwrite(os.path.join(output_path, f"{frame_count}.jpg"), cv2.resize(frame, (224, 224)))
        # this is for easyocr & fasterrcnn
        frame = cv2.resize(frame, (300, int(300 * frame.shape[0] / frame.shape[1])))
        cv2.imwrite(os.path.join(output_path, "unknown", f"{frame_count}.jpg"), frame)
        frame_count = frame_count + 1/fps
        vidcap.set(cv2.CAP_PROP_POS_MSEC,(frame_count*1000))
        success, frame = vidcap.read()
    vidcap.release()


def sort_helper(path):
    file_name = os.path.basename(path)
    noext, _ = os.path.splitext(file_name)
    return float(noext)
