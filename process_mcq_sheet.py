# process_mcq_sheet.py
import cv2
import os
import re
import numpy as np
from ultralytics import YOLO

# === CONFIGURATION ===
MCQ_MODEL_PATH = 'models/yolov8_bubbles_best.pt'
REG_MODEL_PATH = 'models/reg_number_model_v1.pt'

CONF_THRESHOLD_MCQ = 0.5   # ↑ bump threshold for bubble detection
CONF_THRESHOLD_REG = 0.25
EXPECTED_REG_LENGTH = 9
REGEX_PATTERN = r'^U\d{2}[A-Z]{2}\d{4}$'

# === ANSWER KEY ===
CORRECT_ANSWERS = {
    1: "A", 2: "C", 3: "A", 4: "B", 5: "A",
    6: "D", 7: "A", 8: "C", 9: "A", 10: "A",
    11: "A", 12: "A", 13: "C", 14: "A", 15: "A",
    16: "A", 17: "A", 18: "C", 19: "A", 20: "A",
    21: "A", 22: "A", 23: "C", 24: "A", 25: "A",
    26: "A", 27: "A", 28: "C", 29: "B", 30: "A"
}

CLASS_NAMES = ['A', 'B', 'C', 'D', 'E', 'INVALID']

# === UTILS ===
def correct_character(c):
    subs = {'0': 'O', 'O': 'O', '1': '1', 'I': '1', 'L': '1', 'C ': 'C', ' ': ''}
    return subs.get(c, c)

# === STEP 1: ALIGN SHEET USING 4 CORNER MARKERS ===
def find_markers_and_align(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h_img, w_img = image.shape[:2]

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 500 < area < 20000:
            rect = cv2.minAreaRect(cnt)
            w, h = rect[1]
            if w == 0 or h == 0:
                continue
            aspect_ratio = max(w, h) / min(w, h)
            fill_ratio = area / (w * h)
            if aspect_ratio < 1.2 and fill_ratio > 0.85:
                cx, cy = np.int0(rect[0])
                candidates.append((cx, cy, cnt))

    if len(candidates) < 4:
        raise ValueError("Marker detection failed: insufficient markers found.")

    # Choose closest to each sheet corner
    corners = [(0, 0), (w_img, 0), (0, h_img), (w_img, h_img)]
    closest = [min(candidates, key=lambda c: np.hypot(c[0] - cx, c[1] - cy))[:2] for cx, cy in corners]

    src_pts = np.array(closest, dtype='float32')
    dst_pts = np.array([[0, 0], [2480, 0], [0, 3508], [2480, 3508]], dtype='float32')

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(image, M, (2480, 3508))

# === STEP 2: PREDICT BUBBLES ===
def predict_mcq(image):
    model = YOLO(MCQ_MODEL_PATH)
    results = model.predict(image, conf=CONF_THRESHOLD_MCQ)[0]
    return results.boxes.xyxy, results.boxes.conf, results.boxes.cls

# === STEP 3: DIVIDE QUESTIONS ===
def divide_question_box(box, num_questions=15):
    x, y, w, h = box
    row_height = h / num_questions
    return [(x, y + i * row_height, w, row_height) for i in range(num_questions)]

def map_bubbles_to_questions(boxes, confs, classes, q1_15_box, q16_30_box):
    q1_rows = divide_question_box(q1_15_box, 15)
    q2_rows = divide_question_box(q16_30_box, 15)

    question_map = {i: ["", 0.0] for i in range(1, 31)}

    for box, conf, cls in zip(boxes, confs, classes):
        x1, y1, x2, y2 = box.tolist()
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        option = CLASS_NAMES[int(cls)]

        # First 15
        for i, (x, y, w, h) in enumerate(q1_rows):
            if x <= cx <= x + w and y <= cy <= y + h and conf > question_map[i + 1][1]:
                question_map[i + 1] = [option, float(conf)]
                break
        # Next 15
        for i, (x, y, w, h) in enumerate(q2_rows):
            if x <= cx <= x + w and y <= cy <= y + h and conf > question_map[i + 16][1]:
                question_map[i + 16] = [option, float(conf)]
                break

    return question_map

# === STEP 4: GRADE ===
def grade_answers(question_map):
    score, results = 0, []
    for q_num, (marked, conf) in question_map.items():
        correct = CORRECT_ANSWERS.get(q_num, "?")
        if marked == correct:
            status, score = "Correct", score + 1
        elif marked == "INVALID":
            status = "Invalid Mark"
        elif marked:
            status = f"Wrong (marked {marked})"
        else:
            status = "Blank"

        results.append({
            "question": q_num,
            "marked": marked if marked else "-",
            "correct": correct,
            "status": status,
            "confidence": conf
        })
    return {"score": score, "total": len(CORRECT_ANSWERS), "details": results}

# === STEP 5: REG NUMBER ===
def extract_reg_number(image):
    model = YOLO(REG_MODEL_PATH)
    results = model.predict(image, conf=CONF_THRESHOLD_REG)[0]

    boxes = results.boxes.xyxy.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy()
    class_names = model.names

    detections = sorted(zip(boxes, classes, confs), key=lambda x: x[2], reverse=True)[:EXPECTED_REG_LENGTH]
    detections = sorted(detections, key=lambda x: x[0][0])

    raw_chars = [class_names[int(cls)].strip().upper() for _, cls, _ in detections]
    corrected_chars = [correct_character(c) for c in raw_chars]

    return ''.join(corrected_chars)

# === MAIN PIPELINE ===
def process_sheet(image_path: str):
    if not os.path.exists(image_path):
        raise FileNotFoundError("Image not found.")

    original = cv2.imread(image_path)
    aligned = find_markers_and_align(original)

    reg_number = extract_reg_number(aligned)
    boxes, confs, classes = predict_mcq(aligned)

    # Fixed bounding zones for Q1–Q15 and Q16–Q30
    q1_15_box = (270, 760, 950, 1340)
    q16_30_box = (1300, 760, 950, 1340)

    question_map = map_bubbles_to_questions(boxes, confs, classes, q1_15_box, q16_30_box)
    grading = grade_answers(question_map)
    answers = [item["marked"] for item in grading["details"]]

    return {
        "reg_number": reg_number,
        "score": grading["score"],
        "total": grading["total"],
        "details": grading["details"],
        "answers": answers  # required for mobile app
    }
