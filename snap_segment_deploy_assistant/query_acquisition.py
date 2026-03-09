"""Query Acquisition logic equivalent to the original implementation."""

import cv2


def calculate_iou(bbox1, bbox2):
    """Compute IoU for two boxes represented by x/y/width/height dictionaries."""
    x1, y1, w1, h1 = bbox1["x"], bbox1["y"], bbox1["width"], bbox1["height"]
    x2, y2, w2, h2 = bbox2["x"], bbox2["y"], bbox2["width"], bbox2["height"]

    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1 + w1, x2 + w2)
    inter_y2 = min(y1 + h1, y2 + h2)

    if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area1 = w1 * h1
    area2 = w2 * h2

    return inter_area / float(area1 + area2 - inter_area)


def merge_detections(all_detections, iou_threshold, min_votes):
    """Merge detections from multiple frames using confidence-weighted averaging."""
    merged_results = []
    for frame_detections in all_detections:
        for det in frame_detections:
            label = det["label"]
            bbox = det["bbox"]
            conf = det["confidence"]

            matched = False
            for merged in merged_results:
                if merged["label"] != label:
                    continue
                iou = calculate_iou(merged["bbox"], bbox)
                if iou > iou_threshold:
                    old_conf = merged["confidence"]
                    new_conf = conf

                    merged["bbox"]["x"] = (merged["bbox"]["x"] * old_conf + bbox["x"] * new_conf) / (old_conf + new_conf)
                    merged["bbox"]["y"] = (merged["bbox"]["y"] * old_conf + bbox["y"] * new_conf) / (old_conf + new_conf)
                    merged["bbox"]["width"] = (
                        merged["bbox"]["width"] * old_conf + bbox["width"] * new_conf
                    ) / (old_conf + new_conf)
                    merged["bbox"]["height"] = (
                        merged["bbox"]["height"] * old_conf + bbox["height"] * new_conf
                    ) / (old_conf + new_conf)
                    merged["confidence"] = max(old_conf, new_conf)
                    merged["votes"] += 1
                    matched = True
                    break

            if not matched:
                merged_results.append(
                    {
                        "label": label,
                        "bbox": bbox,
                        "confidence": conf,
                        "votes": 1,
                    }
                )
    return [det for det in merged_results if det["votes"] >= min_votes]


def suppress_duplicate_boxes(detections, iou_thresh=0.4):
    """Filter duplicate boxes for the same label based on IoU."""
    filtered = []
    detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)
    while detections:
        best = detections.pop(0)
        filtered.append(best)
        detections = [
            d
            for d in detections
            if d["label"] != best["label"] or calculate_iou(d["bbox"], best["bbox"]) < iou_thresh
        ]
    return filtered


def equalize_frame(frame):
    """Perform CLAHE equalization on the L channel of the LAB image."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    frame_equalized = cv2.cvtColor(limg, cv2.COLOR_Lab2BGR)
    return frame_equalized

