from pathlib import Path
import os
import json
import random
import math
from typing import Tuple, Optional, Dict, Any, List

# Third-party
import cv2
import numpy as np
from PIL import Image

# ==========================
# TUNABLE CONSTANTS (để dễ điều chỉnh)
# ==========================
PADDING = 16                      # padding xung quanh bounding polygon khi crop
BRIGHTNESS_TARGET = 120          # mục tiêu mean V channel sau tăng sáng
MIN_BRIGHTNESS_THRESHOLD = 80    # nếu mean V < threshold thì tăng sáng
DENOISE_METHOD = "nl_means"      # 'nl_means' hoặc 'bilateral'
DENOISE_H = 10                   # tham số h cho nl_means
DENOISE_TEMPLATE_WIN = 7
DENOISE_SEARCH_WIN = 21

DEBUG = True

def dbg(msg):
    if DEBUG:
        print(msg)

# ==========================
# IO HELPERS (windows-safe)
# ==========================
def _read_image_cv2(path: Path) -> np.ndarray:
    """
    Đọc ảnh từ đường dẫn, xử lý unicode path trên Windows.
    Trả về image BGR (numpy ndarray).
    """
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        # fallback to PIL
        with Image.open(str(path)) as im:
            im = im.convert("RGB")
            img = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    return img


def _save_image_cv2(path: Path, img_bgr: np.ndarray) -> None:
    """
    Lưu ảnh sử dụng imencode -> tofile để tránh lỗi với unicode paths trên Windows.
    """
    ext = os.path.splitext(str(path))[1] or ".jpg"
    ok, enc = cv2.imencode(ext, img_bgr)
    if not ok:
        raise RuntimeError(f"Failed to encode image for saving: {path}")
    enc.tofile(str(path))


# ==========================
# IMAGE CLEANING FUNCTIONS
# ==========================
def crop_leaf_polygon(
        image_bgr: np.ndarray,
        padding: int = 16):

    img = image_bgr.copy()
    h, w = img.shape[:2]

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Ngưỡng màu xanh tối ưu hơn cho lá lúa
    mask_green = cv2.inRange(
        hsv,
        np.array([25, 30, 20]),
        np.array([95, 255, 255])
    )

    mask_brown = cv2.inRange(
        hsv,
        np.array([5, 20, 20]),
        np.array([25, 255, 255])
    )

    mask_yellow = cv2.inRange(
        hsv,
        np.array([15, 20, 40]),
        np.array([40, 255, 255])
    )

    mask = cv2.bitwise_or(mask_green, mask_brown)
    mask = cv2.bitwise_or(mask, mask_yellow)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    min_area = max(
        200,
        int(0.001 * w * h)
    )
    clean = np.zeros_like(mask)

    for i in range(1, num_labels):

        area = stats[i, cv2.CC_STAT_AREA]

        if area > min_area:
            clean[labels == i] = 255

    mask = clean

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (9, 9)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=4
    )

    mask = cv2.dilate(
        mask,
        np.ones((5,5), np.uint8),
        iterations=1
    )
    
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    dbg("\n========== DEBUG ==========")
    dbg(f"Image size: {w}x{h}")
    dbg(f"Contours found: {len(contours)}")

    if len(contours) == 0:

        dbg("NO CONTOURS")

        return (
            img,
            np.ones((h, w), dtype=np.uint8) * 255,
            (0, 0, w, h),
            None
        )

    valid_contours = []

    for idx, c in enumerate(contours):

        area = cv2.contourArea(c)

        min_area = max(
            200,
            int(0.001 * w * h)
        )

        if area > min_area:

            x, y, bw, bh = cv2.boundingRect(c)

            ratio = max(bw, bh) / (min(bw, bh) + 1e-6)

            dbg(
                f"[{idx}] "
                f"area={area:.0f} "
                f"bbox=({bw}x{bh}) "
                f"ratio={ratio:.2f}"
            )

            valid_contours.append(c)

    dbg(f"Valid contours: {len(valid_contours)}")

    if len(valid_contours) == 0:

        dbg("NO VALID CONTOURS")

        return (
            img,
            np.ones((h, w), dtype=np.uint8) * 255,
            (0, 0, w, h),
            None
        )

    largest = max(
        valid_contours,
        key=cv2.contourArea
    )

    largest_area = cv2.contourArea(largest)

    filtered = []

    for c in valid_contours:

        area = cv2.contourArea(c)

        if area > largest_area * 0.25:
            filtered.append(c)

    dbg(f"Largest area: {largest_area:.0f}")
    dbg(f"Area ratio: {largest_area/(w*h):.4f}")

    x, y, bw, bh = cv2.boundingRect(largest)

    largest_ratio = max(bw, bh) / (min(bw, bh) + 1e-6)

    dbg(
        f"Largest contour bbox="
        f"({bw}x{bh}) "
        f"ratio={largest_ratio:.2f}"
    )

    # DEBUG: xem nếu dùng largest contour thì sao
    rect_largest = cv2.minAreaRect(largest)

    rw, rh = rect_largest[1]

    if min(rw, rh) > 0:
        ratio_largest = max(rw, rh) / min(rw, rh)
    else:
        ratio_largest = 0

    dbg(
        f"Largest contour "
        f"minAreaRect ratio={ratio_largest:.2f}"
    )

    # DEBUG: xem nếu gộp toàn bộ contour thì sao
    all_points = np.vstack(filtered)

    rect_all = cv2.minAreaRect(all_points)

    rw, rh = rect_all[1]

    if min(rw, rh) > 0:
        ratio_all = max(rw, rh) / min(rw, rh)
    else:
        ratio_all = 0

    dbg(
        f"ALL contour "
        f"minAreaRect ratio={ratio_all:.2f}"
    )

    # so sánh trực tiếp
    dbg(
        f"COMPARE -> "
        f"largest={ratio_largest:.2f} "
        f"all={ratio_all:.2f}"
    )

    # tạm thời dùng largest contour
    all_points = np.vstack(valid_contours)

    dbg("===========================\n")
        
    hull = cv2.convexHull(all_points)
    pts = hull.reshape(-1,2).astype(np.float32)

    mean, eigenvectors = cv2.PCACompute(
        pts,
        mean=None
    )

    direction = eigenvectors[0]
    print(direction)
    hull_area = cv2.contourArea(hull)

    coverage = hull_area / (w * h)

    dbg(f"Hull area: {hull_area:.0f}")
    dbg(f"Coverage : {coverage:.4f}")
    if coverage > 0.50:

        dbg("Coverage too large -> fallback largest contour")

        largest = max(valid_contours, key=cv2.contourArea)

        hull = cv2.convexHull(largest)

        hull_area = cv2.contourArea(hull)

        coverage = hull_area / (w * h)

    print(f"Hull area: {hull_area:.0f}")
    print(f"Coverage: {coverage:.3f}")
    rect = cv2.minAreaRect(hull)

    box = cv2.boxPoints(rect)

    box = np.int32(box)

    xs = box[:, 0]
    ys = box[:, 1]

    x = max(0, int(xs.min()))
    y = max(0, int(ys.min()))

    bw = int(xs.max() - xs.min())
    bh = int(ys.max() - ys.min())
    # x, y, bw, bh = cv2.boundingRect(hull)
    print(
        f"Crop box: "
        f"x={x}, y={y}, "
        f"w={bw}, h={bh}"
    )

    x0 = max(0, x - padding)
    y0 = max(0, y - padding)

    x1 = min(w, x + bw + padding)
    y1 = min(h, y + bh + padding)

    crop_w = x1 - x0
    crop_h = y1 - y0

    crop_ratio = (
        crop_w * crop_h
    ) / (w * h)

    dbg(
        f"Crop box = "
        f"({crop_w}x{crop_h})"
    )

    dbg(
        f"Crop ratio = "
        f"{crop_ratio:.3f}"
    )

    if crop_ratio > 0.90:

        dbg(
            "WARNING: crop gần full image"
        )

    # Mask đúng
    mask_full = np.zeros(
        (h, w),
        dtype=np.uint8
    )

    cv2.drawContours(
        mask_full,
        [hull],
        -1,
        255,
        -1
    )

    cropped = img[y0:y1, x0:x1]

    mask_crop = mask_full[
        y0:y1,
        x0:x1
    ]
    if DEBUG:

        cv2.imwrite(
            "debug_mask.png",
            mask
        )

        cv2.imwrite(
            "debug_mask_crop.png",
            mask_crop
        )

        overlay = img.copy()

        cv2.drawContours(
            overlay,
            [hull],
            -1,
            (0,0,255),
            4
        )

        cv2.imwrite(
            "debug_hull.png",
            overlay
        )
    return (
        cropped,
        mask_crop,
        (x0, y0, x1 - x0, y1 - y0),
        hull
    )


def adjust_brightness(image_bgr: np.ndarray,
                      target_mean: int = BRIGHTNESS_TARGET,
                      min_mean_threshold: int = MIN_BRIGHTNESS_THRESHOLD) -> Tuple[np.ndarray, bool, float]:
    """
    Nếu độ sáng (mean V channel trong HSV) < min_mean_threshold -> scale V để đạt target_mean.
    Trả về (img_bgr, changed_flag, mean_before).
    """
    try:
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        v = hsv[..., 2]
        mean_v = float(np.mean(v))
        if mean_v < min_mean_threshold and mean_v > 0:
            gain = float(target_mean) / (mean_v + 1e-6)
            hsv[..., 2] = np.clip(hsv[..., 2] * gain, 0, 255)
            out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            return out, True, mean_v
        return image_bgr, False, mean_v
    except Exception:
        return image_bgr, False, 0.0


def denoise_image(image_bgr: np.ndarray,
                  method: str = DENOISE_METHOD,
                  h: int = DENOISE_H,
                  templateWindowSize: int = DENOISE_TEMPLATE_WIN,
                  searchWindowSize: int = DENOISE_SEARCH_WIN) -> np.ndarray:
    """
    Giảm nhiễu: 'nl_means' (fastNlMeansDenoisingColored) hoặc 'bilateral'.
    """
    try:
        if method == 'bilateral':
            return cv2.bilateralFilter(image_bgr, d=9, sigmaColor=75, sigmaSpace=75)
        else:
            return cv2.fastNlMeansDenoisingColored(image_bgr, None, h, h, templateWindowSize, searchWindowSize)
    except Exception:
        return image_bgr


def clean_image_pipeline(image_path: Path,
                         save_path: Optional[Path] = None,
                         do_crop: bool = True,
                         do_bright: bool = True,
                         do_denoise: bool = True,
                         denoise_method: str = DENOISE_METHOD,
                         save_bbox: bool = True,
                         bbox_color: Tuple[int,int,int] = (0,255,0),
                         bbox_thickness: int = 4) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Pipeline đầy đủ: đọc -> crop leaf -> adjust brightness -> denoise -> save (nếu save_path).
    Nếu save_bbox=True thì vẽ bounding box (trên ảnh gốc) và lưu thành *_bbox<ext>.
    Trả về (processed_bgr, metadata).
    """
    original_img = _read_image_cv2(Path(image_path))  # keep original for bbox overlay
    img = original_img.copy()
    meta: Dict[str, Any] = {'original_shape': img.shape}

    if do_crop:
        cropped, mask, bbox, hull = crop_leaf_polygon(img)
        img = cropped
        meta['cropped_bbox'] = bbox
    else:
        mask = None
        bbox = (0,0,img.shape[1], img.shape[0])

    if do_bright:
        img, bright_changed, mean_before = adjust_brightness(img)
        meta['brightness_adjusted'] = bool(bright_changed)
        meta['mean_brightness_before'] = float(mean_before)

    if do_denoise:
        img = denoise_image(img, method=denoise_method)
        meta['denoised'] = True
        meta['denoise_method'] = denoise_method
    else:
        meta['denoised'] = False

    if save_path is not None:
        os.makedirs(os.path.dirname(str(save_path)), exist_ok=True)
        _save_image_cv2(Path(save_path), img)
        if mask is not None:
            mask_out = os.path.splitext(str(save_path))[0] + "_mask.png"
            cv2.imwrite(mask_out, mask)

        # draw bbox on original image and save
        if save_bbox and bbox is not None:
            try:
                x0, y0, w_box, h_box = bbox
                overlay = original_img.copy()
                if hull is not None:
                    cv2.drawContours(
                        overlay,
                        [hull],
                        -1,
                        (0, 0, 255),
                        3,
                        cv2.LINE_AA
                    )
                bbox_out = os.path.splitext(str(save_path))[0] + "_bbox" + os.path.splitext(str(save_path))[1]
                _save_image_cv2(Path(bbox_out), overlay)
            except Exception:
                pass

    return img, meta


# ==========================
# QUICK MAIN / TEST (10 ảnh across up to 4 nhãn)
# ==========================
def main_sample_clean(target_total: int = 10,
                      labels: Optional[List[str]] = None,
                      dataset_root: Path = Path("dataset/raw"),
                      out_root: Path = Path("dataset/cleaned"),
                      random_seed: int = 42) -> List[Dict[str, Any]]:
    """
    Chọn ngẫu nhiên ~target_total ảnh từ dataset_root theo labels (mặc định lấy 4 nhãn đầu).
    Lưu ảnh đã xử lý vào out_root/<label>/...
    Trả về danh sách metadata cho các ảnh đã xử lý.
    """
    random.seed(random_seed)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # default labels nếu không truyền vào
    if labels is None:
        labels = ["BrownSpot", "Healthy", "Hispa", "LeafBlast"]
    labels = labels[:4]

    # collect candidates
    candidates: Dict[str, List[Path]] = {lbl: [] for lbl in labels}
    for lbl in labels:
        p = Path(dataset_root) / lbl
        if p.exists() and p.is_dir():
            files = []
            for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
                files.extend(list(p.glob(ext)))
            candidates[lbl] = files

    per_label = math.ceil(target_total / len(labels))
    selected = []
    for lbl in labels:
        src_files = candidates.get(lbl, [])
        if src_files:
            chosen = random.sample(src_files, min(per_label, len(src_files)))
            selected.extend([(lbl, p) for p in chosen])

    # if not enough, add from any label
    if len(selected) < target_total:
        all_files = [p for lbl in labels for p in candidates.get(lbl, []) if (lbl, p) not in selected]
        more = random.sample(all_files, min(target_total - len(selected), len(all_files))) if all_files else []
        selected.extend([(None, p) for p in more])

    selected = selected[:target_total]

    summary = []
    for idx, (lbl, path) in enumerate(selected):
        rel_out = out_root / (lbl if lbl else "misc")
        rel_out.mkdir(parents=True, exist_ok=True)
        out_path = rel_out / f"{path.stem}_clean{path.suffix}"
        proc_img, meta = clean_image_pipeline(path, save_path=out_path)
        record = {'index': idx, 'label': lbl, 'src': str(path), 'out': str(out_path), **meta}
        summary.append(record)
        print(f"[{idx+1}/{len(selected)}] {path} -> {out_path} | meta: {meta}")

    summary_path = out_root / "clean_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Done. Saved {len(selected)} cleaned images to {out_root}. Summary: {summary_path}")
    return summary


if __name__ == "__main__":
    # ví dụ chạy nhanh: xử lý 10 ảnh across 4 nhãn
    main_sample_clean(target_total=16)
