# openmv_fomo_face_v2.py
# OpenMV: official ml.Model + Fomo(postprocess) + present state machine + ROI crop + TILE(3) JPEG send (V2 strict)

import sensor, image, time, gc
from pyb import UART

import ml
from ml.postprocessing.edgeimpulse import Fomo

# =========================
# Config
# =========================
UART_PORT = 3
UART_BAUD = 921600

# Sensor / windowing
# 强约束：必须 GRAYSCALE
sensor.reset()
sensor.set_pixformat(sensor.GRAYSCALE)

# 建议：QVGA 320x240 + windowing 240x240（官方示例）
# 注意：windowing 会改变坐标系基准为 window 内坐标
sensor.set_framesize(sensor.QVGA)         # 320x240
sensor.set_windowing((240, 240))          # 240x240 window (center crop)
sensor.skip_frames(time=1500)
sensor.set_auto_gain(False)
sensor.set_auto_whitebal(False)

IMG_W = 240
IMG_H = 240

# FOMO model (official built-in)
FOMO_THRESHOLD = 0.40
MODEL_PATH = "/rom/fomo_face_detection.tflite"
model = ml.Model(MODEL_PATH, postprocess=Fomo(threshold=FOMO_THRESHOLD))

# present/absent hysteresis + consecutive frames
ENTER_TH = 0.65
EXIT_TH  = 0.45
ENTER_FRAMES = 2
EXIT_FRAMES  = 3

# neighborhood aggregation around best box center (in box units)
# 这里用 box 级别聚合：取 top-K box 的“簇中心”，更适配 face fomo（输出是 boxes）
TOPK = 3

# ROI sizing / margin
MARGIN = 18
MIN_ROI_W = 96
MIN_ROI_H = 96
MAX_ROI_W = 200
MAX_ROI_H = 200

# ROI smoothing
EMA_A = 0.35

# send throttling
SEND_INTERVAL_MS = 350
MOVE_TRIGGER_PX = 18          # ROI center move triggers resend even within interval

# JPEG quality per tile
JPEG_Q = 50

# log throttling
LOG_INTERVAL_MS = 700

# =========================
# UART helpers
# =========================
uart = UART(UART_PORT, UART_BAUD, timeout_char=10)

def write_line(s):
    uart.write(s)
    uart.write("\n")

# =========================
# Utils
# =========================
def clamp(v, lo, hi):
    if v < lo: return lo
    if v > hi: return hi
    return v

def ema(prev, cur, a):
    if prev is None:
        return cur
    return int(prev + a * (cur - prev))

def rect_expand_clip(x, y, w, h, margin, img_w, img_h):
    x -= margin
    y -= margin
    w += 2 * margin
    h += 2 * margin

    w = clamp(w, MIN_ROI_W, MAX_ROI_W)
    h = clamp(h, MIN_ROI_H, MAX_ROI_H)

    x = clamp(x, 0, img_w - w)
    y = clamp(y, 0, img_h - h)
    return (x, y, w, h)

def send_det_only(frm_id, det, score, cx=-1, cy=-1):
    write_line("FRM:%d" % frm_id)
    if det:
        if cx >= 0 and cy >= 0:
            write_line("DET:1;S=%.3f;C=%d,%d" % (score, cx, cy))
        else:
            write_line("DET:1;S=%.3f" % score)
    else:
        write_line("DET:0;S=%.3f" % score)
    write_line("END:%d" % frm_id)

def send_tile(k, tx, ty, tw, th, tile_img):
    # TL header + IMG payload (binary) exactly by len
    jpg = tile_img.compress(quality=JPEG_Q)
    ln = len(jpg)
    write_line("TL:%d;X=%d;Y=%d;W=%d;H=%d;IMG:%d" % (k, tx, ty, tw, th, ln))
    uart.write(jpg)
    del jpg
    gc.collect()

def send_image_frame(frm_id, det_score, best_cx, best_cy, rx, ry, rw, rh, roi_img):
    # header
    write_line("FRM:%d" % frm_id)
    write_line("DET:1;S=%.3f;C=%d,%d" % (det_score, best_cx, best_cy))
    write_line("ROI:%d,%d,%d,%d" % (rx, ry, rw, rh))
    write_line("TILE:3;W=%d;H=%d;Q=%d" % (rw, rh, JPEG_Q))

    # 3 tiles fixed layout
    h2 = rh // 2
    w2 = rw // 2

    # k=0 top
    t0 = roi_img.copy(roi=(0, 0, rw, h2))
    send_tile(0, 0, 0, rw, h2, t0)
    del t0
    gc.collect()

    # k=1 bottom-left
    t1 = roi_img.copy(roi=(0, h2, w2, rh - h2))
    send_tile(1, 0, h2, w2, rh - h2, t1)
    del t1
    gc.collect()

    # k=2 bottom-right
    t2 = roi_img.copy(roi=(w2, h2, rw - w2, rh - h2))
    send_tile(2, w2, h2, rw - w2, rh - h2, t2)
    del t2
    gc.collect()

    write_line("END:%d" % frm_id)

# =========================
# FOMO output -> best ROI
# =========================
def pick_best_from_fomo(result):
    """
    result = model.predict(img)
    官方 Fomo postprocess 通常返回：
      result[0].rect -> list of rectangles / bounding boxes
    在 OpenMV 的 EI FOMO 示例里常见使用：for obj in result: obj.x, obj.y, obj.w, obj.h, obj.value
    这里做兼容：尝试从 result 迭代取 bbox + score。
    返回：(best_score, best_box(x,y,w,h), coarse_cell(cx,cy))
    coarse_cell：用 bbox 中心量化到 12x12 网格（仅用于展示/调试字段 C=）
    """
    best = None
    best_s = 0.0
    boxes = []

    try:
        # result 可能是列表
        for obj in result:
            # obj 可能有 .rect() 或 .x/.y/.w/.h 或者是 tuple
            if hasattr(obj, "value"):
                s = float(obj.value)
            elif hasattr(obj, "score"):
                s = float(obj.score)
            else:
                # unknown, skip
                continue

            if hasattr(obj, "x") and hasattr(obj, "y") and hasattr(obj, "w") and hasattr(obj, "h"):
                x, y, w, h = int(obj.x), int(obj.y), int(obj.w), int(obj.h)
            elif hasattr(obj, "rect"):
                r = obj.rect
                x, y, w, h = int(r[0]), int(r[1]), int(r[2]), int(r[3])
            else:
                continue

            boxes.append((s, x, y, w, h))
            if s > best_s:
                best_s = s
                best = (x, y, w, h)
    except Exception:
        return (0.0, None, (-1, -1), [])

    # coarse cell for debug (quantize to 12x12)
    if best is None:
        return (0.0, None, (-1, -1), boxes)

    x, y, w, h = best
    cxp = x + w // 2
    cyp = y + h // 2
    gw = 12
    gh = 12
    cx = clamp(int(cxp * gw / IMG_W), 0, gw - 1)
    cy = clamp(int(cyp * gh / IMG_H), 0, gh - 1)
    return (best_s, best, (cx, cy), boxes)

def aggregate_score(boxes):
    """
    用 top-K box 的 score 做一个更稳的 present 判定分数（0..1）。
    boxes: list of (s,x,y,w,h)
    """
    if not boxes:
        return 0.0
    boxes.sort(key=lambda t: t[0], reverse=True)
    k = TOPK if len(boxes) >= TOPK else len(boxes)
    acc = 0.0
    for i in range(k):
        acc += boxes[i][0]
    return acc / k

# =========================
# Main loop state machine
# =========================
clock = time.clock()
frm_id = 0

present = False
enter_cnt = 0
exit_cnt = 0

last_send_ms = 0
last_log_ms = 0

# EMA ROI
ema_x = None
ema_y = None
ema_w = None
ema_h = None

# track ROI center for movement-trigger resend
last_roi_c = None  # (cx,cy)

while True:
    clock.tick()
    frm_id = (frm_id + 1) & 0x7FFFFFFF
    now = time.ticks_ms()

    try:
        img = sensor.snapshot()

        # official API: model.predict(img)
        result = model.predict(img)

        best_s, best_box, coarse_cell, all_boxes = pick_best_from_fomo(result)
        agg_s = aggregate_score(all_boxes)

        # hysteresis + consecutive confirm
        if not present:
            if agg_s >= ENTER_TH:
                enter_cnt += 1
            else:
                enter_cnt = 0
            if enter_cnt >= ENTER_FRAMES:
                present = True
                exit_cnt = 0
        else:
            if agg_s <= EXIT_TH:
                exit_cnt += 1
            else:
                exit_cnt = 0
            if exit_cnt >= EXIT_FRAMES:
                present = False
                enter_cnt = 0

        # log throttle
        if time.ticks_diff(now, last_log_ms) >= LOG_INTERVAL_MS:
            last_log_ms = now
            write_line("DBG:fps=%.1f present=%d agg=%.3f best=%.3f cell=%d,%d boxes=%d" %
                       (clock.fps(), 1 if present else 0, agg_s, best_s, coarse_cell[0], coarse_cell[1], len(all_boxes)))

        if not present:
            send_det_only(frm_id, 0, agg_s, coarse_cell[0], coarse_cell[1])
            del img
            gc.collect()
            continue

        # present but no box yet -> det-only (keeps Pi alive)
        if best_box is None:
            send_det_only(frm_id, 1, agg_s, coarse_cell[0], coarse_cell[1])
            del img
            gc.collect()
            continue

        # build ROI around best box
        bx, by, bw, bh = best_box

        # 基于 box 做 ROI：先扩大到不小于 MIN，再 margin
        # 先扩到 min size（以 box center 为基准）
        cxp = bx + bw // 2
        cyp = by + bh // 2

        rw = clamp(max(bw + 2 * MARGIN, MIN_ROI_W), MIN_ROI_W, MAX_ROI_W)
        rh = clamp(max(bh + 2 * MARGIN, MIN_ROI_H), MIN_ROI_H, MAX_ROI_H)

        rx = cxp - rw // 2
        ry = cyp - rh // 2
        rx, ry, rw, rh = rect_expand_clip(rx, ry, rw, rh, 0, IMG_W, IMG_H)  # margin已体现在rw/rh上了

        # EMA smoothing
        ema_x = ema(ema_x, rx, EMA_A)
        ema_y = ema(ema_y, ry, EMA_A)
        ema_w = ema(ema_w, rw, EMA_A)
        ema_h = ema(ema_h, rh, EMA_A)

        rx, ry, rw, rh = ema_x, ema_y, ema_w, ema_h
        rw = clamp(rw, MIN_ROI_W, MAX_ROI_W)
        rh = clamp(rh, MIN_ROI_H, MAX_ROI_H)
        rx = clamp(rx, 0, IMG_W - rw)
        ry = clamp(ry, 0, IMG_H - rh)

        # send throttle: time-based OR ROI center move
        force = False
        cur_c = (rx + rw // 2, ry + rh // 2)
        if last_roi_c is None:
            force = True
        else:
            dx = abs(cur_c[0] - last_roi_c[0])
            dy = abs(cur_c[1] - last_roi_c[1])
            if (dx + dy) >= MOVE_TRIGGER_PX:
                force = True

        if (not force) and (time.ticks_diff(now, last_send_ms) < SEND_INTERVAL_MS):
            send_det_only(frm_id, 1, agg_s, coarse_cell[0], coarse_cell[1])
            del img
            gc.collect()
            continue

        last_send_ms = now
        last_roi_c = cur_c

        # crop ROI (still grayscale)
        roi_img = img.copy(roi=(rx, ry, rw, rh))
        del img
        gc.collect()

        # send full image frame (3 tiles)
        send_image_frame(frm_id, agg_s, coarse_cell[0], coarse_cell[1], rx, ry, rw, rh, roi_img)

        del roi_img
        gc.collect()

    except MemoryError:
        # must close frame strictly with END
        write_line("FRM:%d" % frm_id)
        write_line("ERR:MEM")
        write_line("END:%d" % frm_id)
        gc.collect()
        continue
    except Exception as e:
        # keep running, but close frame strictly
        write_line("FRM:%d" % frm_id)
        write_line("ERR:EXC;%s" % str(e).replace("\n", " "))
        write_line("END:%d" % frm_id)
        gc.collect()
        continue
