#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import cv2
import time
import json
import argparse
import signal
import serial
import numpy as np
from typing import Optional, Tuple, Dict, Any

# ========= 可选：tflite runtime 优先 =========
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter


# =========================
# 默认配置（可通过命令行覆盖）
# =========================
DEFAULTS = {
    "OPENMV_PORT": "/dev/ttyACM0",
    "OPENMV_BAUD": 115200,
    "OPENMV_TIMEOUT": 0.15,
    "OPENMV_MAX_JPEG_LEN": 120000,

    "MODEL_PATH": "facenet_512.tflite",
    "INPUT_W": 160,
    "INPUT_H": 160,
    "PREPROCESS_MODE": "minus1_to_1",  # minus1_to_1 / zero_to_1 / raw_0_255

    "OUT_DIR_JSON": "face_db_json",
    "OUT_DIR_ROI": "face_db_roi_samples",  # 可选保存ROI图像目录
    "SAVE_ROI_IMAGES": False,

    "TARGET_SAMPLES": 20,            # 每人目标样本数
    "CAPTURE_INTERVAL_SEC": 0.6,     # 最小采样间隔（防止连续帧太像）
    "DEDUP_SIM_THRESHOLD": 0.985,    # 与已收样本过于相似则跳过
    "MIN_DET_SCORE": 0.0,            # 可设 0.05 / 0.1。FOMO分数低不一定识别差，所以默认不过滤
    "PRINT_DET_EVERY_SEC": 1.0,
    "LOG_EVERY_ACCEPT": True,
    "SHOW_PREVIEW": False,           # 服务器无桌面时建议 False
}


# =========================
# 工具函数
# =========================
def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

def now_ts() -> float:
    return time.time()

def safe_float(s, default=None):
    try:
        return float(s)
    except Exception:
        return default

def l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v) + 1e-12
    return v / n

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)

def sanitize_filename(name: str) -> str:
    # 保留中文、字母、数字、下划线、横线，其它替换为 _
    out = []
    for ch in name.strip():
        if ch.isalnum() or ch in ("_", "-", " "):
            out.append(ch)
        elif "\u4e00" <= ch <= "\u9fff":
            out.append(ch)
        else:
            out.append("_")
    s = "".join(out).strip().replace(" ", "_")
    return s if s else "unknown"


# =========================
# 协议解析（OpenMV -> 树莓派）
# =========================
class OpenMVReceiver:
    """
    协议：
      DET:1;S=0.1234;C=4,6
      DET:0;S=0.0123
      ROI:x,y,w,h
      IMG:<len>
      <len bytes jpeg payload>
    """
    def __init__(self, port: str, baud: int, timeout: float, max_jpeg_len: int):
        self.ser = serial.Serial(port=port, baudrate=baud, timeout=timeout)
        self.max_jpeg_len = max_jpeg_len

        self.last_det = None               # 0/1/None
        self.last_det_score = None         # float/None
        self.last_best_cell = None         # (x,y)/None
        self.last_roi = None               # (x,y,w,h)/None
        self.last_img_len = None
        self.parse_errors = 0

        self._last_det_log_ts = 0.0
        self._print_det_every = DEFAULTS["PRINT_DET_EVERY_SEC"]

    def close(self):
        try:
            self.ser.close()
        except Exception:
            pass

    def _readline_ascii(self) -> Optional[str]:
        raw = self.ser.readline()
        if not raw:
            return None
        try:
            return raw.decode("ascii", errors="strict").strip()
        except UnicodeDecodeError:
            # 读到二进制残片/错位
            self.parse_errors += 1
            return None

    def _read_exact(self, n: int) -> Optional[bytes]:
        data = bytearray()
        deadline = time.time() + 1.5  # 给足一点时间
        while len(data) < n:
            chunk = self.ser.read(n - len(data))
            if chunk:
                data.extend(chunk)
                continue
            if time.time() > deadline:
                break
        if len(data) == n:
            return bytes(data)
        return None

    def poll(self) -> Tuple[str, Optional[bytes]]:
        """
        returns:
            ("NONE", None)
            ("TXT", None)
            ("IMG", bytes)
            ("IMG_FAIL", None)
        """
        line = self._readline_ascii()
        if line is None:
            return ("NONE", None)

        # DET
        if line.startswith("DET:"):
            # 例: DET:1;S=0.1234;C=4,6
            # 或 DET:0;S=0.0123
            try:
                payload = line[4:]
                parts = payload.split(";")
                det_str = parts[0].strip()
                self.last_det = int(det_str) if det_str in ("0", "1") else None

                score = None
                cell = None
                for p in parts[1:]:
                    if p.startswith("S="):
                        score = safe_float(p[2:].strip(), None)
                    elif p.startswith("C="):
                        c = p[2:].strip()
                        if "," in c:
                            cx, cy = c.split(",", 1)
                            cell = (int(cx), int(cy))
                self.last_det_score = score
                self.last_best_cell = cell

                t = time.time()
                if t - self._last_det_log_ts >= self._print_det_every:
                    self._last_det_log_ts = t
                    print(f"[{now_str()}] DET={self.last_det} S={self.last_det_score} C={self.last_best_cell}")

            except Exception:
                self.parse_errors += 1
            return ("TXT", None)

        # ROI
        if line.startswith("ROI:"):
            # ROI:x,y,w,h
            try:
                vals = [int(v) for v in line[4:].split(",")]
                if len(vals) == 4:
                    self.last_roi = tuple(vals)
            except Exception:
                self.parse_errors += 1
            return ("TXT", None)

        # IMG
        if line.startswith("IMG:"):
            try:
                img_len = int(line[4:].strip())
            except Exception:
                self.parse_errors += 1
                return ("IMG_FAIL", None)

            if img_len <= 0 or img_len > self.max_jpeg_len:
                self.parse_errors += 1
                print(f"[{now_str()}] IMG len invalid: {img_len}")
                return ("IMG_FAIL", None)

            payload = self._read_exact(img_len)
            if payload is None:
                self.parse_errors += 1
                print(f"[{now_str()}] IMG payload short read (expect {img_len})")
                return ("IMG_FAIL", None)

            self.last_img_len = img_len
            return ("IMG", payload)

        # 其它行忽略（不报错，增强兼容）
        return ("TXT", None)


# =========================
# Facenet TFLite 推理
# =========================
class FaceNetTFLite:
    def __init__(self, model_path: str, input_w: int, input_h: int, preprocess_mode: str):
        self.model_path = model_path
        self.input_w = input_w
        self.input_h = input_h
        self.preprocess_mode = preprocess_mode

        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.in_index = self.input_details[0]["index"]
        self.out_index = self.output_details[0]["index"]
        self.in_dtype = self.input_details[0]["dtype"]
        self.out_dtype = self.output_details[0]["dtype"]
        self.in_shape = self.input_details[0]["shape"]

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        # OpenMV ROI通常是灰度JPEG，cv2可能解成单通道
        if img is None:
            raise ValueError("input image is None")

        if img.ndim == 2:
            rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.ndim == 3 and img.shape[2] == 3:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"unsupported image shape: {img.shape}")

        rgb = cv2.resize(rgb, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)

        if self.in_dtype == np.float32:
            x = rgb.astype(np.float32)
            if self.preprocess_mode == "minus1_to_1":
                x = (x / 127.5) - 1.0
            elif self.preprocess_mode == "zero_to_1":
                x = x / 255.0
            elif self.preprocess_mode == "raw_0_255":
                pass
            else:
                raise ValueError(f"unknown preprocess mode: {self.preprocess_mode}")
        elif self.in_dtype == np.uint8:
            x = rgb.astype(np.uint8)
        else:
            raise ValueError(f"unsupported model input dtype: {self.in_dtype}")

        x = np.expand_dims(x, axis=0)
        return x

    def infer_embedding(self, img: np.ndarray) -> np.ndarray:
        x = self._preprocess(img)
        self.interpreter.set_tensor(self.in_index, x)
        self.interpreter.invoke()
        y = self.interpreter.get_tensor(self.out_index)
        emb = np.array(y, dtype=np.float32).reshape(-1)
        emb = l2_normalize(emb)
        return emb


# =========================
# JSON 数据管理（每人一个 json）
# =========================
class PersonJsonStore:
    def __init__(self, out_dir_json: str, out_dir_roi: str, save_roi_images: bool):
        self.out_dir_json = out_dir_json
        self.out_dir_roi = out_dir_roi
        self.save_roi_images = save_roi_images

        os.makedirs(self.out_dir_json, exist_ok=True)
        if self.save_roi_images:
            os.makedirs(self.out_dir_roi, exist_ok=True)

    def _json_path(self, person_name: str) -> str:
        fn = sanitize_filename(person_name) + ".json"
        return os.path.join(self.out_dir_json, fn)

    def load_or_init(self, person_name: str, meta: Dict[str, Any]) -> Dict[str, Any]:
        path = self._json_path(person_name)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # 增量更新元信息（不覆盖历史关键字段）
            data.setdefault("person_name", person_name)
            data.setdefault("embeddings", [])
            data.setdefault("samples", [])
            data.setdefault("meta", {})
            data["meta"].update({
                "last_collect_time": now_str(),
                "last_model_path": meta.get("model_path"),
                "last_preprocess_mode": meta.get("preprocess_mode"),
            })
            return data
        else:
            data = {
                "person_name": person_name,
                "version": 1,
                "created_at": now_str(),
                "updated_at": now_str(),
                "meta": {
                    "model_path": meta.get("model_path"),
                    "preprocess_mode": meta.get("preprocess_mode"),
                    "input_size": meta.get("input_size"),
                    "embedding_dim": 512,
                    "source": "openmv_roi_protocol",
                    "protocol": "DET/ROI/IMG(text+binary_jpeg)",
                },
                "embeddings": [],  # list[list[float]]
                "samples": [],     # list[dict] 非必须，但对调试很有用
            }
            return data

    def save(self, person_name: str, data: Dict[str, Any]):
        data["updated_at"] = now_str()
        path = self._json_path(person_name)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

    def save_roi_image(self, person_name: str, img: np.ndarray, idx: int) -> Optional[str]:
        if not self.save_roi_images:
            return None
        person_dir = os.path.join(self.out_dir_roi, sanitize_filename(person_name))
        os.makedirs(person_dir, exist_ok=True)
        fp = os.path.join(person_dir, f"{idx:04d}.jpg")
        cv2.imwrite(fp, img)
        return fp


# =========================
# 采集主流程
# =========================
class CollectorApp:
    def __init__(self, args):
        self.args = args
        self.running = True

        self.receiver = OpenMVReceiver(
            port=args.port,
            baud=args.baud,
            timeout=args.timeout,
            max_jpeg_len=args.max_jpeg_len,
        )
        self.facenet = FaceNetTFLite(
            model_path=args.model,
            input_w=args.input_w,
            input_h=args.input_h,
            preprocess_mode=args.preprocess_mode,
        )
        self.store = PersonJsonStore(
            out_dir_json=args.out_json_dir,
            out_dir_roi=args.out_roi_dir,
            save_roi_images=args.save_roi_images,
        )

        self.person_name = args.name
        self.person_data = self.store.load_or_init(
            self.person_name,
            meta={
                "model_path": args.model,
                "preprocess_mode": args.preprocess_mode,
                "input_size": [args.input_w, args.input_h],
            }
        )

        self.last_accept_ts = 0.0
        self.accepted = len(self.person_data.get("embeddings", []))
        self.skipped_dup = 0
        self.skipped_interval = 0
        self.skipped_det = 0
        self.skipped_decode = 0
        self.total_img_frames = 0
        self.last_status_print_ts = 0.0

    def close(self):
        try:
            self.receiver.close()
        except Exception:
            pass
        if self.args.show_preview:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    def _decode_jpeg(self, payload: bytes) -> Optional[np.ndarray]:
        arr = np.frombuffer(payload, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        return img

    def _embedding_is_duplicate(self, emb: np.ndarray) -> Tuple[bool, float]:
        embs = self.person_data.get("embeddings", [])
        if not embs:
            return False, -1.0

        # 和已有样本求最大相似度
        max_sim = -1.0
        for e in embs:
            e_np = np.asarray(e, dtype=np.float32)
            sim = cosine_similarity(emb, e_np)
            if sim > max_sim:
                max_sim = sim
        return (max_sim >= self.args.dedup_sim_threshold), max_sim

    def _accept_sample(self, emb: np.ndarray, img: np.ndarray):
        idx = len(self.person_data["embeddings"]) + 1

        roi_path = self.store.save_roi_image(self.person_name, img, idx)

        # 保存 embedding + 样本元信息
        self.person_data["embeddings"].append(emb.astype(np.float32).tolist())
        self.person_data["samples"].append({
            "idx": idx,
            "time": now_str(),
            "det": self.receiver.last_det,
            "det_score": self.receiver.last_det_score,
            "best_cell": list(self.receiver.last_best_cell) if self.receiver.last_best_cell else None,
            "roi": list(self.receiver.last_roi) if self.receiver.last_roi else None,
            "img_len": self.receiver.last_img_len,
            "roi_image_path": roi_path,
        })

        self.store.save(self.person_name, self.person_data)
        self.accepted = len(self.person_data["embeddings"])
        self.last_accept_ts = time.time()

        if self.args.log_every_accept:
            print(
                f"[{now_str()}] ACCEPT {self.accepted}/{self.args.target_samples} "
                f"name={self.person_name} det={self.receiver.last_det} "
                f"S={self.receiver.last_det_score} ROI={self.receiver.last_roi}"
            )

    def _print_status(self, force=False):
        t = time.time()
        if (not force) and (t - self.last_status_print_ts < 1.0):
            return
        self.last_status_print_ts = t

        print(
            f"[{now_str()}] STATUS name={self.person_name} "
            f"accepted={self.accepted}/{self.args.target_samples} "
            f"img_frames={self.total_img_frames} "
            f"skip_dup={self.skipped_dup} skip_interval={self.skipped_interval} "
            f"skip_det={self.skipped_det} skip_decode={self.skipped_decode} "
            f"parse_err={self.receiver.parse_errors}"
        )

    def run(self):
        print("========== collecting.py ==========")
        print(f"开始采集人脸 embedding（来源：OpenMV ROI）")
        print(f"Person Name        : {self.person_name}")
        print(f"OpenMV Port        : {self.args.port} @ {self.args.baud}")
        print(f"Model              : {self.args.model}")
        print(f"Input Size         : {self.args.input_w}x{self.args.input_h}")
        print(f"Preprocess Mode    : {self.args.preprocess_mode}")
        print(f"Target Samples     : {self.args.target_samples}")
        print(f"Capture Interval   : {self.args.capture_interval_sec}s")
        print(f"Dedup Sim Threshold: {self.args.dedup_sim_threshold}")
        print(f"Min DET Score      : {self.args.min_det_score}")
        print(f"JSON Dir           : {self.args.out_json_dir}")
        print(f"Save ROI Images    : {self.args.save_roi_images}")
        if self.args.save_roi_images:
            print(f"ROI Dir            : {self.args.out_roi_dir}")
        print("说明：请让同一人以不同角度/距离/表情站在摄像头前，避免只采到几乎相同的帧。")
        print("按 Ctrl+C 结束。")
        print("===================================")

        while self.running:
            evt, payload = self.receiver.poll()

            if evt == "NONE":
                self._print_status(force=False)
                continue

            if evt == "TXT":
                self._print_status(force=False)
                continue

            if evt == "IMG_FAIL":
                self._print_status(force=False)
                continue

            if evt == "IMG" and payload is not None:
                self.total_img_frames += 1

                # 1) 可选：检测状态过滤（默认不强过滤）
                if self.receiver.last_det is not None and self.receiver.last_det == 0:
                    self.skipped_det += 1
                    self._print_status(force=False)
                    continue

                if (self.receiver.last_det_score is not None and
                        self.receiver.last_det_score < self.args.min_det_score):
                    self.skipped_det += 1
                    self._print_status(force=False)
                    continue

                # 2) 节流：采样间隔
                if time.time() - self.last_accept_ts < self.args.capture_interval_sec:
                    self.skipped_interval += 1
                    self._print_status(force=False)
                    continue

                # 3) 解码 JPEG
                img = self._decode_jpeg(payload)
                if img is None:
                    self.skipped_decode += 1
                    self._print_status(force=False)
                    continue

                # 4) facenet embedding
                try:
                    emb = self.facenet.infer_embedding(img)
                except Exception as e:
                    print(f"[{now_str()}] facenet infer error: {e}")
                    self._print_status(force=False)
                    continue

                # 5) 去重（避免连续帧太像）
                is_dup, max_sim = self._embedding_is_duplicate(emb)
                if is_dup:
                    self.skipped_dup += 1
                    # 去重日志节流，避免刷屏
                    if self.skipped_dup % 10 == 1:
                        print(f"[{now_str()}] DUP skip: max_sim={max_sim:.4f} >= {self.args.dedup_sim_threshold}")
                    self._print_status(force=False)
                    continue

                # 6) 接收样本
                self._accept_sample(emb, img)

                # 7) 可选预览
                if self.args.show_preview:
                    preview = img.copy()
                    if preview.ndim == 2:
                        cv2.imshow("collecting_preview", preview)
                    else:
                        cv2.imshow("collecting_preview", preview)
                    cv2.waitKey(1)

                # 8) 达到目标样本数，退出
                if self.accepted >= self.args.target_samples:
                    print(f"[{now_str()}] 已达到目标样本数 {self.args.target_samples}，采集完成。")
                    self._print_status(force=True)
                    break

                self._print_status(force=True)

        self.close()


# =========================
# 命令行参数
# =========================
def build_argparser():
    ap = argparse.ArgumentParser(
        description="从 OpenMV ROI 串口协议采集 facenet embedding，并保存为每人一个 JSON 文件"
    )
    ap.add_argument("--name", required=True, help="人员姓名（用于输出 json 文件名）")
    ap.add_argument("--port", default=DEFAULTS["OPENMV_PORT"], help="OpenMV 串口，如 /dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=DEFAULTS["OPENMV_BAUD"], help="OpenMV 波特率")
    ap.add_argument("--timeout", type=float, default=DEFAULTS["OPENMV_TIMEOUT"], help="串口超时秒数")
    ap.add_argument("--max-jpeg-len", type=int, default=DEFAULTS["OPENMV_MAX_JPEG_LEN"], help="最大JPEG长度保护")

    ap.add_argument("--model", default=DEFAULTS["MODEL_PATH"], help="facenet_512.tflite 路径")
    ap.add_argument("--input-w", type=int, default=DEFAULTS["INPUT_W"], help="模型输入宽")
    ap.add_argument("--input-h", type=int, default=DEFAULTS["INPUT_H"], help="模型输入高")
    ap.add_argument("--preprocess-mode", default=DEFAULTS["PREPROCESS_MODE"],
                    choices=["minus1_to_1", "zero_to_1", "raw_0_255"],
                    help="facenet 预处理模式")

    ap.add_argument("--target-samples", type=int, default=DEFAULTS["TARGET_SAMPLES"], help="目标样本数")
    ap.add_argument("--capture-interval-sec", type=float, default=DEFAULTS["CAPTURE_INTERVAL_SEC"], help="采样最小间隔")
    ap.add_argument("--dedup-sim-threshold", type=float, default=DEFAULTS["DEDUP_SIM_THRESHOLD"], help="去重相似度阈值")
    ap.add_argument("--min-det-score", type=float, default=DEFAULTS["MIN_DET_SCORE"], help="最低DET分数过滤（默认0不过滤）")

    ap.add_argument("--out-json-dir", default=DEFAULTS["OUT_DIR_JSON"], help="json输出目录")
    ap.add_argument("--out-roi-dir", default=DEFAULTS["OUT_DIR_ROI"], help="ROI图像输出目录")
    ap.add_argument("--save-roi-images", action="store_true", help="保存采样ROI图像（用于复盘）")
    ap.add_argument("--show-preview", action="store_true", help="显示预览窗口（有桌面环境时使用）")
    ap.add_argument("--log-every-accept", action="store_true", default=DEFAULTS["LOG_EVERY_ACCEPT"], help="每次接收样本都打印日志")
    return ap


def main():
    args = build_argparser().parse_args()

    app = None

    def _sig_handler(sig, frame):
        if app:
            app.running = False

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    try:
        app = CollectorApp(args)
        app.run()
    except KeyboardInterrupt:
        pass
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    except serial.SerialException as e:
        print(f"[ERROR] 串口异常: {e}")
        sys.exit(2)
    except Exception as e:
        print(f"[ERROR] 未处理异常: {e}")
        raise
    finally:
        if app:
            app.close()


if __name__ == "__main__":
    main()