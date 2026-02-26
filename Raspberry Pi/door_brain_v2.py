#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import io
import cv2
import time
import json
import math
import queue
import serial
import signal
import traceback
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

# ========= 可选：tflite runtime 优先 =========
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter


# =========================
# 配置区（默认值可直接用）
# =========================
CONFIG = {
    # ---- OpenMV 串口 ----
    "OPENMV_PORT": "/dev/ttyACM0",   # 常见为 /dev/ttyACM0 或 /dev/ttyUSB0
    "OPENMV_BAUD": 115200,           # OpenMV USB-VCP 常见 115200（你的固件若不同按实际改）
    "OPENMV_TIMEOUT": 0.15,          # 串口读超时（秒）
    "OPENMV_MAX_JPEG_LEN": 120000,   # 安全上限，防止协议错位时疯狂读内存
    "OPENMV_FRAME_TIMEOUT_MS": 1500,  # 从FRM到END的最大允许时长，超时丢帧并resync
    "OPENMV_TILE_TIMEOUT_MS": 350,    # 从TL行开始读到JPEG payload的最大允许时长

    # ---- STM32 串口 ----
    "ENABLE_STM32": False,
    "STM32_PORT": "/dev/ttyUSB0",    # 按实际改
    "STM32_BAUD": 115200,
    "STM32_TIMEOUT": 0.2,
    "STM32_OPEN_TIMEOUT": 1.5,       # 发 CMD:OPEN 后等待响应总超时
    "STM32_RETRY_ON_TIMEOUT": 0,     # 默认不重发，避免重复开门动作

    # ---- Facenet 模型 ----
    "MODEL_PATH": "facenet_512.tflite",
    "FACE_DB_NPZ": "face_db_embeddings.npz",
    "FACE_DB_DIR": "face_db",        # 若 NPZ 不存在，可提示使用 build_face_db.py 构建
    "INPUT_SIZE": (160, 160),        # facenet 常见 160x160
    "PREPROCESS_MODE": "minus1_to_1",# ["minus1_to_1", "zero_to_1", "raw_0_255"]
    "CHANNEL_MODE": "rgb",           # facenet 通常需要 RGB
    "L2_NORMALIZE_EMB": True,

    # ---- 识别阈值/判定 ----
    "SIM_THRESHOLD": 0.60,           # 初始阈值，后续校准
    "REQUIRE_CONSECUTIVE_PASS": 2,   # 连续通过帧数
    "RECOG_COOLDOWN_SEC_OK": 2.5,    # 开门成功冷却
    "RECOG_COOLDOWN_SEC_FAIL": 1.2,  # 开门失败/异常冷却
    "UNKNOWN_COOLDOWN_SEC": 0.5,     # 未通过短冷却，避免狂算

    # ---- 面板/日志 ----
    "PANEL_REFRESH_HZ": 5,           # 滚动面板刷新频率
    "MAX_EVENTS": 10,                # Recent Events 条数
    "PRINT_DEBUG_LINES": False,      # 是否打印所有文本行（调协议时可开）
    "PRINT_DET_EVERY_SEC": 1.0,      # DET日志节流
    "PRINT_PARSE_ERROR_EVERY_SEC": 1.0,

    # ---- 性能/健壮性 ----
    "DROP_FRAME_WHEN_BUSY": True,    # 正在冷却或忙时可选择快速跳过图像解码/推理
    "MAX_BAD_FRAMES_BEFORE_RESYNC": 3,
}

# ANSI 控制（终端滚动面板）
ANSI_CLEAR = "\x1b[2J"
ANSI_HOME = "\x1b[H"
ANSI_HIDE_CURSOR = "\x1b[?25l"
ANSI_SHOW_CURSOR = "\x1b[?25h"


# =========================
# 数据结构
# =========================
@dataclass
class OpenMVDetState:
    det: Optional[int] = None
    score: Optional[float] = None
    best_cell: Optional[Tuple[int, int]] = None
    roi: Optional[Tuple[int, int, int, int]] = None
    last_img_len: Optional[int] = None
    last_img_ts: float = 0.0
    last_line_ts: float = 0.0

@dataclass
class OpenMVTile:
    k: int
    x: int
    y: int
    w: int
    h: int
    jpeg: bytes  # 原始 JPEG payload（len 字节）


@dataclass
class OpenMVFrame:
    frame_id: int
    det: Optional[int] = None
    score: Optional[float] = None
    best_cell: Optional[Tuple[int, int]] = None
    roi: Optional[Tuple[int, int, int, int]] = None
    tile_count: int = 0
    roi_w: Optional[int] = None  # 来自 TILE:... W/H（你已定义为 ROI 宽高）
    roi_h: Optional[int] = None
    jpeg_quality: Optional[int] = None
    tiles: List[OpenMVTile] = None

    def __post_init__(self):
        if self.tiles is None:
            self.tiles = []



@dataclass
class RecogResult:
    ok: bool
    name: str
    similarity: float
    distance: float
    reason: str = ""


# =========================
# 工具函数
# =========================
def now_str():
    return time.strftime("%H:%M:%S")

def safe_float(s, default=None):
    try:
        return float(s)
    except Exception:
        return default

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    # a,b assumed float32 1D
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)

def l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v) + 1e-12
    return v / n

def parse_kv_line(line: str) -> Dict[str, str]:
    """
    解析 'DET:1;S=0.1234;C=4,6' 这种格式中 ':' 后面的 KV 段
    """
    out = {}
    if ":" not in line:
        return out
    head, tail = line.split(":", 1)
    out["_head"] = head.strip()
    parts = tail.strip().split(";")
    if head.strip() == "DET":
        # DET 的第一个字段可能是纯值 1/0
        if parts and "=" not in parts[0]:
            out["DET"] = parts[0].strip()
            parts = parts[1:]
    for p in parts:
        if "=" in p:
            k, v = p.split("=", 1)
            out[k.strip()] = v.strip()
    return out


# =========================
# 终端滚动面板
# =========================
class RollingPanel:
    def __init__(self, max_events=10):
        self.max_events = max_events
        self.events = deque(maxlen=max_events)
        self.last_render_ts = 0.0

        self.state = {
            "openmv_port": "-",
            "stm32_port": "-",
            "openmv_status": "INIT",
            "stm32_status": "INIT",
            "det": "-",
            "det_score": "-",
            "best_cell": "-",
            "roi": "-",
            "img_len": "-",
            "fps_img": "-",
            "fps_loop": "-",
            "model": "-",
            "db_count": "-",
            "last_match": "-",
            "last_similarity": "-",
            "last_distance": "-",
            "door_action": "-",
            "cooldown_left": "-",
            "consecutive_pass": "0",
            "parse_errors": "0",
            "bad_frames": "0",
        }

    def set(self, **kwargs):
        for k, v in kwargs.items():
            self.state[k] = v

    def add_event(self, msg: str):
        self.events.appendleft(f"[{now_str()}] {msg}")

    def render(self, force=False):
        hz = CONFIG["PANEL_REFRESH_HZ"]
        interval = 1.0 / max(hz, 1)
        t = time.time()
        if (not force) and (t - self.last_render_ts < interval):
            return
        self.last_render_ts = t

        lines = []
        lines.append("Door Brain (Raspberry Pi) - Rolling Panel")
        lines.append("=" * 78)
        lines.append(f"OpenMV: {self.state['openmv_port']} | Status: {self.state['openmv_status']}")
        lines.append(f"STM32 : {self.state['stm32_port']} | Status: {self.state['stm32_status']}")
        lines.append(f"Model : {self.state['model']} | DB count: {self.state['db_count']}")
        lines.append("-" * 78)
        lines.append(
            f"DET={self.state['det']}  Score={self.state['det_score']}  Cell={self.state['best_cell']}  ROI={self.state['roi']}"
        )
        lines.append(
            f"IMG_LEN={self.state['img_len']}  FPS(img)={self.state['fps_img']}  FPS(loop)={self.state['fps_loop']}"
        )
        lines.append(
            f"Match={self.state['last_match']}  Sim={self.state['last_similarity']}  Dist={self.state['last_distance']}"
        )
        lines.append(
            f"Door={self.state['door_action']}  PassCount={self.state['consecutive_pass']}  Cooldown={self.state['cooldown_left']}"
        )
        lines.append(
            f"ParseErr={self.state['parse_errors']}  BadFrames={self.state['bad_frames']}"
        )
        lines.append("-" * 78)
        lines.append("Recent Events:")
        if self.events:
            for e in list(self.events)[:self.max_events]:
                lines.append(f"  {e}")
        else:
            lines.append("  (none)")
        lines.append("=" * 78)

        sys.stdout.write(ANSI_HOME)
        sys.stdout.write("\n".join(lines))
        sys.stdout.write("\n")
        sys.stdout.flush()


# =========================
# Face DB 加载
# =========================
class FaceDB:
    def __init__(self, npz_path: str):
        self.npz_path = npz_path
        self.names: List[str] = []
        self.embs: Optional[np.ndarray] = None  # [N,512]

    def load(self):
        if not os.path.exists(self.npz_path):
            raise FileNotFoundError(
                f"未找到人脸库 NPZ: {self.npz_path}。请先运行 build_face_db.py 构建。"
            )
        data = np.load(self.npz_path, allow_pickle=True)
        names = data["names"]
        embs = data["embeddings"].astype(np.float32)

        # 兼容 object/string 类型
        self.names = [str(x) for x in names.tolist()]
        if embs.ndim != 2:
            raise ValueError(f"embeddings 维度异常: {embs.shape}")
        # 保证库 embedding 已归一化（若未归一化也不怕，这里再做一次）
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
        self.embs = embs / norms

    def __len__(self):
        return 0 if self.embs is None else self.embs.shape[0]

    def match(self, emb: np.ndarray) -> RecogResult:
        if self.embs is None or len(self.names) == 0:
            return RecogResult(False, "N/A", 0.0, 1.0, "face_db_empty")
        sims = self.embs @ emb  # [N]，前提双方已L2 norm
        idx = int(np.argmax(sims))
        sim = float(sims[idx])
        dist = float(1.0 - sim)
        name = self.names[idx]
        ok = sim >= CONFIG["SIM_THRESHOLD"]
        return RecogResult(ok, name if ok else "UNKNOWN", sim, dist, "")

def best_match(self, emb: np.ndarray) -> Tuple[str, float, float]:
    """返回 (best_name, best_similarity, best_distance)，不做阈值截断，便于上层做多tile融合。"""
    if self.embs is None or len(self.names) == 0:
        return ("N/A", 0.0, 1.0)
    sims = self.embs @ emb
    idx = int(np.argmax(sims))
    sim = float(sims[idx])
    dist = float(1.0 - sim)
    return (self.names[idx], sim, dist)


# =========================
# Facenet TFLite 封装
# =========================
class FaceNetTFLite:
    def __init__(self, model_path: str, input_size=(160, 160), preprocess_mode="minus1_to_1"):
        self.model_path = model_path
        self.input_size = input_size
        self.preprocess_mode = preprocess_mode

        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.in_index = self.input_details[0]["index"]
        self.out_index = self.output_details[0]["index"]
        self.in_dtype = self.input_details[0]["dtype"]
        self.out_dtype = self.output_details[0]["dtype"]
        self.in_shape = self.input_details[0]["shape"]  # e.g. [1,160,160,3]

    def _preprocess(self, bgr_img: np.ndarray) -> np.ndarray:
        # OpenMV 发来灰度JPEG，cv2.imdecode 通常得到单通道或BGR，统一处理成RGB
        if bgr_img is None:
            raise ValueError("input image is None")

        if bgr_img.ndim == 2:
            # gray -> rgb
            rgb = cv2.cvtColor(bgr_img, cv2.COLOR_GRAY2RGB)
        elif bgr_img.ndim == 3 and bgr_img.shape[2] == 3:
            rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"unsupported image shape: {bgr_img.shape}")

        w, h = self.input_size
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)

        # 注意：不同 facenet tflite 预处理可能不同，这里提供开关用于排查
        mode = self.preprocess_mode
        if self.in_dtype == np.float32:
            x = rgb.astype(np.float32)
            if mode == "minus1_to_1":
                x = (x / 127.5) - 1.0
            elif mode == "zero_to_1":
                x = x / 255.0
            elif mode == "raw_0_255":
                # 仍为 float32，但不缩放
                pass
            else:
                raise ValueError(f"unknown PREPROCESS_MODE: {mode}")
        elif self.in_dtype == np.uint8:
            # 某些量化模型直接 uint8 输入
            if mode not in ("raw_0_255",):
                # 对量化模型通常不该做 float 归一化
                # 这里允许但提示风险
                x = rgb.astype(np.uint8)
            else:
                x = rgb.astype(np.uint8)
        else:
            raise ValueError(f"unsupported input dtype: {self.in_dtype}")

        x = np.expand_dims(x, axis=0)
        return x

    def infer_embedding(self, img: np.ndarray) -> np.ndarray:
        x = self._preprocess(img)
        self.interpreter.set_tensor(self.in_index, x)
        self.interpreter.invoke()
        y = self.interpreter.get_tensor(self.out_index)

        emb = np.array(y).astype(np.float32).reshape(-1)
        if CONFIG["L2_NORMALIZE_EMB"]:
            emb = l2_normalize(emb)
        return emb


# =========================
# OpenMV 协议读取器
# =========================
class OpenMVReceiver:
    """
    新协议（LF, ASCII 文本 + 二进制 JPEG）：

      FRM:<id>
      DET:<0|1>;S=<score>;C=<cx>,<cy>
      ROI:<x>,<y>,<w>,<h>
      TILE:<n>;W=<roi_w>;H=<roi_h>;Q=<jpeg_quality>
      TL:<k>;X=<tx>;Y=<ty>;W=<tw>;H=<th>;IMG:<len>
      <binary jpeg len bytes>
      ... (重复 n 次 TL+payload)
      END:<id>

    兼容模式：
      - DET-only：只出现 FRM + DET（可不发 ROI/TILE/TL/END）。此时在收到“下一次 FRM”或超时后，
        会将上一帧按 DET-only 结算并上报给上层（FRAME_OK，tiles 为空）。
    """

    def __init__(self, port, baud, timeout):
        self.ser = serial.Serial(port=port, baudrate=baud, timeout=timeout)
        self.state = OpenMVDetState()

        # 统计
        self.buf_line_errors = 0
        self.bad_frames = 0
        self.last_det_log_ts = 0.0
        self.last_parse_err_log_ts = 0.0

        # FPS（按“帧”统计：完成一帧（END 或 DET-only 结算）算 1）
        self.frame_counter = 0
        self.frame_fps_ts = time.time()
        self.frame_fps = 0.0

        # 解析状态机
        self._cur: Optional[OpenMVFrame] = None
        self._cur_start_ts: float = 0.0
        self._expect_tiles: int = 0

    def close(self):
        try:
            self.ser.close()
        except Exception:
            pass

    # ---------- 底层读 ----------
    def _readline_ascii(self) -> Optional[str]:
        raw = self.ser.readline()
        if not raw:
            return None
        try:
            # 协议强制 ASCII；如果这里失败，说明串口错位或混入了二进制残片
            line = raw.decode("ascii", errors="strict").strip()
            return line
        except UnicodeDecodeError:
            self.buf_line_errors += 1
            return None

    def _read_exact(self, n: int, timeout_ms: int) -> Optional[bytes]:
        data = bytearray()
        deadline = time.time() + max(timeout_ms, 50) / 1000.0
        while len(data) < n and time.time() < deadline:
            chunk = self.ser.read(n - len(data))
            if chunk:
                data.extend(chunk)
                continue
        if len(data) == n:
            return bytes(data)
        return None

    # ---------- 工具 ----------
    def _log_parse_error(self, msg: str):
        t = time.time()
        if t - self.last_parse_err_log_ts >= CONFIG["PRINT_PARSE_ERROR_EVERY_SEC"]:
            self.last_parse_err_log_ts = t
            print(f"[{now_str()}] parse_error: {msg}")

    def _update_frame_fps(self):
        self.frame_counter += 1
        t = time.time()
        dt = t - self.frame_fps_ts
        if dt >= 1.0:
            self.frame_fps = self.frame_counter / dt
            self.frame_counter = 0
            self.frame_fps_ts = t

    # ---------- 行解析 ----------
    @staticmethod
    def _parse_int_after(prefix: str, line: str) -> Optional[int]:
        # e.g. prefix="FRM:" line="FRM:123"
        if not line.startswith(prefix):
            return None
        try:
            return int(line[len(prefix):].strip())
        except Exception:
            return None

    @staticmethod
    def _parse_tl_header(line: str) -> Optional[Dict[str, str]]:
        # TL:<k>;X=<tx>;Y=<ty>;W=<tw>;H=<th>;IMG:<len>
        if not line.startswith("TL:"):
            return None
        # 先把 TL: 后面的内容拆出来
        tail = line[3:]
        parts = tail.split(";")
        if not parts:
            return None
        out: Dict[str, str] = {}
        # 第一个字段是 k（纯值）
        out["k"] = parts[0].strip()
        for p in parts[1:]:
            if "=" in p:
                k, v = p.split("=", 1)
                out[k.strip()] = v.strip()
            elif ":" in p:
                # 允许 IMG:<len> 这种写法出现在分号段中
                k, v = p.split(":", 1)
                out[k.strip()] = v.strip()
        # 兼容 IMG:<len> 在行尾的情况（没有被 ; 切到）
        if "IMG" not in out and "IMG:" in line:
            try:
                out["IMG"] = line.split("IMG:", 1)[1].strip()
            except Exception:
                pass
        return out

    @staticmethod
    def _parse_tile_meta(line: str) -> Optional[Dict[str, str]]:
        # TILE:<n>;W=<roi_w>;H=<roi_h>;Q=<jpeg_quality>
        if not line.startswith("TILE:"):
            return None
        tail = line[5:]
        parts = tail.split(";")
        if not parts:
            return None
        out: Dict[str, str] = {"N": parts[0].strip()}
        for p in parts[1:]:
            if "=" in p:
                k, v = p.split("=", 1)
                out[k.strip()] = v.strip()
        return out

    # ---------- 帧状态机 ----------
    def _start_new_frame(self, frame_id: int):
        self._cur = OpenMVFrame(frame_id=frame_id)
        self._cur_start_ts = time.time()
        self._expect_tiles = 0
        # 同步到面板 state
        self.state.last_img_len = None
        self.state.last_img_ts = self._cur_start_ts
        self.state.last_line_ts = self._cur_start_ts

    def _finalize_current_frame(self) -> Optional[OpenMVFrame]:
        """把当前帧结算为可上报对象；返回后会清空 _cur。"""
        if self._cur is None:
            return None
        frm = self._cur
        self._cur = None
        self._expect_tiles = 0
        self._update_frame_fps()
        return frm

    def _is_frame_timeout(self) -> bool:
        if self._cur is None:
            return False
        timeout_ms = int(CONFIG.get("OPENMV_FRAME_TIMEOUT_MS", 1500))
        return (time.time() - self._cur_start_ts) * 1000.0 > timeout_ms

    def _drop_current_frame(self, reason: str):
        self.bad_frames += 1
        self._log_parse_error(f"drop_frame: {reason}")
        self._cur = None
        self._expect_tiles = 0

    def poll(self) -> Tuple[str, Optional[OpenMVFrame]]:
        """
        返回：
          ("NONE", None)          无新数据
          ("TXT_OK", None)        收到并解析了一行文本（但未结算成完整帧）
          ("FRAME_OK", OpenMVFrame) 结算到一帧（完整帧或 DET-only 帧）
          ("FRAME_FAIL", None)    帧读取失败（丢帧并已 resync）
        """
        # 1) 如果帧超时，丢帧并 resync
        if self._is_frame_timeout():
            self._drop_current_frame("frame_timeout")
            return ("FRAME_FAIL", None)

        line = self._readline_ascii()
        if line is None:
            return ("NONE", None)

        self.state.last_line_ts = time.time()

        if CONFIG["PRINT_DEBUG_LINES"]:
            print(f"[DBG] LINE: {line}")

        # 2) FRM：新帧开始（也是最强 resync 点）
        if line.startswith("FRM:"):
            new_id = self._parse_int_after("FRM:", line)
            if new_id is None:
                self._log_parse_error(f"FRM parse fail: {line}")
                return ("TXT_OK", None)

            # 如果已有未结束帧：按 DET-only/半帧策略结算或丢弃
            if self._cur is not None:
                # 只要已经拿到 DET，就按 DET-only 结算；否则直接丢弃
                if self._cur.det is not None:
                    frm = self._finalize_current_frame()
                    self._start_new_frame(new_id)
                    self.state.last_img_len = 0
                    self.state.last_img_ts = time.time()
                    return ("FRAME_OK", frm)
                else:
                    self._drop_current_frame("new_FRM_before_DET")
            self._start_new_frame(new_id)
            return ("TXT_OK", None)

        # 3) 如果没有处于帧内，直接忽略其它行（等待 FRM resync）
        if self._cur is None:
            return ("TXT_OK", None)

        # 4) DET
        if line.startswith("DET:"):
            kv = parse_kv_line(line)
            det = kv.get("DET")
            s = safe_float(kv.get("S", ""), None)
            c = kv.get("C")

            self._cur.det = int(det) if det in ("0", "1") else None
            self._cur.score = s
            if c and "," in c:
                try:
                    cx, cy = c.split(",", 1)
                    self._cur.best_cell = (int(cx), int(cy))
                except Exception:
                    self._cur.best_cell = None
            else:
                self._cur.best_cell = None

            # 同步到 state（面板用）
            self.state.det = self._cur.det
            self.state.score = self._cur.score
            self.state.best_cell = self._cur.best_cell

            return ("TXT_OK", None)

        # 5) ROI
        if line.startswith("ROI:"):
            payload = line[4:]
            try:
                x, y, w, h = [int(v) for v in payload.split(",")]
                self._cur.roi = (x, y, w, h)
                self.state.roi = self._cur.roi
            except Exception:
                self._cur.roi = None
                self.state.roi = None
                self._log_parse_error(f"ROI parse fail: {line}")
            return ("TXT_OK", None)

        # 6) TILE meta
        if line.startswith("TILE:"):
            meta = self._parse_tile_meta(line)
            if meta is None:
                self._log_parse_error(f"TILE parse fail: {line}")
                return ("TXT_OK", None)
            try:
                n = int(meta.get("N", "0"))
                roi_w = int(meta.get("W")) if meta.get("W") is not None else None
                roi_h = int(meta.get("H")) if meta.get("H") is not None else None
                q = int(meta.get("Q")) if meta.get("Q") is not None else None
            except Exception:
                self._log_parse_error(f"TILE meta invalid: {line}")
                return ("TXT_OK", None)

            self._cur.tile_count = n
            self._cur.roi_w = roi_w
            self._cur.roi_h = roi_h
            self._cur.jpeg_quality = q
            self._expect_tiles = n
            return ("TXT_OK", None)

        # 7) TL + binary JPEG
        if line.startswith("TL:"):
            if self._expect_tiles <= 0:
                # 未声明 TILE 就来 TL，视为协议错乱：丢帧并 resync
                self._drop_current_frame("TL_without_TILE_meta")
                return ("FRAME_FAIL", None)

            hdr = self._parse_tl_header(line)
            if hdr is None:
                self._drop_current_frame("TL_header_parse_fail")
                return ("FRAME_FAIL", None)

            try:
                k = int(hdr["k"])
                tx = int(hdr["X"])
                ty = int(hdr["Y"])
                tw = int(hdr["W"])
                th = int(hdr["H"])
                img_len = int(hdr["IMG"])
            except Exception:
                self._drop_current_frame(f"TL_fields_invalid: {line}")
                return ("FRAME_FAIL", None)

            # 协议长度上限保护（按你的建议 MAX_TILE_LEN 12000 更合适，这里用 OPENMV_MAX_JPEG_LEN 兜底）
            if img_len <= 0 or img_len > CONFIG["OPENMV_MAX_JPEG_LEN"]:
                self._drop_current_frame(f"TL_img_len_invalid: {img_len}")
                return ("FRAME_FAIL", None)

            payload = self._read_exact(img_len, timeout_ms=int(CONFIG.get("OPENMV_TILE_TIMEOUT_MS", 350)))
            if payload is None or len(payload) != img_len:
                self._drop_current_frame(f"TL_payload_short_read: k={k} expect={img_len}")
                return ("FRAME_FAIL", None)

            # 更新面板显示用的“最近图像长度”
            self.state.last_img_len = img_len
            self.state.last_img_ts = time.time()

            # 存 tile
            self._cur.tiles.append(OpenMVTile(k=k, x=tx, y=ty, w=tw, h=th, jpeg=payload))

            # tile 全部收齐后，等待 END
            return ("TXT_OK", None)

        # 8) END：帧结束，结算
        if line.startswith("END:"):
            end_id = self._parse_int_after("END:", line)
            if end_id is None or end_id != self._cur.frame_id:
                self._drop_current_frame(f"END_id_mismatch: {line}")
                return ("FRAME_FAIL", None)

            frm = self._finalize_current_frame()
            return ("FRAME_OK", frm)

        # 9) 其他未知行：忽略（但不丢帧）
        return ("TXT_OK", None)


# =========================
# STM32 通信
# =========================
class STM32Controller:
    def __init__(self, port, baud, timeout):
        self.ser = serial.Serial(port=port, baudrate=baud, timeout=timeout)
        self.last_status = "READY"

    def close(self):
        try:
            self.ser.close()
        except Exception:
            pass

    def _readline(self) -> Optional[str]:
        raw = self.ser.readline()
        if not raw:
            return None
        try:
            return raw.decode("ascii", errors="ignore").strip()
        except Exception:
            return None

    def send_open_and_wait(self) -> Tuple[str, str]:
        """
        returns (code, human_msg)
        code:
          OK_DONE / ERR_BUSY / ERR_JAM / ERR_TIMEOUT / ERR_SERIAL / ERR_UNKNOWN
        """
        try:
            self.ser.reset_input_buffer()
            cmd = b"CMD:OPEN\n"
            self.ser.write(cmd)
            self.ser.flush()
        except Exception as e:
            return ("ERR_SERIAL", f"串口发送异常: {e}")

        deadline = time.time() + CONFIG["STM32_OPEN_TIMEOUT"]
        while time.time() < deadline:
            line = self._readline()
            if not line:
                continue

            if line == "ACK:DONE":
                return ("OK_DONE", "开门动作完成")
            elif line == "ERR:BUSY":
                return ("ERR_BUSY", "执行器忙")
            elif line == "ERR:JAM":
                return ("ERR_JAM", "执行器卡滞")
            else:
                # 未知包继续等到超时
                continue

        return ("ERR_TIMEOUT", "等待STM32响应超时")


# =========================
# 主程序
# =========================
class DoorBrainApp:
    def __init__(self):
        self.panel = RollingPanel(max_events=CONFIG["MAX_EVENTS"])
        self.running = True

        self.openmv: Optional[OpenMVReceiver] = None
        self.stm32: Optional[STM32Controller] = None
        self.facenet: Optional[FaceNetTFLite] = None
        self.face_db: Optional[FaceDB] = None

        self.last_loop_ts = time.time()
        self.loop_counter = 0
        self.loop_fps = 0.0
        self.loop_fps_ts = time.time()

        self.cooldown_until = 0.0
        self.consecutive_pass = 0
        self.last_recog_name = "-"
        self.last_recog_sim = None
        self.last_recog_dist = None
        self.last_door_action = "-"

    def init_all(self):
        print("初始化开始...")
        print(f"1) 打开 OpenMV 串口: {CONFIG['OPENMV_PORT']} @ {CONFIG['OPENMV_BAUD']}")
        self.openmv = OpenMVReceiver(CONFIG["OPENMV_PORT"], CONFIG["OPENMV_BAUD"], CONFIG["OPENMV_TIMEOUT"])
        self.panel.set(openmv_port=CONFIG["OPENMV_PORT"], openmv_status="READY")
        self.panel.add_event("OpenMV 串口已连接")

        if CONFIG.get("ENABLE_STM32", True):
            print(f"2) 打开 STM32 串口: {CONFIG['STM32_PORT']} @ {CONFIG['STM32_BAUD']}")
            try:
                self.stm32 = STM32Controller(
                    CONFIG["STM32_PORT"],
                    CONFIG["STM32_BAUD"],
                    CONFIG["STM32_TIMEOUT"]
                )
                self.panel.set(stm32_port=CONFIG["STM32_PORT"], stm32_status="READY")
                self.panel.add_event("STM32 串口已连接")
            except Exception as e:
                # 给出明确错误，便于现场排查；严格模式下仍抛出
                self.stm32 = None
                self.panel.set(stm32_port=CONFIG["STM32_PORT"], stm32_status="ERR")
                self.panel.add_event(f"STM32 串口打开失败: {e}")
                print("   [ERROR] STM32 串口打开失败。")
                print(f"   端口: {CONFIG['STM32_PORT']}")
                print(f"   异常: {e}")
                print("   请先执行: ls /dev/ttyUSB* /dev/ttyACM* 2>/dev/null")
                print("   若当前只调识别链，可将 CONFIG['ENABLE_STM32'] = False")
                raise
        else:
            print("2) STM32 串口已禁用（ENABLE_STM32=False）")
            self.stm32 = None
            self.panel.set(stm32_port="-", stm32_status="DISABLED")
            self.panel.add_event("STM32 已禁用，当前仅验证识别链")
        print(f"3) 加载 facenet 模型: {CONFIG['MODEL_PATH']}")
        self.facenet = FaceNetTFLite(
            CONFIG["MODEL_PATH"],
            input_size=CONFIG["INPUT_SIZE"],
            preprocess_mode=CONFIG["PREPROCESS_MODE"]
        )
        in_shape = tuple(self.facenet.in_shape.tolist()) if hasattr(self.facenet.in_shape, "tolist") else tuple(self.facenet.in_shape)
        self.panel.set(model=f"{os.path.basename(CONFIG['MODEL_PATH'])} | in={in_shape} | prep={CONFIG['PREPROCESS_MODE']}")
        self.panel.add_event(f"模型已加载，PREPROCESS_MODE={CONFIG['PREPROCESS_MODE']}")

        print(f"4) 加载人脸库 NPZ: {CONFIG['FACE_DB_NPZ']}")
        self.face_db = FaceDB(CONFIG["FACE_DB_NPZ"])
        self.face_db.load()
        self.panel.set(db_count=str(len(self.face_db)))
        self.panel.add_event(f"人脸库已加载，共 {len(self.face_db)} 条 embedding")

        print("5) 参数检查")
        print(f"   SIM_THRESHOLD={CONFIG['SIM_THRESHOLD']}")
        print(f"   REQUIRE_CONSECUTIVE_PASS={CONFIG['REQUIRE_CONSECUTIVE_PASS']}")
        print(f"   COOLDOWN_OK={CONFIG['RECOG_COOLDOWN_SEC_OK']}s, COOLDOWN_FAIL={CONFIG['RECOG_COOLDOWN_SEC_FAIL']}s")
        print("初始化完成。进入主循环。")

    def shutdown(self):
        self.running = False
        if self.openmv:
            self.openmv.close()
        if self.stm32:
            self.stm32.close()

    def _decode_jpeg(self, payload: bytes) -> Optional[np.ndarray]:
        arr = np.frombuffer(payload, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        return img

    def _should_skip_heavy_work(self) -> bool:
        # 冷却期间可跳过解码/推理，节省CPU
        if time.time() < self.cooldown_until and CONFIG["DROP_FRAME_WHEN_BUSY"]:
            return True
        return False

    
def _recog_and_decide(self, img: np.ndarray) -> RecogResult:
    """单张图像识别（保留：用于兼容/测试）。"""
    emb = self.facenet.infer_embedding(img)
    name, sim, dist = self.face_db.best_match(emb)
    ok = sim >= CONFIG["SIM_THRESHOLD"]
    return RecogResult(ok, name if ok else "UNKNOWN", sim, dist, "")

def _recog_frame_and_decide(self, frame: OpenMVFrame) -> RecogResult:
    """多 tile 融合：对每个 tile 提 embedding，取全局最大相似度作为该帧判定。"""
    if frame.tiles is None or len(frame.tiles) == 0:
        return RecogResult(False, "N/A", 0.0, 1.0, "no_tiles")
    best_name = "N/A"
    best_sim = -1.0
    best_dist = 1.0

    # 允许 tile 乱序：按收到的顺序即可；如需固定顺序，可按 k 排序
    for tl in frame.tiles:
        img = self._decode_jpeg(tl.jpeg)
        if img is None:
            self.openmv.bad_frames += 1
            continue
        emb = self.facenet.infer_embedding(img)
        name, sim, dist = self.face_db.best_match(emb)
        if sim > best_sim:
            best_sim = sim
            best_name = name
            best_dist = dist

    if best_sim < 0:
        return RecogResult(False, "N/A", 0.0, 1.0, "all_tiles_decode_fail")

    ok = best_sim >= CONFIG["SIM_THRESHOLD"]
    return RecogResult(ok, best_name if ok else "UNKNOWN", float(best_sim), float(best_dist), "")

    def _trigger_open(self, res: RecogResult):
        self.panel.add_event(f"识别通过: {res.name} sim={res.similarity:.3f}，发送 CMD:OPEN")
        if not self.stm32:
            # 未连接 STM32 时做软模拟，避免 NoneType 报错
            self.last_door_action = "SIM_OPEN"
            self.panel.set(door_action="SIM_OPEN")
            self.panel.add_event("STM32 禁用：已模拟开门")
            self.cooldown_until = time.time() + CONFIG["RECOG_COOLDOWN_SEC_OK"]
            return

        code, human = self.stm32.send_open_and_wait()
        self.last_door_action = code
        self.panel.set(door_action=code)

        if code == "OK_DONE":
            self.panel.add_event(f"开门成功: {human}")
            self.cooldown_until = time.time() + CONFIG["RECOG_COOLDOWN_SEC_OK"]
        else:
            self.panel.add_event(f"开门失败: {code} ({human})")
            self.cooldown_until = time.time() + CONFIG["RECOG_COOLDOWN_SEC_FAIL"]

    def _update_loop_fps(self):
        self.loop_counter += 1
        t = time.time()
        dt = t - self.loop_fps_ts
        if dt >= 1.0:
            self.loop_fps = self.loop_counter / dt
            self.loop_counter = 0
            self.loop_fps_ts = t

    def _update_panel_state_from_openmv(self):
        st = self.openmv.state
        self.panel.set(
            det=str(st.det) if st.det is not None else "-",
            det_score=f"{st.score:.4f}" if st.score is not None else "-",
            best_cell=str(st.best_cell) if st.best_cell is not None else "-",
            roi=str(st.roi) if st.roi is not None else "-",
            img_len=str(st.last_img_len) if st.last_img_len is not None else "-",
            fps_img=f"{self.openmv.frame_fps:.1f}",
            fps_loop=f"{self.loop_fps:.1f}",
            parse_errors=str(self.openmv.buf_line_errors),
            bad_frames=str(self.openmv.bad_frames),
            consecutive_pass=str(self.consecutive_pass),
            cooldown_left=f"{max(0.0, self.cooldown_until - time.time()):.2f}s",
        )


def run(self):
    # 初始化阶段按你的要求逐条打印
    self.init_all()

    # 切换到滚动面板显示
    sys.stdout.write(ANSI_CLEAR + ANSI_HOME + ANSI_HIDE_CURSOR)
    sys.stdout.flush()
    self.panel.render(force=True)

    while self.running:
        try:
            self._update_loop_fps()

            evt, frame = self.openmv.poll()

            if evt == "NONE":
                self._update_panel_state_from_openmv()
                self.panel.render()
                continue

            if evt == "TXT_OK":
                self._update_panel_state_from_openmv()
                self.panel.render()
                continue

            if evt == "FRAME_FAIL":
                self.consecutive_pass = 0
                self._update_panel_state_from_openmv()
                self.panel.add_event("帧读取失败/超时，已丢弃并 resync")
                self.panel.render(force=True)
                continue

            if evt == "FRAME_OK" and frame is not None:
                # receiver 内已把 DET/ROI 同步到 self.openmv.state
                self._update_panel_state_from_openmv()

                # DET-only：无 tiles，不做识别链（只更新有人/无人状态）
                if frame.tiles is None or len(frame.tiles) == 0:
                    self.panel.render()
                    continue

                # 若 DET=0，则直接跳过识别链，减少 CPU
                if frame.det == 0:
                    self.consecutive_pass = 0
                    self.last_recog_name = "-"
                    self.last_recog_sim = None
                    self.last_recog_dist = None
                    self.panel.set(last_match="-", last_similarity="-", last_distance="-")
                    self.panel.render()
                    continue

                if self._should_skip_heavy_work():
                    self.panel.render()
                    continue

                # 多 tile 融合识别：取最大相似度
                res = self._recog_frame_and_decide(frame)

                self.last_recog_name = res.name
                self.last_recog_sim = res.similarity
                self.last_recog_dist = res.distance
                self.panel.set(
                    last_match=res.name,
                    last_similarity=f"{res.similarity:.3f}",
                    last_distance=f"{res.distance:.3f}",
                )

                if res.ok:
                    self.consecutive_pass += 1
                    self.panel.add_event(
                        f"匹配候选: {res.name} sim={res.similarity:.3f} dist={res.distance:.3f} "
                        f"({self.consecutive_pass}/{CONFIG['REQUIRE_CONSECUTIVE_PASS']})"
                    )
                    if self.consecutive_pass >= CONFIG["REQUIRE_CONSECUTIVE_PASS"]:
                        self._trigger_open(res)
                        self.consecutive_pass = 0
                else:
                    self.consecutive_pass = 0
                    self.panel.add_event(
                        f"未通过: best={res.name} sim={res.similarity:.3f} < TH={CONFIG['SIM_THRESHOLD']:.2f}"
                    )
                    self.cooldown_until = time.time() + CONFIG["UNKNOWN_COOLDOWN_SEC"]

                self._update_panel_state_from_openmv()
                self.panel.render(force=True)
                continue

        except KeyboardInterrupt:
            break
        except Exception as e:
            self.panel.add_event(f"主循环异常: {e}")
            self.panel.set(openmv_status="ERR")
            self.panel.render(force=True)
            time.sleep(0.2)

    self.shutdown()
    sys.stdout.write(ANSI_SHOW_CURSOR + "\n")
    sys.stdout.flush()

# =========================
# 启动入口
# =========================
def main():
    # Ctrl+C 优雅退出
    app = DoorBrainApp()

    def _sig_handler(sig, frame):
        app.shutdown()

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    app.run()


if __name__ == "__main__":
    main()