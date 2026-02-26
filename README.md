# Face Recognition Access Control (OpenMV + Raspberry Pi + STM32)

A practical 3-node access control system:

- **OpenMV**: lightweight **presence/face-region trigger** using FOMO + **ROI crop** + **send ROI** over serial/USB (no identity recognition on OpenMV).
- **Raspberry Pi (Ubuntu)**: receives ROI → **FaceNet (512-dim) embedding** → match against **local embedding database** → decision (open/deny) → terminal rolling panel UI.
- **STM32**: executes door action (lock/servo/relay) and returns status (`ACK:DONE`, `ERR:*`).

This project is engineered for **memory-constrained OpenMV**, stable serial protocols, and field debugging.

---

## 1. System Architecture (Fixed Responsibilities)

### OpenMV (Trigger + ROI Sender)
- Runs **FOMO** to detect “person present / face region”.
- Works in **GRAYSCALE** (RGB full-frame transmission is not feasible due to memory).
- Crops a **ROI** from the original frame (with neighborhood aggregation + margin).
- Sends **ROI JPEG** to Raspberry Pi using a simple text header + binary payload protocol.

### Raspberry Pi (Recognition + Control)
- Receives ROI JPEG from OpenMV.
- Runs **`facenet_512.tflite`** to produce a 512-dim embedding.
- L2-normalizes embeddings and compares with a local embedding DB (cosine similarity).
- Uses **consecutive pass** and **cooldown** logic to prevent false triggers.
- Talks to STM32 over serial and waits for `ACK:DONE` / `ERR:*`.
- Uses a **rolling terminal panel** (no flooding / no infinite scrolling logs).

### STM32 (Actuation)
- Executes physical door action.
- Responds with:
  - `ACK:DONE`
  - `ERR:BUSY`
  - `ERR:JAM`
  - (others as needed)

---

## 2. Serial Protocol: OpenMV → Raspberry Pi (Must Stay Compatible)

Text lines are ASCII, terminated by newline. Image payload is raw binary JPEG.

### Text Lines
- `DET:1;S=0.1234;C=4,6`  
  Detected presence. `S` is FOMO score, `C` is best cell (cx,cy).
- `DET:0;S=0.0123`  
  No detection.
- `ROI:x,y,w,h`  
  Current ROI crop in original frame coordinates.
- `IMG:<len>`  
  Immediately followed by `<len>` bytes of JPEG payload.

### Binary Payload
- Exactly `<len>` bytes (JPEG). The receiver **must read exactly `<len>` bytes**.
- Do **NOT** print the binary payload to terminal.

---

## 3. Current Components (Implemented)

### `collecting.py` (Data Enrollment)
Collects ROI frames from OpenMV and builds per-person JSON files:

- Receives ROI JPEG via OpenMV protocol
- Runs `facenet_512.tflite` → embedding
- Saves per-person:
  - `face_db_json/<person>.json`
- Supports:
  - sample count per person
  - capture interval throttling
  - dedup by cosine similarity threshold
  - optional ROI image saving for inspection

### `json_to_npz.py` (Database Builder)
Converts JSON enrollment files into a single NPZ database used by runtime recognition:

Outputs `face_db_embeddings.npz` with:
- `names` (array of person names)
- `embeddings` (float32 `[N,512]`)

Two modes:
- `mean` (recommended): one embedding per person (mean of samples + L2 normalize)
- `all`: keep all samples for each person (larger DB)

### `door_brain.py` (Runtime Brain)
- Receives OpenMV ROI frames
- Runs FaceNet inference
- Matches embeddings with cosine similarity threshold
- Uses:
  - `SIM_THRESHOLD`
  - `REQUIRE_CONSECUTIVE_PASS`
  - cooldown timers
- Terminal rolling panel (no spam)
- STM32 control loop:
  - send `CMD:OPEN`
  - wait for `ACK:DONE` / `ERR:*`

**STM32 is optional during early debugging**:
- `ENABLE_STM32=False` → recognition works, door action is simulated (`SIM_OPEN`)

---

## 4. Requirements

### OS / Python
- Ubuntu on Raspberry Pi
- Python 3.x (venv recommended)

### Python Packages
- `numpy`
- `pyserial`
- `opencv-python`
- `tflite-runtime` (preferred) or `tensorflow` (fallback)

Install:
```bash
pip install numpy pyserial opencv-python
pip install tflite-runtime
```

---

## 5. Setup and Usage (Raspberry Pi)

### 5.1 Check Serial Devices
```bash
ls /dev/ttyUSB* /dev/ttyACM* 2>/dev/null
```

Typical:
- OpenMV: `/dev/ttyACM0`
- STM32: `/dev/ttyUSB0` or `/dev/ttyACM1` (depends on board/USB-UART)

---

## 6. Step-by-Step Workflow

### Step 1 — Enrollment (Collect embeddings as JSON)

Collect samples for one person:
```bash
python collecting.py \
  --name Alice \
  --port /dev/ttyACM0 \
  --baud 115200 \
  --model facenet_512.tflite \
  --preprocess-mode minus1_to_1 \
  --target-samples 20 \
  --capture-interval-sec 0.6 \
  --dedup-sim-threshold 0.985 \
  --save-roi-images
```

Repeat for each person (`--name Bob`, etc.).

Notes:
- Move slightly: different angles, distances, expressions.
- Don’t stand perfectly still; otherwise dedup will skip too much.

---

### Step 2 — Build the Face Database (NPZ)

Recommended (one embedding per person):
```bash
python json_to_npz.py \
  --input-dir face_db_json \
  --output face_db_embeddings.npz \
  --mode mean \
  --required-preprocess-mode minus1_to_1 \
  --overwrite
```

---

### Step 3 — Run Runtime Recognition (`door_brain.py`)

#### Option A: Recognition-only (no STM32 yet)
Edit `door_brain.py` config:
- `ENABLE_STM32=False`
- `OPENMV_PORT=/dev/ttyACM0`
- `FACE_DB_NPZ=face_db_embeddings.npz`
- `PREPROCESS_MODE` must match your enrollment

Run:
```bash
python door_brain.py
```

When recognition passes, the panel shows a simulated action `SIM_OPEN`.

#### Option B: Full system (with STM32)
Edit config:
- `ENABLE_STM32=True`
- `STM32_PORT=/dev/ttyUSB0` (or actual device)
- Ensure STM32 firmware responds with:
  - `ACK:DONE`, `ERR:BUSY`, `ERR:JAM`

Run:
```bash
python door_brain.py
```

---

## 7. Key Parameters (Tuning Guide)

### Face Recognition
- `PREPROCESS_MODE`: must match FaceNet model expectation and your enrollment
  - `minus1_to_1` (common)
  - `zero_to_1`
  - `raw_0_255` (rare; sometimes quantized pipelines)
- `SIM_THRESHOLD` (default start: `0.60`)
  - Increase if false accepts
  - Decrease if always rejecting (only after confirming preprocess is correct)
- `REQUIRE_CONSECUTIVE_PASS` (default `2`)
  - Increase if ROI jitter causes instability
- Cooldowns:
  - `RECOG_COOLDOWN_SEC_OK`
  - `RECOG_COOLDOWN_SEC_FAIL`
  - `UNKNOWN_COOLDOWN_SEC`

### Enrollment
- `TARGET_SAMPLES`: 15–30 recommended
- `CAPTURE_INTERVAL_SEC`: 0.5–1.0 typical
- `DEDUP_SIM_THRESHOLD`: 0.98–0.99 typical

---

## 8. Common Troubleshooting

### `door_brain.py` fails to open STM32 port
Error: `No such file or directory: '/dev/ttyUSB0'`

Fix:
1. List devices:
   ```bash
   ls /dev/ttyUSB* /dev/ttyACM* 2>/dev/null
   ```
2. Set `STM32_PORT` to the actual device OR temporarily set `ENABLE_STM32=False`.

### JPEG decode failures / protocol desync
- Ensure OpenMV sends:
  - `IMG:<len>\n` then **exactly `<len>` bytes** of JPEG
- Avoid printing binary payload
- Check `OPENMV_MAX_JPEG_LEN` and actual ROI image sizes

### Recognition always low similarity
Most likely mismatch:
- `PREPROCESS_MODE` differs between enrollment and runtime
- RGB/BGR conversion mismatch (handled in scripts)
- Wrong FaceNet model file or inconsistent DB

---

## 9. Project Directory Layout (Suggested)

```text
.
├── door_brain.py
├── collecting.py
├── json_to_npz.py
├── facenet_512.tflite
├── face_db_json/
│   ├── Alice.json
│   └── Bob.json
├── face_db_embeddings.npz
└── face_db_roi_samples/            # optional
    ├── Alice/
    └── Bob/
```

---

## 10. Safety / Engineering Notes

- OpenMV only triggers and sends ROI; it does not store identities.
- Raspberry Pi handles all recognition and decision logic.
- STM32 executes actions with status feedback; add watchdog/timeout logic in firmware.
- Use cooldown + consecutive-pass to prevent repeated openings.

---

If you want, I can also add:
- a calibration script to automatically recommend `SIM_THRESHOLD` based on your collected JSON/NPZ,
- a small serial test utility for STM32 (`send CMD:OPEN` and verify responses),
- or a lightweight internal web dashboard (LAN-only) mirroring the rolling panel state.
