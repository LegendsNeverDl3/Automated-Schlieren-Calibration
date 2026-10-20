# Spatio-Temporal ML Schlieren Wave Detection & Auto-Alignment

This document establishes the architecture, systematic data acquisition strategy, and phased implementation plan for developing a **Spatio-Temporal Machine Learning model** to detect acoustic standing waves and optimize Schlieren knife-edge cutoff.

---

## 1. Problem Definition & The Power of the Time Dimension

### Why Single-Frame ML Struggles
In a single static snapshot, turbulent thermal convection currents from transducer heating can produce sharp refractive gradients that momentarily resemble standing waves.

### The Physics of the Time Component
| Phenomenon | Spatial Characteristic | Temporal Characteristic |
| :--- | :--- | :--- |
| **Acoustic Standing Waves** | High spatial frequency (periodic vertical fringes / antinodes) | **Temporally Stationary** (fixed in space over time) |
| **Thermal Convection Plumes** | Chaotic, low-to-medium spatial frequency | **Highly Dynamic & Non-Stationary** (fluctuating continuously) |
| **Hardware / Crossbar** | Sharp, high-contrast structural edges | **Completely Static** (zero temporal variation) |

---

## 2. Systematic Data Acquisition Strategy (Addressing Data Limitations)

Existing archival videos are heterogeneous (some show detuned cutoff, some vary beam-splitter light, some show stage shifts). Rather than relying on unstructured video clips, we implement a **structured 3-tier dataset pipeline**:

### Tier 1: Automated Ground-Truth Sweeps (`data_collector_sweep.py`)
- The script controls Duet 2 to sweep across the operational corridor ($X = 48.60$ to $50.20$ mm in steps of $0.02$ mm).
- At each exact coordinate, it pauses and records a **1.0-second video burst (30 frames)** tagged with:
  - Exact motor coordinate $X$
  - Timestamp
  - Classical reference score
- Labeling: The user confirms the visual focal sweet spot $X^*$. Every burst is assigned a continuous ground-truth target score:
  $$Q(X) = \exp\left(-\frac{(X - X^*)^2}{2\sigma^2}\right)$$
  *(where $Q = 1.0$ at the sweet spot, smoothly tapering to $0.0$ when detuned).*

### Tier 2: Multi-Condition Robustness Captures
- **Beam-Splitter Light Variations:** We run sweeps under 3 lighting levels (dim, normal, bright) so the neural network learns to detect wave contrast regardless of raw light intensity.
- **Transducer OFF Captures:** Systematic sweeps with sound power OFF provide true negative baselines (thermal plumes only, zero acoustic waves).

### Tier 3: Archival Hard Negatives (Existing Videos)
- Existing videos in `schlieren_cv_videos/` are indexed into explicit training buckets:
  - `hard_negative_washed_out` (beam splitter overloaded)
  - `hard_negative_off_axis` (stage shifted too far left/right)
  - `hard_negative_no_waves` (detuned cutoff)
  This forces the model to reject non-wave artifacts.

---

## 3. Spatio-Temporal Model Architecture

### Primary Architecture: `SpatioTemporalWaveNet`
- **Input:** Sequence tensor $X \in \mathbb{R}^{T \times 1 \times 128 \times 128}$ ($T = 6$ consecutive frames spanning ~200 ms).
- **Early-Fusion Temporal Block:**
  - 3D Conv or 2D Depthwise-Separable Convolutions cross-correlating across time.
  - Learns spatio-temporal filters that reward features with high spatial frequency but zero phase shift across time (standing waves).
  - Strongly penalizes features with high frame-to-frame variance (turbulent thermal plumes).
- **Dual Prediction Heads:**
  1. **Quality Head ($Q \in [0.0, 1.0]$):** Continuous wave sharpness score.
  2. **Regime Head:** 3-class classifier (`0: Under-Cutoff`, `1: Optimal Cutoff`, `2: Over-Cutoff`).
- **Inference Latency:** $< 18\text{ ms}$ on local CPU.

---

## 4. Directory Layout (`schlieren_ml/`)

```text
Automated Schlieren/
├── classical_cv_implementation_plan.md
├── ml_implementation_plan.md
├── auto_schlieren.py
├── manual_calibration.py
├── duet_controller.py
└── schlieren_ml/                                <-- DEDICATED ML FOLDER
    ├── dataset/
    │   ├── raw_sweeps/                          (Timestamped sweep folders)
    │   ├── sequences/                           (Extracted 6-frame clips: .npy / .pt)
    │   └── dataset_manifest.csv                 (Labels, motor coordinates, regime)
    ├── models/                                  (Trained weights: best_model.pth)
    ├── data_collector_sweep.py                  (Automated hardware-in-the-loop collector)
    ├── ingest_archival_videos.py                (Extracts hard negatives from old mp4s)
    ├── temporal_dataset.py                      (PyTorch Dataset with temporal augmentations)
    ├── model.py                                 (SpatioTemporalWaveNet architecture)
    ├── train.py                                 (Training loop, validation curves, MSE/CE loss)
    └── ml_auto_schlieren.py                     (Live closed-loop alignment controller)
```

---

## 5. Phased Implementation Plan

### Phase 1: Data Infrastructure & Automated Collector
- Create `schlieren_ml/` folder hierarchy.
- Build `data_collector_sweep.py`:
  - Performs automated motor sweeps via Duet 2.
  - Records synchronized 30-frame video bursts at each 0.02 mm step.
  - Automatically crops the Acoustic Chamber ROI and writes `dataset_manifest.csv`.
- Build `ingest_archival_videos.py`:
  - Slices existing MP4s from `schlieren_cv_videos/` into labeled hard-negative clips.

### Phase 2: PyTorch Dataset & Ground-Truth Annotation
- Build `temporal_dataset.py`:
  - Formats sequences into $(T=6, 1, 128, 128)$ tensors.
  - Implements temporal augmentations (random frame skips, reverse time order, illumination scaling).
  - Train/Validation split stratified across different sweep runs and lighting conditions.

### Phase 3: Model Architecture & Training Pipeline
- Build `model.py` containing `SpatioTemporalWaveNet`.
- Build `train.py`:
  - Combined loss: $\mathcal{L} = \text{MSE}(Q, \hat{Q}) + \lambda \text{CrossEntropy}(\text{regime}, \hat{\text{regime}})$.
  - Exports checkpoint to `schlieren_ml/models/best_model.pth`.

### Phase 4: Live Deployment & Closed-Loop Alignment
- Build `ml_auto_schlieren.py`:
  - Feeds live camera frames into a rolling circular buffer (`deque(maxlen=6)`).
  - Evaluates the live Spatio-Temporal ML score in real time (<18 ms).
  - Drives the multi-pass contracting sweep to park the motor at the global ML peak.
