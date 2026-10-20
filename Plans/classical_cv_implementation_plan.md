# AutoSchlieren: Computer Vision-Guided Alignment (Classical CV)

This document outlines the architecture and implementation plan for fully automating the X-axis light alignment of your Schlieren imaging system using Classical Computer Vision (OpenCV) and the Duet 2 controller.

## 1. System Architecture Overview

The system utilizes a "Brain and Muscle" architecture. Your PC acts as the brain, processing the camera feed and running the optimization algorithm. The Duet 2 board acts as the muscle, receiving movement commands over the local network.

1. **Camera Feed (PC):** Captures video using the Arducam IMX179 8MP.
2. **Vision Metric (PC):** Uses Classical OpenCV to calculate a Two-Stage Quality Score (Coarse Heat detection, Fine Acoustic detection).
3. **1D Optimization Controller (PC):** A contracting multi-pass sweep state machine that steps the X-axis motor forward and backward to isolate the acoustic quality score.
4. **Hardware Interface (Duet 2):** The PC sends G-Code over Wi-Fi via HTTP requests to the Duet Web Control API to move the X-axis.

---

## 2. Implementation Phases

### Phase 1: Hardware Communication (Duet 2 Wi-Fi)
Python module (`duet_controller.py`) using the `requests` library to interface with the Duet 2's REST API.
- **Functionality:** 
  - `move_x_absolute(pos)` and `move_x_relative(dist)`: Send HTTP requests to execute G0 movements.
  - `wait_for_move()`: Poll status with M400 / rr_status until completion.
  - `get_position()`: Parse live XYZ coordinates.

### Phase 2: The Classical OpenCV Quality Metric
- **Temporal Averaging:** Rolling average across frames to smooth rapid thermal convection plumes.
- **Acoustic Chamber ROI:** Isolate the open acoustic chamber between the transducer towers and below the crossbar.
- **Scratch Neutralization:** Zero out the central vertical scratch column in the gradient tensor.
- **Sobel-X / Laplacian Contrast:** Quantify periodic standing wave fringe contrast.

### Phase 3: Contracting Multi-Pass Sweep (Bracket Zoom)
- **Pass 1 (Forward Coarse):** Broad sweep (±0.60mm, 0.05mm steps).
- **Pass 2 (Reverse Fine):** Narrowed sweep around Pass 1 peak (±0.16mm, 0.02mm steps).
- **Pass 3 (Micro Lock):** Tight sweep (±0.05mm, 0.01mm steps) to park at global optimum.

### Phase 4: Manual Debugger & Verification
- `manual_calibration.py`: Interactive debugger with live continuous HUD, adjustable step sizes (0.01mm to 0.10mm), and hotkey jogging.
