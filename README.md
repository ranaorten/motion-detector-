# Motion Detector 🎯

Real-time motion detection using dual-validation: MOG2 Background Subtraction + Frame Differencing.

## How It Works

Two independent methods run simultaneously:
- **MOG2**: Learns the background over time, detects deviations
- **Frame Diff**: Compares consecutive frames for fast movement

| Confidence | Condition | Color |
|---|---|---|
| CRITICAL | Both methods triggered | Red |
| SUSPICIOUS | Only one triggered | Orange |
| CLEAN | Neither triggered | — |

## Installation

\`\`\`bash
pip install -r requirements.txt
python main.py
\`\`\`

## Parameters (config.py)

| Parameter | Default | Description |
|---|---|---|
| CAMERA_INDEX | 1 | Camera source |
| MOG2_VAR_THRESHOLD | 80 | Sensitivity |
| MIN_CONTOUR_AREA | 1500 | Min motion area |

## Use Cases
- Perimeter security
- UAV/drone detection
- Restricted zone monitoring
