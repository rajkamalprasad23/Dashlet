# Dashlet — Short Video Streaming QoE Prototype

> **Based on:** *"Dashlet: Taming Swipe Uncertainty for Robust Short Video Streaming"*  
> Li, Xie, Netravali, Jamieson — ACM SIGCOMM 2022 (arXiv: 2204.12954)

## Overview

This is a working prototype demonstrating Dashlet's core algorithm for optimizing Quality of Experience (QoE) in short video streaming applications like TikTok and YouTube Shorts.

### Key Features

- **Dashlet Algorithm**: Out-of-order chunk pre-buffering using swipe probability distributions
- **TikTok Baseline**: Sequential first-chunk buffering (5-video look-ahead) for comparison
- **Interactive Dashboard**: Real-time visualization of buffer states, QoE metrics, and download decisions
- **Network Simulation**: Multiple network profiles (Good LTE, Poor LTE, Variable, WiFi Mall)
- **Side-by-side Comparison**: Run Dashlet and TikTok baseline simultaneously

## Quick Start

```bash
# 1. Install dependencies
cd server
pip install -r requirements.txt

# 2. Run the server
python app.py

# 3. Open in browser
# http://localhost:5000
```

## Architecture

```
cnproject/
└── server/
    ├── app.py                  # Flask backend (API endpoints)
    ├── dashlet_algorithm.py    # Core Dashlet algorithm
    ├── requirements.txt        # Python dependencies
    └── static/
        ├── index.html          # Dashboard UI
        ├── style.css           # Dark glassmorphism theme
        └── app.js              # Simulation engine
```

## How It Works

### The Problem
In short video streaming, users can swipe away at any time, making it wasteful to download entire videos sequentially. Traditional ABR algorithms (designed for long-form video) buffer chunks in order, leading to:
1. **Wasted bandwidth**: Downloaded chunks never viewed after swipe
2. **Rebuffering**: Next video's chunks not ready when user swipes

### Dashlet's Solution

1. **Swipe Distribution Modeling** (§3): Models user swipe behavior as a bimodal distribution — users tend to swipe either early (first 20%) or at video end (last 20%).

2. **Expected Rebuffering Computation** (Eq. 7/11): For each candidate chunk, computes expected rebuffering weighted by the probability the user will actually reach that chunk:

   ```
   E[rebuf(c_ij)] = ∫ f_D(t) × max(0, download_time - t) dt
   ```

3. **Greedy Buffer Sequencing** (§4.2): Instead of downloading chunks sequentially, Dashlet greedily picks chunks that minimize total expected rebuffering over a 25-second lookahead horizon. This may download chunk 1 of video 3 before chunk 2 of video 1.

4. **MPC Bitrate Selection**: Uses Model Predictive Control to optimize:

   ```
   QoE = R_bitrate − 4.3 × P_rebuffer − 1.0 × P_smooth
   ```

### Dashboard Controls

| Control | Description |
|---------|-------------|
| **Algorithm** | Switch between Dashlet, TikTok baseline, or both |
| **Network** | Good LTE (8Mbps), Poor LTE (2Mbps), Variable, WiFi Mall |
| **Speed** | 1× to 10× simulation speed |
| **SWIPE** | Manually swipe to next video (also: Space key) |
| **Start/Pause** | Start or pause the simulation (also: Enter key) |

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/videos` | Video playlist metadata |
| GET | `/api/manifest` | DASH MPD-style manifest (JSON) |
| GET | `/api/swipe-distributions` | Per-video swipe PDF/CDF |
| POST | `/api/abr` | ABR decision (Dashlet or TikTok) |
| GET | `/api/chunk/<vid>/<idx>/<br>` | Simulated video chunk |
| GET/POST | `/api/network-profile` | Network profile management |

## Results (from the paper)

- Dashlet achieves **77–99%** of oracle system's QoE
- Outperforms TikTok by **43.9–45.1×**
- Reduces bytes wasted by **30%**
- Tolerant to 50% swipe distribution errors (QoE reduction <10%)

## Technology Stack

- **Backend**: Python 3 + Flask
- **Frontend**: Vanilla HTML/CSS/JS + Chart.js
- **Algorithm**: NumPy for numerical computation

## References

- [Dashlet Paper (arXiv)](https://arxiv.org/abs/2204.12954)
- [DASH Industry Forum](https://dashif.org/)
