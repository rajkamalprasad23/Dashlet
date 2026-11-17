"""
Dashlet Prototype — Flask Backend Server

Serves the Dashlet short-video streaming prototype, including:
- Video playlist metadata and DASH-style manifest
- ABR decision endpoints (Dashlet + TikTok baseline)
- Swipe distribution data
- Simulated video chunk delivery with network emulation
- Network profile management
"""

import os
import time
import random
import json
from flask import Flask, jsonify, request, send_from_directory
try:
    from flask_cors import CORS
except ImportError:
    CORS = None
import numpy as np

from dashlet_algorithm import (
    generate_swipe_distribution,
    dashlet_abr_decision,
    tiktok_abr_decision,
    AVAILABLE_BITRATES,
    CHUNK_DURATION
)

app = Flask(__name__, static_folder='static')
if CORS:
    CORS(app)
else:
    @app.after_request
    def add_cors_headers(response):
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
        response.headers['Access-Control-Allow-Methods'] = 'GET,PUT,POST,DELETE,OPTIONS'
        return response

# ─── Video Metadata ─────────────────────────────────────────────────────────

VIDEO_CATEGORIES = [
    "Dance", "Comedy", "Cooking", "Sports", "Travel",
    "Music", "Pets", "DIY", "Fashion", "Education"
]

SWIPE_TYPES = [
    "end_heavy", "early_heavy", "mixed", "end_heavy", "mixed",
    "early_heavy", "end_heavy", "mixed", "early_heavy", "end_heavy"
]


def generate_video_metadata():
    """Generate metadata for 10 short videos (matching TikTok content)."""
    videos = []
    for i in range(10):
        duration = random.uniform(8, 20)
        # Round to nearest chunk boundary
        num_chunks = max(2, round(duration / CHUNK_DURATION))
        actual_duration = num_chunks * CHUNK_DURATION

        videos.append({
            'id': i,
            'title': f"{VIDEO_CATEGORIES[i]} #{i + 1}",
            'category': VIDEO_CATEGORIES[i],
            'duration': actual_duration,
            'num_chunks': num_chunks,
            'chunk_duration': CHUNK_DURATION,
            'available_bitrates': AVAILABLE_BITRATES,
            'video_url': f'/static/videos/video_{i}.mp4',
            'swipe_type': SWIPE_TYPES[i]
        })
    return videos


# Generate once at startup (deterministic with seed)
random.seed(42)
VIDEO_METADATA = generate_video_metadata()

# Pre-generate swipe distributions
SWIPE_DISTRIBUTIONS = {}
for vid in VIDEO_METADATA:
    SWIPE_DISTRIBUTIONS[str(vid['id'])] = generate_swipe_distribution(
        vid['duration'], vid['swipe_type']
    )

# ─── Network Profiles ───────────────────────────────────────────────────────

NETWORK_PROFILES = {
    'good_lte': {
        'name': 'Good LTE',
        'description': '8 Mbps stable connection',
        'base_throughput': 8000,  # kbps
        'variation': 0.1,
        'pattern': 'stable'
    },
    'poor_lte': {
        'name': 'Poor LTE',
        'description': '2 Mbps with fluctuations',
        'base_throughput': 2000,
        'variation': 0.3,
        'pattern': 'stable'
    },
    'variable': {
        'name': 'Variable',
        'description': 'Oscillating 1-6 Mbps',
        'base_throughput': 3500,
        'variation': 0.7,
        'pattern': 'oscillating'
    },
    'wifi_mall': {
        'name': 'WiFi Mall',
        'description': 'Bursty WiFi (mall trace)',
        'base_throughput': 5000,
        'variation': 0.5,
        'pattern': 'bursty'
    }
}

current_profile = 'good_lte'


def get_current_throughput():
    """Get instantaneous throughput based on current profile."""
    profile = NETWORK_PROFILES[current_profile]
    base = profile['base_throughput']
    var = profile['variation']

    if profile['pattern'] == 'stable':
        noise = np.random.normal(0, var * base * 0.2)
        return max(200, base + noise)
    elif profile['pattern'] == 'oscillating':
        t = time.time() % 20  # 20-second cycle
        wave = np.sin(2 * np.pi * t / 20)
        return max(200, base + wave * var * base)
    elif profile['pattern'] == 'bursty':
        # Random bursts
        if random.random() < 0.15:
            return max(200, base * random.uniform(0.1, 0.3))
        return max(200, base * random.uniform(0.7, 1.3))
    return base


# ─── API Endpoints ───────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)


@app.route('/api/videos', methods=['GET'])
def get_videos():
    """Return video playlist metadata."""
    return jsonify({
        'videos': VIDEO_METADATA,
        'total_videos': len(VIDEO_METADATA),
        'chunk_duration': CHUNK_DURATION,
        'available_bitrates': AVAILABLE_BITRATES
    })


@app.route('/api/manifest', methods=['GET'])
def get_manifest():
    """DASH MPD-style manifest in JSON format."""
    periods = []
    for vid in VIDEO_METADATA:
        adaptation_sets = []
        for bitrate in AVAILABLE_BITRATES:
            segments = []
            for cidx in range(vid['num_chunks']):
                segments.append({
                    'url': f"/api/chunk/{vid['id']}/{cidx}/{bitrate}",
                    'duration': CHUNK_DURATION,
                    'size_bytes': int(bitrate * CHUNK_DURATION * 1000 / 8)  # bits to bytes
                })
            adaptation_sets.append({
                'bitrate': bitrate,
                'width': {300: 426, 750: 640, 1200: 854, 2000: 1280, 4000: 1920}[bitrate],
                'height': {300: 240, 750: 360, 1200: 480, 2000: 720, 4000: 1080}[bitrate],
                'segments': segments
            })
        periods.append({
            'id': vid['id'],
            'duration': vid['duration'],
            'adaptation_sets': adaptation_sets
        })

    return jsonify({
        'type': 'dynamic',
        'min_buffer_time': CHUNK_DURATION,
        'periods': periods
    })


@app.route('/api/swipe-distributions', methods=['GET'])
def get_swipe_distributions():
    """Return per-video swipe probability distributions."""
    # Downsample for network efficiency
    result = {}
    for vid_id, dist in SWIPE_DISTRIBUTIONS.items():
        step = max(1, len(dist['times']) // 20)
        result[vid_id] = {
            'pdf': dist['pdf'][::step],
            'cdf': dist['cdf'][::step],
            'times': dist['times'][::step],
            'video_duration': dist['video_duration']
        }
    return jsonify(result)


@app.route('/api/abr', methods=['POST'])
def abr_decision():
    """
    ABR decision endpoint.

    Request body:
    {
        "current_video_id": int,
        "current_video_index": int,
        "current_playback_position": float,
        "buffered_chunks": {"video_id": [chunk_indices]},
        "buffered_bitrates": {"video_id": {"chunk_index": bitrate}},
        "video_bound_bitrates": {"video_id": bitrate},
        "throughput_estimate": float (kbps),
        "previous_bitrate": int (kbps),
        "algorithm": "dashlet" | "tiktok"
    }

    Returns the next chunk(s) to download with bitrate assignments.
    """
    data = request.get_json() or {}

    current_video_id = data.get('current_video_id', 0)
    current_video_index = data.get('current_video_index', 0)
    current_playback_pos = data.get('current_playback_position', 0)
    throughput_estimate = data.get('throughput_estimate', 4000)
    previous_bitrate = data.get('previous_bitrate', 1200)
    algorithm = data.get('algorithm', 'dashlet')

    # Parse buffered_chunks: convert string keys to int sets
    raw_buffered = data.get('buffered_chunks', {})
    buffered_chunks = {}
    for vid_id, chunks in raw_buffered.items():
        if isinstance(chunks, list):
            buffered_chunks[str(vid_id)] = set(chunks)
        elif isinstance(chunks, dict):
            buffered_chunks[str(vid_id)] = set(int(k) for k in chunks.keys())
        else:
            buffered_chunks[str(vid_id)] = set()

    # Parse buffered_bitrates
    raw_bitrates = data.get('buffered_bitrates', {})
    buffered_bitrates = {}
    for vid_id, br_map in raw_bitrates.items():
        if isinstance(br_map, dict):
            buffered_bitrates[str(vid_id)] = {int(k): int(v) for k, v in br_map.items()}

    # TikTok video-level bound bitrates
    video_bound_bitrates = data.get('video_bound_bitrates', {})

    if algorithm == 'dashlet':
        result = dashlet_abr_decision(
            current_video_id, current_video_index, current_playback_pos,
            buffered_chunks, throughput_estimate, previous_bitrate,
            VIDEO_METADATA, SWIPE_DISTRIBUTIONS,
            buffered_bitrates=buffered_bitrates
        )
    else:
        result = tiktok_abr_decision(
            current_video_id, current_video_index, current_playback_pos,
            buffered_chunks, throughput_estimate, previous_bitrate,
            VIDEO_METADATA, SWIPE_DISTRIBUTIONS,
            video_bound_bitrates=video_bound_bitrates
        )

    return jsonify(result)


@app.route('/api/chunk/<int:video_id>/<int:chunk_index>/<int:bitrate>', methods=['GET'])
def get_chunk(video_id, chunk_index, bitrate):
    """
    Serve a simulated video chunk.
    Returns synthetic data whose size matches bitrate × chunk_duration.
    Applies artificial delay based on current network profile.
    """
    # Chunk size in bytes: bitrate (kbps) × duration (s) × 1000 / 8
    chunk_size = int(bitrate * CHUNK_DURATION * 1000 / 8)

    # Get current throughput and simulate download delay
    throughput = get_current_throughput()  # kbps
    download_time = (bitrate * CHUNK_DURATION) / throughput  # seconds

    # Cap delay for simulation speed (don't actually wait the full time)
    # Return the theoretical download time in headers instead
    actual_delay = min(download_time * 0.1, 0.5)  # Small real delay for realism
    time.sleep(actual_delay)

    # Generate synthetic chunk data
    data = os.urandom(min(chunk_size, 50000))  # Cap actual data size for efficiency

    from flask import Response
    response = Response(data, mimetype='application/octet-stream')
    response.headers['X-Chunk-Size'] = str(chunk_size)
    response.headers['X-Download-Time'] = f'{download_time:.3f}'
    response.headers['X-Throughput'] = f'{throughput:.0f}'
    response.headers['X-Video-Id'] = str(video_id)
    response.headers['X-Chunk-Index'] = str(chunk_index)
    response.headers['X-Bitrate'] = str(bitrate)
    response.headers['Access-Control-Expose-Headers'] = (
        'X-Chunk-Size, X-Download-Time, X-Throughput, X-Video-Id, X-Chunk-Index, X-Bitrate'
    )

    return response


@app.route('/api/network-profile', methods=['GET'])
def get_network_profile():
    """Return current network profile and available profiles."""
    return jsonify({
        'current': current_profile,
        'current_throughput': get_current_throughput(),
        'profiles': {
            k: {'name': v['name'], 'description': v['description'],
                 'base_throughput': v['base_throughput']}
            for k, v in NETWORK_PROFILES.items()
        }
    })


@app.route('/api/network-profile', methods=['POST'])
def set_network_profile():
    """Switch network profile."""
    global current_profile
    data = request.get_json()
    profile = data.get('profile', 'good_lte')

    if profile in NETWORK_PROFILES:
        current_profile = profile
        return jsonify({
            'status': 'ok',
            'profile': profile,
            'name': NETWORK_PROFILES[profile]['name']
        })
    else:
        return jsonify({'error': f'Unknown profile: {profile}'}), 400


@app.route('/api/qoe-formula', methods=['GET'])
def get_qoe_formula():
    """Return the QoE formula parameters for display."""
    return jsonify({
        'formula': 'QoE = R_bitrate - μ × P_rebuffer - η × P_smooth',
        'mu': 4.3,
        'eta': 1.0,
        'description': 'Eq. 12 from the Dashlet paper',
        'components': {
            'R_bitrate': 'Average video bitrate (reward)',
            'P_rebuffer': 'Cumulative rebuffering penalty',
            'P_smooth': 'Bitrate switching penalty'
        }
    })


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("  Dashlet Prototype Server")
    print("  Short Video Streaming QoE Optimization")
    print("=" * 60)
    print(f"  Videos: {len(VIDEO_METADATA)} short videos")
    print(f"  Bitrates: {AVAILABLE_BITRATES} kbps")
    print(f"  Chunk duration: {CHUNK_DURATION}s")
    print(f"  Network profiles: {list(NETWORK_PROFILES.keys())}")
    print("=" * 60)
    print("  Open http://localhost:5000 in your browser")
    print("=" * 60)

    app.run(host='0.0.0.0', port=5000, debug=True)
