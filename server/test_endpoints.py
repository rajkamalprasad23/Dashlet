import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
from app import app, VIDEO_METADATA, SWIPE_DISTRIBUTIONS

client = app.test_client()

# 1. Test /api/videos
res = client.get('/api/videos')
assert res.status_code == 200, f"/api/videos failed: {res.status_code}"
data = res.get_json()
print("Videos endpoint OK. Total videos:", data['total_videos'])

# 2. Test /api/manifest
res = client.get('/api/manifest')
assert res.status_code == 200
print("Manifest endpoint OK.")

# 3. Test /api/swipe-distributions
res = client.get('/api/swipe-distributions')
assert res.status_code == 200
print("Swipe distributions endpoint OK.")

# 4. Test /api/abr with Dashlet
payload_dashlet = {
    "current_video_id": 0,
    "current_video_index": 0,
    "current_playback_position": 2.0,
    "buffered_chunks": {"0": [0]},
    "buffered_bitrates": {"0": {"0": 1200}},
    "throughput_estimate": 4000,
    "previous_bitrate": 1200,
    "algorithm": "dashlet"
}
res = client.post('/api/abr', json=payload_dashlet)
assert res.status_code == 200, f"Dashlet ABR failed: {res.status_code}"
dashlet_resp = res.get_json()
print("Dashlet ABR decision OK:")
print("  Algorithm:", dashlet_resp['algorithm'])
print("  Next chunk:", dashlet_resp['next_chunk'])
print("  Sequence prefix:", dashlet_resp['sequence'][:3])

# 5. Test /api/abr with TikTok
payload_tiktok = {
    "current_video_id": 0,
    "current_video_index": 0,
    "current_playback_position": 2.0,
    "buffered_chunks": {"0": [0]},
    "buffered_bitrates": {"0": {"0": 1200}},
    "throughput_estimate": 4000,
    "previous_bitrate": 1200,
    "algorithm": "tiktok"
}
res = client.post('/api/abr', json=payload_tiktok)
assert res.status_code == 200, f"TikTok ABR failed: {res.status_code}"
tiktok_resp = res.get_json()
print("TikTok ABR decision OK:")
print("  Algorithm:", tiktok_resp['algorithm'])
print("  Next chunk:", tiktok_resp['next_chunk'])
print("  Sequence prefix:", tiktok_resp['sequence'][:3])

print("\nALL BACKEND API TESTS PASSED!")
