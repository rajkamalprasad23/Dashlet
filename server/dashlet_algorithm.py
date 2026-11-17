"""
Dashlet Algorithm — Core ABR logic for short video streaming.

Implements the Dashlet algorithm from:
"Dashlet: Taming Swipe Uncertainty for Robust Short Video Streaming"
(Li, Xie, Netravali, Jamieson — SIGCOMM 2022 / arXiv:2204.12954)

Key components:
1. Swipe distribution modeling (bimodal: early-swipe + end-of-video) (§3)
2. Expected rebuffering computation via viewing sequence convolution (Eq. 5–11) (§4.1)
3. Greedy buffer sequence determination with equal-bitrate horizon and causality constraints (§4.2)
4. RobustMPC bitrate selection with intra-video smoothness and buffer tracking (Eq. 12) (§4.2.2, §6.1)
5. TikTok-baseline comparison accurately reflecting empirical findings (§2)
"""

import numpy as np
from itertools import product as iter_product

# ─── Constants ───────────────────────────────────────────────────────────────
CHUNK_DURATION = 5.0          # seconds per chunk (Section 6.1)
AVAILABLE_BITRATES = [300, 750, 1200, 2000, 4000]  # kbps
LOOKAHEAD_HORIZON = 25.0      # seconds (Section 4.2)
MPC_LOOKAHEAD_CHUNKS = 5      # MPC planning window (Section 4.2)
QOE_MU = 4.3                  # rebuffering penalty weight (Section 6.1, Eq. 12)
QOE_ETA = 1.0                 # smoothness penalty weight (Section 6.1, Eq. 12)
TIKTOK_BUFFER_TARGET = 5      # TikTok maintains 5 buffered first chunks (Section 2.2.1)
TIKTOK_MANIFEST_GROUP = 10    # TikTok manifest serves groups of 10 videos (Section 2.1)

# Cross-version numpy trapezoidal integration
_trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))


# ─── Swipe Distribution Modeling (Section 3) ─────────────────────────────────

def generate_swipe_distribution(video_duration, video_type="mixed", num_samples=100):
    """
    Generate a bimodal swipe probability distribution for a video.

    From Section 3 of the paper:
    - Users tend to swipe either early (first 20%) or at video end (last 20%)
    - 29% of swipes within first 20%, 42% within last 20% (MTurk data)
    - Different videos exhibit different patterns (Fig 7)

    Args:
        video_duration: total video duration in seconds
        video_type: "early_heavy", "end_heavy", "mixed", or "uniform"
        num_samples: number of time samples for the distribution

    Returns:
        dict with 'pdf', 'cdf', 'times' arrays
    """
    t = np.linspace(0, video_duration, num_samples)

    if video_type == "early_heavy":
        # Like Fig 7(c): ~60% swipes in first 20%
        early_weight, end_weight = 0.7, 0.3
        early_peak, end_peak = 0.15, 0.92
        early_std, end_std = 0.08, 0.06
    elif video_type == "end_heavy":
        # Like Fig 7(a)/(d): 60-80% swipes at end
        early_weight, end_weight = 0.2, 0.8
        early_peak, end_peak = 0.12, 0.95
        early_std, end_std = 0.06, 0.04
    elif video_type == "mixed":
        # Like Fig 7(b): more evenly distributed
        early_weight, end_weight = 0.4, 0.6
        early_peak, end_peak = 0.15, 0.90
        early_std, end_std = 0.10, 0.08
    else:  # uniform
        early_weight, end_weight = 0.5, 0.5
        early_peak, end_peak = 0.3, 0.7
        early_std, end_std = 0.20, 0.20

    t_norm = t / max(0.001, video_duration)
    early_mode = np.exp(-0.5 * ((t_norm - early_peak) / early_std) ** 2)
    end_mode = np.exp(-0.5 * ((t_norm - end_peak) / end_std) ** 2)

    pdf = early_weight * early_mode + end_weight * end_mode
    pdf += 0.02  # small baseline probability across video

    # Normalize to proper PDF
    area = _trapz(pdf, t)
    pdf = pdf / (area + 1e-9)

    # Compute CDF
    cdf = np.cumsum(pdf) * (t[1] - t[0]) if len(t) > 1 else np.array([1.0])
    cdf = np.clip(cdf / (cdf[-1] + 1e-9), 0.0, 1.0)

    return {
        'pdf': pdf.tolist(),
        'cdf': cdf.tolist(),
        'times': t.tolist(),
        'video_duration': video_duration
    }


def get_swipe_cdf_at_time(swipe_dist, time):
    """Interpolate the swipe CDF at a given time: P(swipe <= time)."""
    times = swipe_dist['times']
    cdf = swipe_dist['cdf']
    if time <= times[0]:
        return 0.0
    if time >= times[-1]:
        return 1.0
    return float(np.interp(time, times, cdf))


def get_survival_probability(swipe_dist, time):
    """P(user still watching at time) = 1 - CDF(time)."""
    return max(0.0, 1.0 - get_swipe_cdf_at_time(swipe_dist, time))


def discretize_swipe_distribution(swipe_dist, num_chunks, chunk_duration=CHUNK_DURATION):
    """
    Discretize swipe CDF into per-chunk swipe probability distribution P_{ki}.
    Section 4.1: P_{ki} = {p_{i1}, p_{i2}, ..., p_{iN_i}}.
    p_{ij} is the probability the user swipes during or at chunk j.
    """
    times = np.array(swipe_dist['times'])
    cdf = np.array(swipe_dist['cdf'])

    chunk_p = []
    for j in range(num_chunks):
        t_start = j * chunk_duration
        t_end = (j + 1) * chunk_duration
        f_start = 0.0 if t_start <= times[0] else (1.0 if t_start >= times[-1] else float(np.interp(t_start, times, cdf)))
        f_end = 0.0 if t_end <= times[0] else (1.0 if t_end >= times[-1] else float(np.interp(t_end, times, cdf)))
        chunk_p.append(max(0.0, f_end - f_start))

    chunk_p = np.array(chunk_p)
    total = np.sum(chunk_p)
    if total > 0:
        chunk_p = chunk_p / total
    else:
        chunk_p[-1] = 1.0
    return chunk_p


# ─── Expected Rebuffering Calculation (Section 4.1, Eq. 5–11) ────────────────

def compute_arrival_distributions(current_video_index, current_playback_pos,
                                  video_metadata, swipe_distributions,
                                  max_videos_ahead=7):
    """
    Computes viewing sequence distributions via recursive convolution (Eq. 5, 6).

    P_{Di1} = P_{D(i-1)1} * P_{ki-1}  (Eq. 6)

    Returns:
        arrival_dists: dict mapping vidx -> PMF array of chunks watched until vidx starts
        t_rem: remaining playback time in the current chunk of currently playing video
        current_conditional_p: conditional PMF of remaining chunks in currently playing video
        j_curr: chunk index currently being played
    """
    total_videos = len(video_metadata)
    if current_video_index >= total_videos:
        return {}, 0.0, np.array([1.0]), 0

    curr_vid = video_metadata[current_video_index]
    curr_dist = swipe_distributions.get(str(curr_vid['id']))
    curr_chunks = curr_vid.get('num_chunks', max(1, int(round(curr_vid['duration'] / CHUNK_DURATION))))

    j_curr = min(curr_chunks - 1, int(current_playback_pos // CHUNK_DURATION))
    t_rem = max(0.0, (j_curr + 1) * CHUNK_DURATION - current_playback_pos)

    # Conditional probability for remaining chunks in the current video
    # Given the user has watched up to current_playback_pos
    times = np.array(curr_dist['times'])
    cdf = np.array(curr_dist['cdf'])
    f_pos = 0.0 if current_playback_pos <= times[0] else (1.0 if current_playback_pos >= times[-1] else float(np.interp(current_playback_pos, times, cdf)))
    survival = max(1e-6, 1.0 - f_pos)

    rem_p = []
    for j in range(j_curr, curr_chunks):
        t_start = max(current_playback_pos, j * CHUNK_DURATION)
        t_end = (j + 1) * CHUNK_DURATION
        f_s = 0.0 if t_start <= times[0] else (1.0 if t_start >= times[-1] else float(np.interp(t_start, times, cdf)))
        f_e = 0.0 if t_end <= times[0] else (1.0 if t_end >= times[-1] else float(np.interp(t_end, times, cdf)))
        rem_p.append(max(0.0, f_e - f_s) / survival)

    rem_p = np.array(rem_p)
    if np.sum(rem_p) > 0:
        rem_p = rem_p / np.sum(rem_p)
    else:
        rem_p = np.array([1.0])

    arrival_dists = {}
    next_idx = current_video_index + 1
    if next_idx < total_videos:
        # Video current_video_index + 1 starts after remaining chunks of current video
        arrival_dists[next_idx] = rem_p
        prev_p = rem_p

        # For subsequent videos: convolution P_{Di1} = P_{D(i-1)1} * P_{ki-1} (Eq. 6)
        # Note: In video (vidx - 1), user watches k in {1, ..., N} chunks.
        # Placing probability of watching k chunks at index k ensures that convolution
        # directly maps index d to the exact total number of chunks watched!
        for vidx in range(next_idx + 1, min(total_videos, current_video_index + max_videos_ahead + 1)):
            prev_v = video_metadata[vidx - 1]
            prev_dist = swipe_distributions.get(str(prev_v['id']))
            p_k_raw = discretize_swipe_distribution(prev_dist, prev_v.get('num_chunks', 3))
            p_k = np.zeros(len(p_k_raw) + 1)
            p_k[1:] = p_k_raw  # index k represents k chunks watched (k >= 1)
            conv_p = np.convolve(prev_p, p_k)
            # Bound tail length for efficiency
            if len(conv_p) > 25:
                conv_p = conv_p[:25]
                conv_p = conv_p / np.sum(conv_p)
            arrival_dists[vidx] = conv_p
            prev_p = conv_p

    return arrival_dists, t_rem, rem_p, j_curr


def compute_chunk_expected_rebuffering(vidx, cidx, download_finish_time,
                                       current_video_index, current_playback_pos,
                                       video_metadata, swipe_distributions,
                                       arrival_dists, t_rem, current_conditional_p, j_curr):
    """
    Compute expected rebuffering delay for chunk (vidx, cidx) given download finish time tf (Eq. 7 / 11).

    For current video:
        E[rebuf] = P(reach chunk cidx) * max(0, tf - t_play)

    For future video:
        P_{Dij}[n] = P_{Di1}[n - (j-1)] * (1 - sum_{m=1}^{j-1} p_{im})  (Eq. 8)
        E[rebuf] = sum_n P_{Dij}[n] * max(0, tf - (t_rem + (n + cidx) * T))  (Eq. 7)
    """
    tf = download_finish_time

    if vidx == current_video_index:
        if cidx < j_curr:
            return 0.0  # Already played
        t_play = max(0.0, cidx * CHUNK_DURATION - current_playback_pos)
        offset = cidx - j_curr
        # Probability of reaching chunk cidx in current video
        reach_prob = float(np.sum(current_conditional_p[offset:])) if offset < len(current_conditional_p) else 0.0
        return reach_prob * max(0.0, tf - t_play)

    elif vidx > current_video_index:
        if vidx not in arrival_dists:
            return 0.0
        p_arrival = arrival_dists[vidx]
        vid = video_metadata[vidx]
        vid_dist = swipe_distributions.get(str(vid['id']))
        p_k = discretize_swipe_distribution(vid_dist, vid.get('num_chunks', 3))

        # Eq. 8: user must not swipe before chunk cidx in video vidx
        if cidx == 0:
            reach_prob_internal = 1.0
        else:
            reach_prob_internal = max(0.0, 1.0 - float(np.sum(p_k[:cidx])))

        if reach_prob_internal <= 1e-6:
            return 0.0

        m_vals = np.arange(len(p_arrival))
        t_play_vals = t_rem + (m_vals + cidx) * CHUNK_DURATION
        rebuf_vals = np.maximum(0.0, tf - t_play_vals)

        expected_rebuf = reach_prob_internal * float(np.sum(p_arrival * rebuf_vals))
        return expected_rebuf
    else:
        return 0.0


# ─── Buffer Sequence Determination (Section 4.2) ─────────────────────────────

def determine_buffer_sequence(candidate_chunks, swipe_distributions, throughput_estimate,
                              current_video_id, current_video_index, current_playback_pos,
                              buffered_chunks, video_metadata, horizon=LOOKAHEAD_HORIZON):
    """
    Determine the optimal chunk buffering sequence using Dashlet's greedy algorithm.

    Section 4.2:
    1. Filter candidates using minimum rebuffering threshold (1 / mu) at horizon end (§4.2.1).
    2. Temporarily assume an equal bitrate per chunk: the maximum bitrate that ensures
       all chunks in the list complete downloading before the horizon completes (§4.2.1).
    3. Partition horizon into slots of duration = (ChunkSize(R_temp) / throughput).
    4. Greedily assign chunks to slots based on largest marginal rebuffering increase
       if delayed by one slot, while strictly enforcing application playback constraints
       (cidx cannot be scheduled before cidx-1 is buffered or scheduled) (§1, §4.2.2).

    Args:
        candidate_chunks: list of (video_id, chunk_index, video_index) tuples
        swipe_distributions: dict of video_id -> swipe distribution
        throughput_estimate: estimated network throughput in kbps
        current_video_id: currently playing video ID
        current_video_index: index in playlist
        current_playback_pos: playback position in seconds
        buffered_chunks: dict of video_id -> set of chunk indices already buffered
        video_metadata: list of video metadata dicts
        horizon: lookahead horizon in seconds (default 25.0)

    Returns:
        ordered list of (video_id, chunk_index) to download
    """
    if not candidate_chunks or throughput_estimate <= 0:
        return []

    # Filter out already buffered chunks
    unbuffered = []
    for vid, cidx, vidx in candidate_chunks:
        vid_str = str(vid)
        if vid_str in buffered_chunks and cidx in buffered_chunks[vid_str]:
            continue
        unbuffered.append((vid, cidx, vidx))

    if not unbuffered:
        return []

    # Compute arrival distributions via convolution (Eq. 5, 6)
    arrival_dists, t_rem, cond_p, j_curr = compute_arrival_distributions(
        current_video_index, current_playback_pos, video_metadata, swipe_distributions
    )

    # Section 4.2.1: Filter by minimum rebuffering penalty threshold = 1 / mu
    threshold = 1.0 / QOE_MU  # ~0.2325
    candidates_with_rebuf = []
    for vid, cidx, vidx in unbuffered:
        rebuf_at_horizon = compute_chunk_expected_rebuffering(
            vidx, cidx, horizon,
            current_video_index, current_playback_pos,
            video_metadata, swipe_distributions,
            arrival_dists, t_rem, cond_p, j_curr
        )
        if rebuf_at_horizon >= threshold:
            candidates_with_rebuf.append((vid, cidx, vidx, rebuf_at_horizon))

    if not candidates_with_rebuf:
        # Fallback: take earliest unbuffered chunks of current and next video
        candidates_with_rebuf = [(vid, cidx, vidx, 0.5) for vid, cidx, vidx in unbuffered[:5]]

    # Section 4.2.1: Equal bitrate per chunk
    # "maximum bitrate which ensures that all chunks in the list will complete downloading before horizon completes"
    M = len(candidates_with_rebuf)
    selected_br = AVAILABLE_BITRATES[0]
    for br in reversed(AVAILABLE_BITRATES):
        total_dl = M * (br * CHUNK_DURATION) / throughput_estimate
        if total_dl <= horizon:
            selected_br = br
            break

    slot_duration = max(0.1, (selected_br * CHUNK_DURATION) / throughput_estimate)
    num_slots = max(1, min(M, int(np.ceil(horizon / slot_duration))))

    # Section 4.2.2: Greedy ordering by marginal penalty with playback causality constraints
    buffer_sequence = []
    scheduled_set = set()
    remaining = list(candidates_with_rebuf)

    for slot in range(num_slots):
        tf_current = (slot + 1) * slot_duration
        tf_next = (slot + 2) * slot_duration

        best_idx = None
        best_delta = -float('inf')

        for i, (vid, cidx, vidx, _) in enumerate(remaining):
            # Playback causality constraint (§1, §4):
            # Chunk cidx > 0 can ONLY be scheduled if cidx - 1 is already buffered or scheduled
            vid_str = str(vid)
            if cidx > 0:
                is_prev_buffered = (vid_str in buffered_chunks and (cidx - 1) in buffered_chunks[vid_str])
                is_prev_scheduled = ((vid, cidx - 1) in scheduled_set)
                if not (is_prev_buffered or is_prev_scheduled):
                    continue

            rebuf_curr = compute_chunk_expected_rebuffering(
                vidx, cidx, tf_current,
                current_video_index, current_playback_pos,
                video_metadata, swipe_distributions,
                arrival_dists, t_rem, cond_p, j_curr
            )
            rebuf_next = compute_chunk_expected_rebuffering(
                vidx, cidx, tf_next,
                current_video_index, current_playback_pos,
                video_metadata, swipe_distributions,
                arrival_dists, t_rem, cond_p, j_curr
            )

            delta = rebuf_next - rebuf_curr
            if delta > best_delta:
                best_delta = delta
                best_idx = i

        if best_idx is not None:
            chosen = remaining.pop(best_idx)
            buffer_sequence.append((chosen[0], chosen[1]))
            scheduled_set.add((chosen[0], chosen[1]))
        else:
            break

    return buffer_sequence


# ─── RobustMPC Bitrate Selection (Section 4.2.2, Section 6.1) ─────────────────

def select_bitrate_mpc(buffer_sequence, throughput_estimate, buffered_chunks,
                       buffered_bitrates, current_video_id, current_playback_pos,
                       mu=QOE_MU, eta=QOE_ETA):
    """
    RobustMPC-based bitrate selection for Dashlet's ordered buffer sequence.

    Optimizes Quality of Experience (QoE) (Eq. 12):
        QoE = R_bitrate - mu * P_rebuffer - eta * P_smooth

    Key rules from paper:
    1. Bitrates R in Mbps (0.3, 0.75, 1.2, 2.0, 4.0 Mbps).
    2. P_rebuffer is cumulative stall time in seconds.
    3. P_smooth applies ONLY across adjacent chunks within the same short video (§6.1, §6.2.1).
       Switching bitrates across different videos incurs ZERO smoothness penalty.
    4. Conservative throughput estimate (0.9x safety margin for RobustMPC).

    Args:
        buffer_sequence: ordered list of (video_id, chunk_index) to download
        throughput_estimate: throughput in kbps (harmonic mean of recent downloads)
        buffered_chunks: current buffer state dict {vid: set(chunk_indices)}
        buffered_bitrates: dict {str(vid): {cidx: bitrate}}
        current_video_id: currently playing video ID
        current_playback_pos: playback position in seconds
        mu: rebuffering penalty weight (4.3)
        eta: smoothness penalty weight (1.0)

    Returns:
        list of (video_id, chunk_index, bitrate) tuples
    """
    if not buffer_sequence:
        return []

    N = min(len(buffer_sequence), MPC_LOOKAHEAD_CHUNKS)
    sequence = buffer_sequence[:N]
    safe_tp = max(200.0, throughput_estimate * 0.9)

    # Initial buffer for currently playing video (seconds of playback available)
    curr_vid_str = str(current_video_id)
    curr_buffered = buffered_chunks.get(curr_vid_str, set())
    initial_buffer = max(0.0, len(curr_buffered) * CHUNK_DURATION - current_playback_pos)

    best_qoe = -float('inf')
    best_combo = None

    for combo in iter_product(AVAILABLE_BITRATES, repeat=N):
        total_bitrate = 0.0
        total_rebuf = 0.0
        total_smooth = 0.0
        b_curr = initial_buffer
        combo_bitrates = {}

        for i, br in enumerate(combo):
            vid, cidx = sequence[i]
            vid_str = str(vid)

            # Bitrate reward in Mbps (Section 6.1)
            total_bitrate += br / 1000.0

            # Download time in seconds
            t_dl = (br * CHUNK_DURATION) / safe_tp

            # Rebuffering stall on currently playing video
            rebuf = max(0.0, t_dl - b_curr)
            total_rebuf += rebuf

            # Buffer update
            if vid == current_video_id:
                b_curr = max(0.0, b_curr - t_dl) + CHUNK_DURATION
            else:
                b_curr = max(0.0, b_curr - t_dl)

            # Intra-video smoothness penalty (§6.1, §6.2.1):
            # Only between adjacent chunks of the same video
            prev_br = None
            if cidx > 0:
                if (vid, cidx - 1) in combo_bitrates:
                    prev_br = combo_bitrates[(vid, cidx - 1)]
                elif vid_str in buffered_bitrates and (cidx - 1) in buffered_bitrates[vid_str]:
                    prev_br = buffered_bitrates[vid_str][cidx - 1]

            if prev_br is not None:
                total_smooth += abs(br - prev_br) / 1000.0

            combo_bitrates[(vid, cidx)] = br

        qoe = total_bitrate - mu * total_rebuf - eta * total_smooth
        if qoe > best_qoe:
            best_qoe = qoe
            best_combo = combo

    result = []
    if best_combo is not None:
        for i, br in enumerate(best_combo):
            result.append((sequence[i][0], sequence[i][1], br))
    else:
        for vid, cidx in sequence:
            result.append((vid, cidx, AVAILABLE_BITRATES[0]))
    return result


# ─── TikTok Baseline Implementation (Section 2) ──────────────────────────────

def tiktok_baseline_sequence(current_video_index, current_playback_pos,
                             video_metadata, buffered_chunks, video_bound_bitrates,
                             throughput_estimate, is_playing=True):
    """
    TikTok-style baseline algorithm adhering to empirical findings in Section 2:

    1. Premature Bitrate Binding (§2.1, §2.2.2, Fig. 5):
       Bitrate adapts to throughput at chunk 0 download time and binds to the entire video.
    2. Buffer Control (§2.2.1):
       - Ramping-up / Maintaining: keeps 5 first-chunks of upcoming videos in buffer.
       - Chunk 2: Downloads second chunk when and only when the video starts playing!
       - NEVER pre-buffers second chunks of future videos (§2.2.3).
    3. Prebuffer-Idling State (§2.2.1):
       After buffering 10 first-chunks in manifest group, idles until video 9.

    Args:
        current_video_index: index of currently playing video
        current_playback_pos: playback position in current video
        video_metadata: list of video metadata dicts
        buffered_chunks: dict of video_id -> set of chunk indices
        video_bound_bitrates: dict of video_id -> bound bitrate
        throughput_estimate: throughput in kbps
        is_playing: whether playback is active

    Returns:
        ordered list of (video_id, chunk_index, bitrate) tuples
    """
    total_videos = len(video_metadata)
    sequence = []

    def get_or_bind_bitrate(vid_id):
        vid_str = str(vid_id)
        if vid_str in video_bound_bitrates:
            return video_bound_bitrates[vid_str]
        # TikTok adapts bitrate to throughput when starting first chunk (Fig. 5)
        chosen = AVAILABLE_BITRATES[0]
        for br in reversed(AVAILABLE_BITRATES):
            if br <= throughput_estimate * 0.85:
                chosen = br
                break
        video_bound_bitrates[vid_str] = chosen
        return chosen

    # Phase 1: Ensure current video has chunk 0
    if current_video_index < total_videos:
        curr_vid = video_metadata[current_video_index]
        curr_id_str = str(curr_vid['id'])
        curr_buffered = buffered_chunks.get(curr_id_str, set())
        if 0 not in curr_buffered:
            br = get_or_bind_bitrate(curr_vid['id'])
            sequence.append((curr_vid['id'], 0, br))

    # Phase 2: TikTok downloads second chunk when and ONLY when video starts playing! (§2.2.1)
    if is_playing and current_video_index < total_videos:
        curr_vid = video_metadata[current_video_index]
        curr_id_str = str(curr_vid['id'])
        curr_buffered = buffered_chunks.get(curr_id_str, set())
        num_chunks = curr_vid.get('num_chunks', 3)
        for cidx in range(1, num_chunks):
            if cidx not in curr_buffered:
                br = get_or_bind_bitrate(curr_vid['id'])
                sequence.append((curr_vid['id'], cidx, br))

    # Phase 3: Maintain 5 first-chunks of upcoming videos (Ramping-up / Maintaining)
    # Check Prebuffer-Idling State (§2.2.1): manifest groups of 10
    manifest_group_start = (current_video_index // TIKTOK_MANIFEST_GROUP) * TIKTOK_MANIFEST_GROUP
    manifest_group_end = min(total_videos, manifest_group_start + TIKTOK_MANIFEST_GROUP)

    first_chunks_buffered_in_group = sum(
        1 for vidx in range(manifest_group_start, manifest_group_end)
        if 0 in buffered_chunks.get(str(video_metadata[vidx]['id']), set())
    )

    # Prebuffer idling: stop prebuffering first chunks after all 10 are buffered until video 9
    is_prebuffer_idling = (first_chunks_buffered_in_group >= TIKTOK_MANIFEST_GROUP) and (current_video_index < manifest_group_start + 9)

    if not is_prebuffer_idling:
        for offset in range(1, TIKTOK_BUFFER_TARGET + 1):
            vidx = current_video_index + offset
            if vidx >= total_videos or vidx >= manifest_group_end:
                break
            vid = video_metadata[vidx]
            vid_id_str = str(vid['id'])
            vid_buffered = buffered_chunks.get(vid_id_str, set())
            if 0 not in vid_buffered:
                br = get_or_bind_bitrate(vid['id'])
                sequence.append((vid['id'], 0, br))

    return sequence


# ─── Top-Level ABR Decision Handlers ─────────────────────────────────────────

def dashlet_abr_decision(current_video_id, current_video_index, current_playback_pos,
                         buffered_chunks, throughput_estimate, previous_bitrate,
                         video_metadata, swipe_distributions, buffered_bitrates=None):
    """
    Top-level Dashlet ABR decision combining convolution-based buffer sequencing and MPC.
    """
    if buffered_bitrates is None:
        buffered_bitrates = {}

    candidates = []
    total_videos = len(video_metadata)

    # Consider current video and up to 7 upcoming videos (covers 25s horizon)
    for offset in range(min(8, total_videos - current_video_index)):
        vidx = current_video_index + offset
        vid = video_metadata[vidx]
        num_chunks = vid.get('num_chunks', 3)
        for cidx in range(num_chunks):
            candidates.append((vid['id'], cidx, vidx))

    # Step 1: Greedy buffer sequence (§4.2)
    buffer_seq = determine_buffer_sequence(
        candidates, swipe_distributions, throughput_estimate,
        current_video_id, current_video_index, current_playback_pos,
        buffered_chunks, video_metadata
    )

    if not buffer_seq:
        return {
            'algorithm': 'dashlet',
            'sequence': [],
            'next_chunk': None
        }

    # Step 2: RobustMPC bitrate selection (§4.2.2)
    bitrate_seq = select_bitrate_mpc(
        buffer_seq, throughput_estimate, buffered_chunks,
        buffered_bitrates, current_video_id, current_playback_pos
    )

    sequence = [{'video_id': vid, 'chunk_index': cidx, 'bitrate': br} for vid, cidx, br in bitrate_seq]
    return {
        'algorithm': 'dashlet',
        'sequence': sequence,
        'next_chunk': sequence[0] if sequence else None
    }


def tiktok_abr_decision(current_video_id, current_video_index, current_playback_pos,
                        buffered_chunks, throughput_estimate, previous_bitrate,
                        video_metadata, swipe_distributions, video_bound_bitrates=None):
    """
    TikTok-baseline ABR decision.
    """
    if video_bound_bitrates is None:
        video_bound_bitrates = {}

    tiktok_seq = tiktok_baseline_sequence(
        current_video_index, current_playback_pos,
        video_metadata, buffered_chunks, video_bound_bitrates,
        throughput_estimate
    )

    sequence = [{'video_id': vid, 'chunk_index': cidx, 'bitrate': br} for vid, cidx, br in tiktok_seq]
    return {
        'algorithm': 'tiktok',
        'sequence': sequence,
        'next_chunk': sequence[0] if sequence else None
    }
