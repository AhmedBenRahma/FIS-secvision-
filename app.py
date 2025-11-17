# app.py
import streamlit as st
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
import os, json, csv, datetime, time
from collections import defaultdict
import tempfile
import pandas as pd
import matplotlib.pyplot as plt

# ---------- GLOBAL CLIP STATE ----------
clip_writer = None
clip_start_time = None
clip_path = None

# ---------- THEME & COLORS ----------
st.set_page_config(page_title="Fis Vision ", layout="wide")

# Custom CSS for Colors
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    }
    .stApp {
        background: transparent;
    }
    h1 {
        color: #00ff88 !important;
        text-align: center;
        font-weight: 900;
        text-shadow: 0 0 10px #00ff88;
        font-size: 3.5rem !important;
    }
    .stButton>button {
        background: linear-gradient(45deg, #ff416c, #ff4b2b);
        color: white;
        border: none;
        padding: 12px 30px;
        font-weight: bold;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(255, 65, 108, 0.4);
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(255, 65, 108, 0.6);
    }
    .stTextInput>div>div>input {
        background-color: #1e1e2e;
        color: #00ff88;
        border: 2px solid #00ff88;
        border-radius: 10px;
    }
    .stFileUploader>div>div {
        background: #1e1e2e;
        border: 2px dashed #00ff88;
        border-radius: 12px;
    }
    .event-alert {
        background: linear-gradient(45deg, #ff0844, #ffb199);
        padding: 10px;
        border-radius: 10px;
        color: white;
        font-weight: bold;
        text-align: center;
        margin: 10px 0;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(255, 8, 68, 0.7); }
        70% { box-shadow: 0 0 0 15px rgba(255, 8, 68, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 8, 68, 0); }
    }
</style>
""", unsafe_allow_html=True)

# ---------- TITLE ----------
st.markdown("<h1> FIS SECURITY </h1>", unsafe_allow_html=True)

# --- ROI INPUT (GREEN ZONE) ---
default_roi = "850,350 10,550 10,1400 2700,1400 2700,700"
roi_input = st.text_input("**GREEN ZONE** – Enter points (x,y)", default_roi, help="Format: x1,y1 x2,y2 x3,y3 ...")

try:
    pts = [list(map(int, p.split(','))) for p in roi_input.split()]
    if len(pts) < 3:
        raise ValueError()
    ZONE_POLY = np.array(pts, dtype=np.int32)
except:
    st.error("Invalid ROI – using default.")
    ZONE_POLY = np.array([[100,200],[400,200],[400,500],[100,500]], np.int32)

# --- VIDEO UPLOAD ---
video_file = st.file_uploader("**Upload Video (MP4)**", type=["mp4"])

if video_file and st.button("RUN DETECTION", key="run"):
    with st.spinner("Loading video..."):
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(video_file.read())
        tfile.close()
        VIDEO_PATH = tfile.name

    # ---------- LIVE UI ----------
    live_placeholder = st.empty()
    progress_bar = st.progress(0)
    status_text = st.empty()

    # ---------- CONFIG ----------
    OUTPUT_DIR = "analytics"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    CLOSE_DIST = 80
    CLOSE_TIME = 3.0
    RUN_SPEED = 120
    OCCLUSION_GRACE_PERIOD = 3.0

    # ---------- LOGGING ----------
    log_file = os.path.join(OUTPUT_DIR, "events.csv")
    with open(log_file, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["timestamp", "event", "details"])

    def log_event(event, details=""):
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([ts, event, details])

    # ---------- MODEL ----------
    model = YOLO("yolov8n.pt")
    tracker = sv.ByteTrack()
    box_annotator = sv.BoxAnnotator(thickness=2, color=sv.Color.from_hex("#00ff88"))
    label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.6, color=sv.Color.from_hex("#00ff88"))

    # ---------- ZONE ----------
    zone = sv.PolygonZone(polygon=ZONE_POLY)
    zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.from_hex("#00ff88"), thickness=4)

    # ---------- STATE ----------
    clients_present = {}
    total_zone_entries = 0
    path_history = defaultdict(list)
    close_pairs = {}
    robbery_alerted = set()
    finished_durations = []
    final_stats = {}
    frame_timestamps = []
    people_in_zone_over_time = []

    # ---------- VIDEO SETUP ----------
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        st.error("Cannot open video.")
        st.stop()
    ret, first_frame = cap.read()
    if not ret:
        st.error("Cannot read first frame.")
        st.stop()

    h, w = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS) or 20
    out_full_path = os.path.join(OUTPUT_DIR, "output_full.mp4")
    out_full = cv2.VideoWriter(out_full_path, fourcc, fps, (w, h))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # ---------- CLIP HELPERS ----------
    def start_clip():
        global clip_writer, clip_start_time, clip_path
        if clip_writer is not None: return
        clip_start_time = time.time()
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        clip_path = os.path.join(OUTPUT_DIR, f"robbery_{ts}.mp4")
        clip_writer = cv2.VideoWriter(clip_path, fourcc, fps, (w, h))
        log_event("CLIP_START", os.path.basename(clip_path))

    def stop_clip():
        global clip_writer, clip_path
        if clip_writer is not None:
            clip_writer.release()
            log_event("CLIP_SAVED", os.path.basename(clip_path))
            clip_writer = None
            clip_path = None

    # ---------- MAIN LOOP ----------
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1
        t_now = time.time()
        elapsed_video = frame_idx / fps

        # --- DETECTION ---
        results = model(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)

        # Keep only person
        try:
            mask_person = np.array(detections.class_id) == 0
            detections = detections[mask_person]
        except: pass

        # --- ZONE FILTER ---
        if len(detections) > 0:
            mask_in_zone = zone.trigger(detections=detections)
            detections = detections[mask_in_zone]

        # Track
        detections = tracker.update_with_detections(detections)
        current_ids = set(detections.tracker_id) if detections.tracker_id is not None else set()

        # Centroids
        centroids = {}
        if len(detections) > 0:
            for i, tid in enumerate(detections.tracker_id):
                x1, y1, x2, y2 = detections.xyxy[i]
                centroids[tid] = np.array([(x1 + x2) / 2, (y1 + y2) / 2])

        # --- ENTRY / EXIT ---
        for tid in current_ids:
            if tid not in clients_present:
                clients_present[tid] = {"entry_time": t_now, "last_seen_time": t_now}
                total_zone_entries += 1
                log_event("ZONE_ENTER", f"ID {tid}")
            else:
                clients_present[tid]["last_seen_time"] = t_now

        for tid in list(clients_present):
            if tid not in current_ids:
                if t_now - clients_present[tid]["last_seen_time"] > OCCLUSION_GRACE_PERIOD:
                    dwell = clients_present[tid]["last_seen_time"] - clients_present[tid]["entry_time"]
                    finished_durations.append(dwell)
                    log_event("ZONE_EXIT", f"ID {tid} ({dwell:.1f}s)")
                    final_stats[int(tid)] = {
                        "entry": datetime.datetime.fromtimestamp(clients_present[tid]["entry_time"]).isoformat(),
                        "exit": datetime.datetime.fromtimestamp(clients_present[tid]["last_seen_time"]).isoformat(),
                        "duration_sec": dwell,
                        "path": path_history.get(tid, [])
                    }
                    del clients_present[tid]

        # --- ROBBERY ---
        ids_list = sorted(current_ids)
        for i, id1 in enumerate(ids_list):
            for id2 in ids_list[i+1:]:
                if id1 not in centroids or id2 not in centroids: continue
                dist = np.linalg.norm(centroids[id1] - centroids[id2])
                key = tuple(sorted([int(id1), int(id2)]))
                if dist < CLOSE_DIST:
                    if key not in close_pairs:
                        close_pairs[key] = {"start": t_now, "centroids": []}
                    close_pairs[key]["centroids"].append((centroids[id1].copy(), centroids[id2].copy()))
                else:
                    close_pairs.pop(key, None)

        for key, data in list(close_pairs.items()):
            if t_now - data["start"] >= CLOSE_TIME and key not in robbery_alerted and len(data["centroids"]) >= 2:
                c1p, c2p = data["centroids"][-2]
                c1c, c2c = data["centroids"][-1]
                s1 = np.linalg.norm(c1c - c1p) * fps
                s2 = np.linalg.norm(c2c - c2p) * fps
                if s1 > RUN_SPEED or s2 > RUN_SPEED:
                    id1, id2 = key
                    log_event("ROBBERY_ALERT", f"IDs {id1}-{id2}")
                    robbery_alerted.add(key)
                    start_clip()

        # --- TRAJECTORY ---
        for tid, c in centroids.items():
            path_history[tid].append(c.tolist())
            if len(path_history[tid]) > 50:
                path_history[tid].pop(0)

        # --- RECORD FOR GRAPH ---
        frame_timestamps.append(elapsed_video)
        people_in_zone_over_time.append(len(clients_present))

        # --- ANNOTATIONS ---
        labels = [f"ID:{tid} ({t_now - clients_present.get(tid, {'entry_time': t_now})['entry_time']:.0f}s)" 
                 for tid in (detections.tracker_id if detections.tracker_id is not None else [])]

        annotated = frame.copy()
        annotated = box_annotator.annotate(annotated, detections)
        annotated = label_annotator.annotate(annotated, detections, labels=labels)

        # Robbery Alert
        for key in robbery_alerted:
            id1, id2 = key
            if id1 in centroids and id2 in centroids:
                cv2.line(annotated, tuple(centroids[id1].astype(int)), tuple(centroids[id2].astype(int)), (0,0,255), 5)
                cv2.putText(annotated, "ROBBERY!", (50,200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 4)

        # Green Zone + Stats
        annotated = zone_annotator.annotate(annotated)
        cv2.putText(annotated, f"People: {len(clients_present)}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
        cv2.putText(annotated, f"Entries: {total_zone_entries}", (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
        if finished_durations:
            cv2.putText(annotated, f"Avg: {np.mean(finished_durations):.1f}s", (50,150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,0), 3)

        # Trajectories
        for pts in path_history.values():
            for a, b in zip(pts[:-1], pts[1:]):
                cv2.line(annotated, tuple(map(int,a)), tuple(map(int,b)), (255,255,0), 3)

        # --- WRITE ---
        if clip_writer is not None:
            clip_writer.write(annotated)
            if time.time() - clip_start_time > 5:
                stop_clip()
        out_full.write(annotated)

        # --- LIVE DISPLAY ---
        if frame_idx % 5 == 0:
            frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            live_placeholder.image(frame_rgb, use_container_width=True)
            progress = min(frame_idx / frame_count, 1.0)
            progress_bar.progress(progress)
            status_text.markdown(f"<p style='color:#00ff88; font-weight:bold;'>Frame {frame_idx}/{frame_count} – {progress*100:.1f}%</p>", unsafe_allow_html=True)

    # ---------- CLEANUP ----------
    out_full.release()
    cap.release()
    if clip_writer is not None: clip_writer.release()
    os.unlink(VIDEO_PATH)

    # ---------- SAVE STATS ----------
    for tid, data in clients_present.items():
        final_stats[int(tid)] = {
            "entry": datetime.datetime.fromtimestamp(data["entry_time"]).isoformat(),
            "exit": None,
            "duration_sec": None,
            "path": path_history.get(tid, [])
        }
    final_stats = {int(k): v for k, v in final_stats.items()}
    with open(os.path.join(OUTPUT_DIR, "persons.json"), "w") as f:
        json.dump(final_stats, f, indent=2, default=str)

    # ---------- FINAL UI ----------
    live_placeholder.empty()
    progress_bar.empty()
    status_text.empty()

    st.success("Processing Complete!")
    st.video(out_full_path)

    # --- ALERT IF ROBBERY ---
    if any("ROBBERY" in e for e in pd.read_csv(log_file)['event']):
        st.markdown("<div class='event-alert'>ROBBERY DETECTED!</div>", unsafe_allow_html=True)

    # --- EVENTS TABLE ---
    st.subheader("Events Log")
    df = pd.read_csv(log_file)
    st.dataframe(df.style.set_properties(**{'background-color': '#1e1e2e', 'color': '#00ff88'}), use_container_width=True)
    st.download_button("Download events.csv", open(log_file).read(), "events.csv")

    # --- GRAPHS WITH COLORS ---
    st.subheader("Analytics Dashboard")
    col1, col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots(figsize=(6,4), facecolor='#1e1e2e')
        ax1.plot(frame_timestamps, people_in_zone_over_time, color='#00ff88', linewidth=3)
        ax1.set_title("People in Zone", color='white')
        ax1.set_xlabel("Time (s)", color='white')
        ax1.set_ylabel("Count", color='white')
        ax1.tick_params(colors='white')
        ax1.grid(True, alpha=0.3, color='#00ff88')
        ax1.set_facecolor('#0f0c29')
        st.pyplot(fig1, use_container_width=True)

    with col2:
        event_counts = df['event'].value_counts()
        fig2, ax2 = plt.subplots(figsize=(6,4), facecolor='#1e1e2e')
        colors = ['#ff416c', '#00ff88', '#ffb199', '#4ecdc4']
        event_counts.plot(kind='bar', ax=ax2, color=colors[:len(event_counts)])
        ax2.set_title("Event Frequency", color='white')
        ax2.set_ylabel("Count", color='white')
        ax2.tick_params(colors='white', axis='x', rotation=0)
        ax2.set_facecolor('#0f0c29')
        st.pyplot(fig2, use_container_width=True)

    # --- DOWNLOADS ---
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Person Stats (JSON)**")
        with open(os.path.join(OUTPUT_DIR, "persons.json")) as f: st.json(json.load(f))
        st.download_button("Download JSON", open(os.path.join(OUTPUT_DIR, "persons.json")).read(), "persons.json")
    with col2:
        if clip_path and os.path.exists(clip_path):
            st.write("**Robbery Alert Clip**")
            st.video(clip_path)
            st.download_button("Download Clip", open(clip_path, "rb").read(), os.path.basename(clip_path))
