# === streamlit_pose_viewer.py (fixed session state initialization) ===

# streamlit run "/Users/Christian/Christian Home Drive/Christian/Projekte/CURRENTLY RUNNING PROJECTS/CV and NLP/Python_codes and apps/streamlit_pose_viewer.py"

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import time
from pathlib import Path
import atexit
from scipy.signal import butter, filtfilt
from scipy.stats import linregress
from scipy.signal import find_peaks
from scipy.ndimage import label 
import base64



# === SETTINGS ===
VIDEO_INPUT_FOLDER = Path("/Users/Christian/Downloads/test")
FINAL_DF_INPUT_FOLDER = VIDEO_INPUT_FOLDER / "corrected"
CACHE_FOLDER = Path("cached_videos"); CACHE_FOLDER.mkdir(exist_ok=True)
ONLY_SHOW_HIGH_IDS = False
MIN_ID_THRESHOLD = 2000
LIKELIHOOD_THRESHOLD = 0.5
FRAME_STRIDE = 1


# PARAMETERS:
# For javelin leaves the hand: 
WHEN_START_MS = 800 # in milliseconds
MIN_DELTA = 150
MIN_FRACTION = 0.55
MIN_SUSTAIN_SEC = 0.25  # duration in seconds


THRESHOLD = 10  # how big should the rise be?


# Detecting peaks and troughs for foot-strike:
PROMINENCE = 3
DISTANCE_MS = 300
        

# Stride length: 
STEP_RATIO = 1

# For Footstrike: 
MAX_SHIFT_MS = 400  # adjust if needed
Y_PROMINENCE = 1  # controls how "sharp" a trough must be


# Parameters for if arm is outstretched during the run-up: 
ONSET_OFFSET_MS = 150
OFFSET_BEFORE_RELEASE_MS = 100
WINDOW_BEFORE_RELEASE_MS = 2500
REQUIRED_ABOVE_ANGLE_MS = 150
THRESHOLD_ANGLE = 150
MIN_MAX_DURATION_FRAMES = 2


# Parameters for stopping:
STOP_WIN_MS = 300
STOP_RATIO_THRESHOLD = 0.55

time_window_ms = 400  # Analyze steps at xx ms after release (change as needed)






# Global plot settings:
plt.rcParams["font.size"] = 8
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.linewidth"] = 0.6
plt.rcParams["lines.linewidth"] = 0.8
plt.rcParams["lines.antialiased"] = False
plt.rcParams["text.antialiased"] = False




# === INIT SESSION STATE ===
if "play" not in st.session_state:
    st.session_state.play = False
if "frame_idx" not in st.session_state:
    st.session_state.frame_idx = 0


# Initialize optional rerun flag
if "needs_rerun" not in st.session_state:
    st.session_state.needs_rerun = False


# Handle delayed rerun to avoid losing frame index
if st.session_state.needs_rerun:
    st.session_state.needs_rerun = False
    st.experimental_rerun()

if "frame_display" not in st.session_state:
    st.session_state.frame_display = 0  # this will follow playback




# === COLOR MAP ===
COLOR_MAP = {
    "green": (0, 255, 0),
    "white": (255, 255, 255),
    "blue": (255, 0, 0),
    "red": (0, 0, 255),
    "yellow": (0, 255, 255),
    "magenta": (255, 0, 255),
    "cyan": (255, 255, 0),
    "orange": (0, 165, 255),
    "gray": (128, 128, 128),
    "black": (0, 0, 0),
}


# Functions:

# Callback slider:
def user_moved_slider():
    st.session_state.play = False  # Pause when user moves slider
    st.session_state.playback_index = st.session_state.frame_idx  # Set playback start
    st.session_state.frame_display = st.session_state.frame_idx   # Show this frame


# Sharp plot:
def show_sharp_plot(fig, placeholder):
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", pad_inches=0.05)
    svg_data = buf.getvalue()
    placeholder.markdown(f'<div>{svg_data}</div>', unsafe_allow_html=True)
    plt.close(fig)

    

def smooth_series(series, window_size=5):
    return pd.Series(series).rolling(window=window_size, min_periods=1, center=True).mean().tolist()



# low butterworth filter
def butter_lowpass_filter(signal, cutoff=5, order=1, fs=30):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal)

    

def calculate_angle(ax, ay, bx, by, cx, cy):
    if any(pd.isna([ax, ay, bx, by, cx, cy])):
        return np.nan
    v1, v2 = np.array([ax - bx, ay - by]), np.array([cx - bx, cy - by])
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm == 0:
        return np.nan
    angle = np.arccos(np.clip(np.dot(v1, v2) / norm, -1.0, 1.0))
    return np.degrees(angle)

def frame_diff(x, y):
    return [0] + [np.linalg.norm([x[i] - x[i - 1], y[i] - y[i - 1]]) for i in range(1, len(x))]

    
def get_person_frames_and_data(session_name, person_id):
    video_path = VIDEO_INPUT_FOLDER / f"{session_name}.MP4"
    csv_path = FINAL_DF_INPUT_FOLDER / f"{session_name}_merged.csv"
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Load and filter data
    df = pd.read_csv(csv_path)
    person_df = df[df["New_ID"] == person_id].copy()
    
    # Interpolate all numeric pose values (x, y, conf) by frame
    pose_columns = [col for col in person_df.columns if col.endswith(("_x", "_y", "_conf"))]
    person_df.sort_values("Frame", inplace=True)
    person_df[pose_columns] = person_df[pose_columns].interpolate(method="linear", limit_direction="both")
    
    # Recalculate visible frames AFTER interpolation
    visible_frames = sorted(person_df["Frame"].dropna().unique())

    # Parameters
    max_gap_frames = int(fps * 0.25)
    interpolated_rows = []

    # Gap detection and interpolation
    for i in range(1, len(visible_frames)):
        prev_f = visible_frames[i - 1]
        curr_f = visible_frames[i]
        gap = curr_f - prev_f

        if 1 < gap <= max_gap_frames:
            df_prev = person_df[person_df["Frame"] == prev_f].iloc[0]
            df_next = person_df[person_df["Frame"] == curr_f].iloc[0]

            for f in range(prev_f + 1, curr_f):
                alpha = (f - prev_f) / (curr_f - prev_f)
                interpolated = {"Frame": f, "New_ID": person_id}

                # Interpolate only numeric pose columns
                for col in person_df.columns:
                    if col.endswith("_x") or col.endswith("_y") or col.endswith("_conf"):
                        val_prev = df_prev[col]
                        val_next = df_next[col]
                        if pd.notna(val_prev) and pd.notna(val_next):
                            interpolated[col] = (1 - alpha) * val_prev + alpha * val_next
                        else:
                            interpolated[col] = np.nan
                    elif col not in interpolated:
                        interpolated[col] = df_prev[col]  # Copy static info

                interpolated_rows.append(interpolated)

    # Add interpolated data and sort
    if interpolated_rows:
        person_df = pd.concat([person_df, pd.DataFrame(interpolated_rows)], ignore_index=True)

    person_df = person_df.sort_values("Frame").reset_index(drop=True)
    visible_frames = sorted(person_df["Frame"].dropna().unique())[::FRAME_STRIDE]

    return cap, person_df, visible_frames, fps


def draw_pose_on_frame(frame, row, bodyparts, connections, point_color="green", line_color="white"):
    point_bgr = COLOR_MAP.get(point_color, (0, 255, 0))
    line_bgr = COLOR_MAP.get(line_color, (255, 255, 255))

    keypoints = {}
    for bp in bodyparts:
        x, y, conf = row.get(f"{bp}_x", np.nan), row.get(f"{bp}_y", np.nan), row.get(f"{bp}_conf", 0)
        if not np.isnan(x) and not np.isnan(y) and conf > LIKELIHOOD_THRESHOLD:
            keypoints[bp] = (int(x), int(y))
            cv2.circle(frame, keypoints[bp], 13, point_bgr, -1)

    for bp1, bp2 in connections:
        if bp1 in keypoints and bp2 in keypoints:
            cv2.line(frame, keypoints[bp1], keypoints[bp2], line_bgr, 9)

    return frame


# Distance function
def dist(wrist_x, wrist_y, obj_x, obj_y):
    if any(pd.isna([wrist_x, wrist_y, obj_x, obj_y])):
        return np.nan
    return np.linalg.norm([wrist_x - obj_x, wrist_y - obj_y])
    



def draw_object_parts(frame, row, object_parts):
    color_map = {
        "Tail": (147, 20, 255),
        "Handle": (147, 0, 255),
        "Tip": (147, 0, 255),
    }
    
    for part in object_parts:
        obj_x = row.get(f"{part}_center_x", np.nan)
        obj_y = row.get(f"{part}_center_y", np.nan)
        if pd.notna(obj_x) and pd.notna(obj_y):
            obj_coords = (int(obj_x), int(obj_y))
            cv2.circle(frame, obj_coords, 18, color_map.get(part, (200, 200, 200)), -1)

    return frame




def refine_to_next_ankle_y_maximum(
    coords_df,
    candidate_indices,
    side,
    fps,
    MAX_SHIFT_MS,
    Y_PROMINENCE=0.0
):
    """
    For each candidate frame, search forward up to MAX_SHIFT_MS in the ankle_y signal
    for the NEXT local maximum (peak). If none is found, keep the original index.

    Parameters
    ----------
    coords_df : pd.DataFrame
        DataFrame containing pose data.
    candidate_indices : list[int] or np.array[int]
        Frame indices of initial candidates.
    side : str
        'left' or 'right'
    fps : float
        Frames per second.
    MAX_SHIFT_MS : int
        Max window after candidate to search (in milliseconds).
    Y_PROMINENCE : float
        Optional minimum prominence for maxima.

    Returns
    -------
    List[int] : Refined frame indices (each being either the next peak or original).
    """
    ankle_y = coords_df[f"{side}_ankle_y"].values
    max_shift = int(np.round(MAX_SHIFT_MS * fps / 1000))
    N = len(ankle_y)
    refined_indices = []

    for idx in candidate_indices:
        # Define the window: starts just after idx to idx+max_shift (inclusive)
        start = idx + 1
        end = min(idx + max_shift, N - 1)
        if start > end:
            # If window is empty, just keep idx
            refined_indices.append(idx)
            continue
        segment = ankle_y[start:end+1]
        peaks, _ = find_peaks(segment, prominence=Y_PROMINENCE)
        if len(peaks) > 0:
            # peaks[0] is relative to start, so add to start (which is idx+1)
            refined_indices.append(start + peaks[0])
        else:
            refined_indices.append(idx)
    return refined_indices
    

    

# === STREAMLIT APP ===
st.set_page_config(layout="wide")

st.markdown("""
    <style>
    /* Reduce top padding of main container */
    .block-container {
        padding-top: 0.1rem !important;
    }

    /* Hide Streamlit's main header if it exists */
    header {
        visibility: hidden;
        height: 0px;
    }

    /* Prevent white bar from covering content */
    .css-18ni7ap {
        padding-top: 0rem !important;
    }

    /* Reduce space between elements globally */
    .element-container {
        margin-bottom: 0.5rem !important;
    }
    </style>
""", unsafe_allow_html=True)




# === Unified Frame Navigation Slider ===
slider_placeholder = st.empty()

# Add this near the frame display

frame_col, control_col, slider_col = st.columns([5, 1, 4])# More space for image


with frame_col:
    frame_placeholder = st.empty()

graph_col, _ = st.columns([9.5, 0.5])  # X% width, Y% spacer
with graph_col:
    graph_placeholder = st.empty()



if "just_pressed_play" not in st.session_state:
    st.session_state.just_pressed_play = False
    

with control_col:
    st.markdown("""
        <div style="margin-bottom: 0.2rem; font-size: 16px; font-weight: 600;">🎮 Controls</div>
    """, unsafe_allow_html=True)
    play_col, pause_col = st.columns(2)
    
    with play_col:
        if st.button("▶️"):
            st.session_state.play = True
            st.session_state.playback_index = st.session_state.frame_display
    
    with pause_col:
        if st.button("⏸"):
            st.session_state.play = False
            st.session_state.playback_index = st.session_state.frame_display
    

    st.markdown('<div style="font-size: 16px; font-weight: 600; margin-bottom: 0.2rem;">🎨 Colors</div>', unsafe_allow_html=True)
    
    # Marker Color
    st.markdown('<div style="font-size: 12px; margin-bottom: -0.2rem;">● Marker Color</div>', unsafe_allow_html=True)
    point_color = st.selectbox("Marker Color", list(COLOR_MAP.keys()), index=7, key="point_color", label_visibility="collapsed")
    
    # Line Color
    st.markdown('<div style="font-size: 12px; margin-bottom: -0.2rem;">─ Line Color</div>', unsafe_allow_html=True)
    line_color = st.selectbox("Line Color", list(COLOR_MAP.keys()), index=0, key="line_color", label_visibility="collapsed")
    
    st.markdown("""
        <div style="margin-top: 0.5rem; margin-bottom: 0.2rem;"><strong>⏱️ Filter (sec visible)</strong></div>
    """, unsafe_allow_html=True)

    min_duration_sec = st.slider("Minimum duration (s)", 1, 60, 4, label_visibility="collapsed", key="duration_filter")

    

csv_files = sorted(FINAL_DF_INPUT_FOLDER.glob("*_merged.csv"))
session_names = [f.stem.replace("_merged", "") for f in csv_files]





with slider_col:
    st.markdown('<div style="font-size: 16px; font-weight: 600; margin-bottom: 0.2rem;">📂 Select Session and ID</div>', unsafe_allow_html=True)

    # === Session & Person ID Selection ===
    csv_files = sorted(FINAL_DF_INPUT_FOLDER.glob("*_merged.csv"))
    session_names = [f.stem.replace("_merged", "") for f in csv_files]
    session_name = st.selectbox("Select Session", session_names, key="session_select")

    # Load the selected merged file
    if session_name:
        session_df = pd.read_csv(FINAL_DF_INPUT_FOLDER / f"{session_name}_merged.csv")



    # === FILTER PERSON IDS BY MINIMUM SEGMENT DURATION ===
    cap = cv2.VideoCapture(str(VIDEO_INPUT_FOLDER / f"{session_name}.MP4"))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    min_duration_sec = st.session_state.duration_filter
    max_allowed_gap_sec = 0.25
    min_frames = int(min_duration_sec * fps)
    max_gap = int(max_allowed_gap_sec * fps)
    
    valid_ids = []
    
    for pid in session_df["New_ID"].unique():
        df_pid = session_df[session_df["New_ID"] == pid].sort_values("Frame").copy()
        df_pid["Frame_Diff"] = df_pid["Frame"].diff().fillna(1)
        df_pid["Segment_ID"] = (df_pid["Frame_Diff"] > max_gap).cumsum()
    
        for _, seg in df_pid.groupby("Segment_ID"):
            if len(seg) >= min_frames:
                valid_ids.append(pid)
                break  # One valid segment is enough
    
    person_ids_all = sorted(valid_ids)
    person_ids = [pid for pid in person_ids_all if pid >= MIN_ID_THRESHOLD] if ONLY_SHOW_HIGH_IDS else person_ids_all

    if not person_ids:
        st.warning("⚠️ No valid person IDs found in this session. Try another session.")
        st.stop()

    person_id = st.selectbox("Select Person ID", person_ids, key="person_select")
    
    # === Reset frame index if session or person changed ===
    prev_session = st.session_state.get("prev_session", None)
    prev_person = st.session_state.get("prev_person", None)
    
    if session_name != prev_session or person_id != prev_person:
        st.session_state.frame_idx = 0
        st.session_state.frame_display = 0
        st.session_state.playback_index = 0
        st.session_state.play = False  # Also pause just in case
    
    # Update state for next comparison
    st.session_state.prev_session = session_name
    st.session_state.prev_person = person_id

    cap, person_df, visible_frames, fps = get_person_frames_and_data(session_name, person_id)


    


if not st.session_state.play:
    st.session_state.frame_idx = st.session_state.frame_display  # ✅ only safe before slider appears

    
with slider_col:
    st.markdown('<div style="font-size: 16px; font-weight: 600; margin-bottom: 0.2rem;">📊 Frame & Graph Settings</div>', unsafe_allow_html=True)

    # Create 4 columns: first two for buttons, last two empty
    nav_col1, nav_col2, _, _ = st.columns([1, 1, 1, 1])
    
    with nav_col1:
        if st.button("⬅️ Back"):
            st.session_state.frame_idx = max(0, st.session_state.frame_idx - 1)
            st.session_state.frame_display = st.session_state.frame_idx
            st.rerun()
    
    with nav_col2:
        if st.button("➡️ Forward"):
            st.session_state.frame_idx = min(len(visible_frames) - 1, st.session_state.frame_idx + 1)
            st.session_state.frame_display = st.session_state.frame_idx
            st.rerun()
    
    # Sliders below
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.write("**Frame Index**")
        st.session_state.frame_idx = min(st.session_state.frame_idx, len(visible_frames) - 1)

        st.slider(
            "Frame Index",
            0,
            len(visible_frames) - 1,
            key="frame_idx",
            disabled=st.session_state.play,
            on_change=user_moved_slider
        )
    
    with col2:
        st.write("**Graph Window Size**")
        window_size = st.slider("Graph Window Size", 10, 1500, 500, step=100, key="graph_slider")
    



# === Ensure valid frame index ===
if len(visible_frames) == 0:
    st.stop()
if "frame_idx" not in st.session_state:
    st.session_state.frame_idx = 0

 

# Prepare plot data
plot_df = person_df[person_df.Frame.isin(visible_frames)].reset_index(drop=True)


# Smoothing the signals: 
# === Step 1: Setup ===
joint_names = ["ankle", "knee", "hip", "shoulder", "elbow", "wrist"]
sides = ["left", "right"]
axes = ["x", "y"]

# Initialize dictionary to collect all coordinate time series
coords = {f"{side}_{joint}_{axis}": [] for side in sides for joint in joint_names for axis in axes}

# === Step 2: Fill from plot_df row by row ===
for row in plot_df.itertuples():
    row_data = row._asdict()
    for key in coords:
        coords[key].append(row_data.get(key, np.nan))

# Define joint motion types
slow_joints = ["hip", "shoulder"]
fast_joints = ["ankle", "wrist", "elbow", "Tail", "Handle", "Tip"]

# Apply Butterworth filter with joint-specific cutoffs
for key in coords:
    signal = coords[key]
    
    if len(signal) > 5 and not all(pd.isna(signal)):
        joint_name = key.split("_")[1] if "_" in key else key  # Extract joint name

        # Select cutoff frequency based on joint type
        if any(j in joint_name for j in slow_joints):
            cutoff = 3
        elif any(j in joint_name for j in fast_joints):
            cutoff = 7
        else:
            cutoff = 5

        # Try filtering
        try:
            coords[key] = butter_lowpass_filter(np.array(signal), cutoff=cutoff, order=1, fs=fps)
        except Exception as e:
            print(f"⚠️ Skipping filter for {key} due to error: {e}")

# === Step 4: Convert to DataFrame ===
coords_df = pd.DataFrame(coords)

# Optionally add frame column if needed
coords_df["frame"] = plot_df["Frame"].values
frame_to_row = {frame: i for i, frame in enumerate(coords_df["frame"])}


parts = ["Tip", "Handle", "Tail"]

for part in parts:
    x_col = f"{part}_center_x"
    y_col = f"{part}_center_y"
    if x_col not in plot_df.columns or y_col not in plot_df.columns:
        print(f"⚠️ Warning: {x_col} or {y_col} not found in {session_name} (will fill with NaN).")
    coords_df[x_col] = plot_df.get(x_col, pd.Series([np.nan]*len(plot_df))).values
    coords_df[y_col] = plot_df.get(y_col, pd.Series([np.nan]*len(plot_df))).values

def mean_if_at_least_one(row, cols):
    vals = row[cols].values
    not_nan = np.isfinite(vals)
    if np.sum(not_nan) >= 1:
        return np.nanmean(vals)
    else:
        return np.nan

x_cols = [f"{part}_center_x" for part in parts]
y_cols = [f"{part}_center_y" for part in parts]

coords_df["javelin_center_x"] = coords_df.apply(lambda row: mean_if_at_least_one(row, x_cols), axis=1)
coords_df["javelin_center_y"] = coords_df.apply(lambda row: mean_if_at_least_one(row, y_cols), axis=1)

# Distance from javelin center to right wrist
coords_df["dist_javelin_to_right_wrist"] = np.sqrt(
    (coords_df["javelin_center_x"] - coords_df["right_wrist_x"])**2 +
    (coords_df["javelin_center_y"] - coords_df["right_wrist_y"])**2
)

# Distance from javelin center to left wrist
coords_df["dist_javelin_to_left_wrist"] = np.sqrt(
    (coords_df["javelin_center_x"] - coords_df["left_wrist_x"])**2 +
    (coords_df["javelin_center_y"] - coords_df["left_wrist_y"])**2
)


def border_nanmean(arr, n=2, window_size=8):
    """Mean of up to n closest valid values in first/last window_size."""
    arr = np.array(arr)
    # First part
    first = arr[:window_size]
    first_valid = first[np.isfinite(first)]
    if len(first_valid) >= n:
        first_border_mean = np.mean(first_valid[:n])
    elif len(first_valid) > 0:
        first_border_mean = np.mean(first_valid)
    else:
        first_border_mean = np.nan
    # Last part
    last = arr[-window_size:]
    last_valid = last[np.isfinite(last)]
    if len(last_valid) >= n:
        last_border_mean = np.mean(last_valid[-n:])
    elif len(last_valid) > 0:
        last_border_mean = np.mean(last_valid)
    else:
        last_border_mean = np.nan
    return first_border_mean, last_border_mean


def find_sustained_increases(
        signal, deriv, threshold, min_sustain=100, min_delta=MIN_DELTA, min_fraction=MIN_FRACTION, min_start_ms=WHEN_START_MS):
    """
    min_sustain: number of frames
    min_start_ms: where to start searching, in milliseconds (converted to frames)
    fps: frames per second
    """
    min_start = int(round(min_start_ms * fps / 1000))
    i = min_start
    N = len(signal)
    while i <= N - min_sustain - 1:
        if deriv[i] > threshold:
            window = signal[i:i+min_sustain+1]
            start_mean, end_mean = border_nanmean(window, n=2, window_size=5)
            if np.isnan(start_mean) or np.isnan(end_mean):
                i += 1
                continue
            diffs = np.diff(window)
            fraction_increasing = np.mean(diffs > 0)
            sustained = fraction_increasing >= min_fraction
            total_increase = end_mean - start_mean
            if sustained and total_increase > min_delta:
                return np.array([i])
        i += 1
    return np.array([])

    
    

# First derivate to get the sharp increase in javelin to wrist:
left_signal  = coords_df["dist_javelin_to_left_wrist"].values
right_signal = coords_df["dist_javelin_to_right_wrist"].values

left_deriv  = np.diff(left_signal, prepend=left_signal[0])
right_deriv = np.diff(right_signal, prepend=right_signal[0])

min_sustain = int(fps * MIN_SUSTAIN_SEC)

left_sustained_indices  = find_sustained_increases(
    left_signal, left_deriv, threshold=THRESHOLD, min_sustain=min_sustain, min_delta=MIN_DELTA)
right_sustained_indices = find_sustained_increases(
    right_signal, right_deriv, threshold=THRESHOLD, min_sustain=min_sustain, min_delta=MIN_DELTA)


release_indices = []
if len(left_sustained_indices) > 0 and len(right_sustained_indices) > 0:
    a = left_sustained_indices[0]
    b = right_sustained_indices[0]
    release_idx = int(np.round((a + b) / 2))
elif len(left_sustained_indices) > 0:
    release_idx = left_sustained_indices[0]
elif len(right_sustained_indices) > 0:
    release_idx = right_sustained_indices[0]
else:
    release_idx = None
    

# Calculate slope for both hips
x = np.arange(len(coords_df))  # Frame indices

# Mean hip x for each frame (averaging left and right hip)
hip_x_mean = (coords_df["left_hip_x"] + coords_df["right_hip_x"]) / 2

# Fit a line: slope tells direction
slope, intercept, r_value, p_value, std_err = linregress(x, hip_x_mean)

if slope > 0:
    direction = "→ Right"
elif slope < 0:
    direction = "← Left"
else:
    direction = "No Movement"



# Calculate hip to ankle distance: 
# --- Step 1: Prepare signals ---
coords_df["left_ankle_rel_x"] = coords_df["left_hip_x"] - coords_df["left_ankle_x"]
coords_df["right_ankle_rel_x"] = coords_df["right_hip_x"] - coords_df["right_ankle_x"]

left_diff = coords_df["left_ankle_rel_x"].values
right_diff = coords_df["right_ankle_rel_x"].values

DISTANCE = int(np.round(DISTANCE_MS * fps / 1000))
MAX_SHIFT_FRAMES = int(np.round(MAX_SHIFT_MS * fps / 1000))  # max shift in frames

# --- Step 2: Find extrema in horizontal distance ---
if slope > 0:
    left_extrema, _ = find_peaks(-left_diff, distance=DISTANCE, prominence=PROMINENCE)
    right_extrema, _ = find_peaks(-right_diff, distance=DISTANCE, prominence=PROMINENCE)
    extrema_label = "Minima (before release)"
elif slope < 0:
    left_extrema, _ = find_peaks(left_diff, distance=DISTANCE, prominence=PROMINENCE)
    right_extrema, _ = find_peaks(right_diff, distance=DISTANCE, prominence=PROMINENCE)
    extrema_label = "Maxima (before release)"
else:
    left_extrema, right_extrema = np.array([]), np.array([])
    extrema_label = "No movement"

# --- Step 3: Filter to those before release ---
if release_idx is not None:
    left_extrema_before = left_extrema[left_extrema < release_idx]
    right_extrema_before = right_extrema[right_extrema < release_idx]
else:
    left_extrema_before, right_extrema_before = [], []

# --- Step 4: Refine using ankle_y minima ---
left_refined = refine_to_next_ankle_y_maximum(
    coords_df, left_extrema_before, side="left", fps=fps,
    MAX_SHIFT_MS=MAX_SHIFT_MS, Y_PROMINENCE=Y_PROMINENCE
)
right_refined = refine_to_next_ankle_y_maximum(
    coords_df, right_extrema_before, side="right", fps=fps,
    MAX_SHIFT_MS=MAX_SHIFT_MS, Y_PROMINENCE=Y_PROMINENCE
)



            


# Calculations of joint angles:

coords_df["elbow_l"] = coords_df.apply(
    lambda row: calculate_angle(row["left_wrist_x"], row["left_wrist_y"],
                                 row["left_elbow_x"], row["left_elbow_y"],
                                 row["left_shoulder_x"], row["left_shoulder_y"]), axis=1)

coords_df["elbow_r"] = coords_df.apply(
    lambda row: calculate_angle(row["right_wrist_x"], row["right_wrist_y"],
                                 row["right_elbow_x"], row["right_elbow_y"],
                                 row["right_shoulder_x"], row["right_shoulder_y"]), axis=1)

coords_df["knee_l"] = coords_df.apply(
    lambda row: calculate_angle(row["left_ankle_x"], row["left_ankle_y"],
                                 row["left_knee_x"], row["left_knee_y"],
                                 row["left_hip_x"], row["left_hip_y"]), axis=1)

coords_df["knee_r"] = coords_df.apply(
    lambda row: calculate_angle(row["right_ankle_x"], row["right_ankle_y"],
                                 row["right_knee_x"], row["right_knee_y"],
                                 row["right_hip_x"], row["right_hip_y"]), axis=1)




# Calculate stride length: 
# leg length: 
leg_lengths_px = []
for i, row in coords_df.iterrows():
    dists = []
    if row['knee_r'] > 170:
        dx_r = abs(row['right_hip_x'] - row['right_ankle_x'])
        dy_r = abs(row['right_hip_y'] - row['right_ankle_y'])
        d_r = np.sqrt(dx_r**2 + dy_r**2)
        dists.append(d_r)
    if row['knee_l'] > 170:
        dx_l = abs(row['left_hip_x'] - row['left_ankle_x'])
        dy_l = abs(row['left_hip_y'] - row['left_ankle_y'])
        d_l = np.sqrt(dx_l**2 + dy_l**2)
        dists.append(d_l)
    if dists:
        leg_lengths_px.append(np.mean(dists))
leg_length_px = np.nanmedian(leg_lengths_px)




# 1. Gather events (frame idx, x, y, side)
contacts = []
for idx in left_refined:
    contacts.append((idx, coords_df.loc[idx, 'left_ankle_x'], coords_df.loc[idx, 'left_ankle_y'], 'L'))
for idx in right_refined:
    contacts.append((idx, coords_df.loc[idx, 'right_ankle_x'], coords_df.loc[idx, 'right_ankle_y'], 'R'))

# 2. Sort by frame index (time)
contacts.sort(key=lambda x: x[0])  # ascending by frame/time

# 3. Only keep those before release, as above (already done if you used *_before arrays)
# Optional: Go backwards from last before release
contacts = [c for c in contacts if c[0] < release_idx]

# 4. Calculate all steps before release, regardless of alternation
step_events = []
step_lengths_px = []
step_types = []

for i in range(1, len(contacts)):  # forwards: earliest to latest
    idx_prev, x_prev, y_prev, side_prev = contacts[i-1]
    idx, x, y, side = contacts[i]
    step_length = np.sqrt((x - x_prev)**2 + (y - y_prev)**2)
    step_lengths_px.append(step_length)
    ratio = step_length / leg_length_px if leg_length_px else np.nan
    step_types.append("short" if ratio < STEP_RATIO else "long")
    step_events.append({
        "from_frame": int(idx_prev),
        "to_frame": int(idx),
        "from_side": side_prev,
        "to_side": side,
        "step_length_px": float(step_length),
        "leg_length_px": float(leg_length_px),
        "step_length/leg_length": float(ratio),
        "type": "short" if ratio < 1 else "long"
    })

# Sequence of all foot sides before release
step_sequence = [c[3] for c in contacts]
st.write("Step side sequence (before release):", step_sequence)


stride_pairs = []
for i in range(1, len(step_events)):
    prev = step_events[i-1]
    curr = step_events[i]
    if prev["to_side"] != curr["to_side"]:  # alternation (e.g., L to R or R to L)
        stride_pairs.append((prev["to_frame"], curr["to_frame"], prev["to_side"], curr["to_side"]))







# Did the student stop before release?
# --- Detect steps after release within a restricted time window ---
# We only count steps that happen shortly after the release event, not later during walking.
# Adjust `time_window_ms` to set the window (e.g., 400–1200 ms).
frames_window = int(np.ceil((time_window_ms / 1000) * fps))

# Gather all detected foot contact indices (across both sides, all steps)
all_contact_indices = np.sort(np.concatenate([left_refined, right_refined]))

def steps_after_release_within_window(all_contact_indices, release_idx, frames_window):
    """Return indices of steps occurring after release within a specific window."""
    post_release = all_contact_indices[all_contact_indices > release_idx]
    return [int(idx) for idx in post_release if (idx - release_idx) <= frames_window]

if release_idx is not None:
    post_release_indices_in_window = steps_after_release_within_window(all_contact_indices, release_idx, frames_window)
    if len(post_release_indices_in_window) > 0:
        criterion_release_before_stop = False  # At least one step in window: student did NOT stop immediately after release
        first_post_release_idx = post_release_indices_in_window[0]
    else:
        criterion_release_before_stop = True   # No step in window: student stopped after release
        first_post_release_idx = None
else:
    criterion_release_before_stop = None
    first_post_release_idx = None


if first_post_release_idx is not None:
    first_post_release_ms = (first_post_release_idx - release_idx) * 1000 / fps
else:
    first_post_release_ms = None



# --- Deceleration Detection: Is there a "strong stop" before release? ---

# Safe defaults in case of early exit
vel_before = np.nan
vel_stop = np.nan
ratio = np.nan
strong_stop = None
stop_accel_value = None
stop_vel_after = None

# Only calculate if release_idx is valid
if release_idx is not None:
    coords_df["mean_hip_x"] = (coords_df["left_hip_x"] + coords_df["right_hip_x"]) / 2

    # Velocity per frame (pixels/frame)
    coords_df["mean_hip_x_vel"] = coords_df["mean_hip_x"].diff().fillna(0)
    # Convert to velocity in pixels/sec
    coords_df["mean_hip_x_vel"] *= fps

    # Frame indices for windows
    onset_offset_frames = int(np.ceil(ONSET_OFFSET_MS / 1000 * fps))
    stop_win_frames = int(STOP_WIN_MS / 1000 * fps)

    # Indices for stop window (500 ms before release up to release)
    idx_stop_win_start = max(onset_offset_frames, release_idx - stop_win_frames)
    idx_stop_win_end = release_idx + 1  # include release frame

    # Indices for reference window (max 2.5s before stop window, starting at 150 ms after onset)
    idx_runup_end = idx_stop_win_start
    idx_runup_start = max(onset_offset_frames, idx_runup_end - int(WINDOW_BEFORE_RELEASE_MS / 1000 * fps))

    # Use mean hip x velocity series
    hip_vel_series = coords_df["mean_hip_x_vel"].values

    # Calculate mean velocities in the windows
    vel_before = np.abs(hip_vel_series[idx_runup_start:idx_runup_end]).mean() if idx_runup_end > idx_runup_start else np.nan
    vel_stop   = np.abs(hip_vel_series[idx_stop_win_start:idx_stop_win_end]).mean() if idx_stop_win_end > idx_stop_win_start else np.nan

    # Stop ratio and criterion
    ratio = vel_stop / vel_before if vel_before > 0 else np.nan
    strong_stop = ratio < STOP_RATIO_THRESHOLD if not np.isnan(ratio) else None

    # --- Additional: Compute hip acceleration (stop_accel_value) and post-release velocity (stop_vel_after) ---
    # Hip acceleration (pixels/sec²)
    hip_acc_series = np.diff(hip_vel_series, prepend=hip_vel_series[0]) * fps

    # Minimum acceleration (strongest deceleration) in stop window
    if idx_stop_win_end > idx_stop_win_start:
        stop_accel_value = np.nanmin(hip_acc_series[idx_stop_win_start:idx_stop_win_end])
    else:
        stop_accel_value = None

    # Mean hip velocity in window after release (e.g., next 200 ms)
    POST_WIN_MS = 200
    post_win_frames = int(np.ceil(POST_WIN_MS / 1000 * fps))
    idx_post_win_start = release_idx + 1
    idx_post_win_end = min(len(hip_vel_series), release_idx + 1 + post_win_frames)

    if idx_post_win_end > idx_post_win_start:
        stop_vel_after = np.abs(hip_vel_series[idx_post_win_start:idx_post_win_end]).mean()
    else:
        stop_vel_after = None



    



# Calculation for assessing if elbow was pointing in the throwing direction
# PRE window: exactly 500 ms
if release_idx is None:
    st.warning("No release event detected for this throw. Skipping elbow window calculations.")
    # Optionally skip the rest of the analysis for this trial
    st.stop()

n_pre_frames = int(np.round(0.500 * fps))
idx_pre_start = max(0, release_idx - n_pre_frames + 1)
idx_pre_end = release_idx + 1
pre_window = coords_df.iloc[idx_pre_start:idx_pre_end]

# POST window: exactly 500 ms
n_post_frames = int(np.round(0.500 * fps))
idx_post_start = release_idx
idx_post_end = min(len(coords_df), release_idx + n_post_frames)
post_window = coords_df.iloc[idx_post_start:idx_post_end]

if direction == "→ Right":
    throwing_side = "right"
else:
    throwing_side = "left"

elbow_x_col = f"{throwing_side}_elbow_x"
wrist_x_col = f"{throwing_side}_wrist_x"
elbow_angle_col = f"elbow_{throwing_side[0]}"  # 'elbow_r' or 'elbow_l'


# 1. Elbow ahead of wrist (20%+)
if direction == "→ Right":
    ahead_mask = pre_window[elbow_x_col] > pre_window[wrist_x_col]
else:
    ahead_mask = pre_window[elbow_x_col] < pre_window[wrist_x_col]

frac_ahead = np.mean(ahead_mask)
criterion_1 = frac_ahead >= 0.20

# 2. Min elbow angle <90° in at least 3% of frames
frac_min_angle = np.mean(pre_window[elbow_angle_col] < 90)
criterion_2 = frac_min_angle >= 0.03
# For min elbow angle < 90° (pre-release)
num_frames_min_angle = np.sum(pre_window[elbow_angle_col] < 90)
duration_min_angle_ms = num_frames_min_angle * (1000 / fps)

# 3. Max elbow angle >150° in at least 3% of frames after release
frac_max_angle = np.mean(post_window[elbow_angle_col] > 150)
criterion_3 = frac_max_angle >= 0.03
# For max elbow angle > 150° (post-release)
num_frames_max_angle = np.sum(post_window[elbow_angle_col] > 150)
duration_max_angle_ms = num_frames_max_angle * (1000 / fps)

# 4. Positive slope of elbow angle before release 
slope_win = int(0.300 * fps)
idx_slope_start = max(0, release_idx - slope_win)
idx_slope_end = release_idx + 1

elbow_angles_slope = coords_df[elbow_angle_col].iloc[idx_slope_start:idx_slope_end].values
x_vals = np.arange(len(elbow_angles_slope))

if len(elbow_angles_slope) > 1:
    slope_val, _, _, _, _ = linregress(x_vals, elbow_angles_slope)
    criterion_4 = slope_val > 0
else:
    slope_val = np.nan
    criterion_4 = False

# For min in pre-window
tol = 5  # or whatever you use for float tolerance
min_val = np.nanmin(pre_window[elbow_angle_col])
min_mask = np.abs(pre_window[elbow_angle_col] - min_val) < tol
labeled, n_features = label(min_mask)
durations = [np.sum(labeled == i) for i in range(1, n_features + 1)]
if any(d >= 3 for d in durations):
    min_angle_valid = min_val
else:
    min_angle_valid = np.nan

# For max in post-window
max_val = np.nanmax(post_window[elbow_angle_col])
max_mask = np.abs(post_window[elbow_angle_col] - max_val) < tol
labeled, n_features = label(max_mask)
durations = [np.sum(labeled == i) for i in range(1, n_features + 1)]
if any(d >= 3 for d in durations):
    max_angle_valid = max_val
else:
    max_angle_valid = np.nan


num_frames_ahead = np.sum(ahead_mask)


st.write(f"Using columns: {elbow_x_col}, {wrist_x_col}, {elbow_angle_col}")
st.write(f"Fraction elbow ahead of wrist (pre-release): {frac_ahead:.2f} ({'PASS' if criterion_1 else 'FAIL'})")
st.write(
    f"Fraction min elbow angle < 90° (pre-release): {frac_min_angle:.2f} ({'PASS' if criterion_2 else 'FAIL'}) | "
    f"Duration: {duration_min_angle_ms:.1f} ms"
)
st.write(
    f"Fraction max elbow angle > 150° (post-release): {frac_max_angle:.2f} ({'PASS' if criterion_3 else 'FAIL'}) | "
    f"Duration: {duration_max_angle_ms:.1f} ms"
)
st.write(f"Slope of elbow angle before release (last 150 ms): {slope_val:.2f} ({'PASS' if criterion_4 else 'FAIL'})")
st.write(f"pre_window shape: {pre_window.shape}")



# Elbow outstretched AND behind shoulder

# Determine correct elbow and shoulder columns based on throwing side
if direction == "→ Right":
    throwing_side = "right"
    elbow_x_col = "right_elbow_x"
    shoulder_x_col = "right_shoulder_x"
    elbow_angle_col = "elbow_r"
elif direction == "← Left":
    throwing_side = "left"
    elbow_x_col = "left_elbow_x"
    shoulder_x_col = "left_shoulder_x"
    elbow_angle_col = "elbow_l"
else:
    raise ValueError("Unknown throwing direction!")

    

fps = float(fps)

# Calculate frame indices
onset_offset_frames = int(np.ceil(ONSET_OFFSET_MS / 1000 * fps))
offset_frames = int(np.round(OFFSET_BEFORE_RELEASE_MS / 1000 * fps))
idx_win_end = max(onset_offset_frames, release_idx - offset_frames + 1)
idx_win_start = max(onset_offset_frames, idx_win_end - int(WINDOW_BEFORE_RELEASE_MS / 1000 * fps))

elbow_series = coords_df[elbow_angle_col].iloc[idx_win_start:idx_win_end].values
elbow_x_series = coords_df[elbow_x_col].iloc[idx_win_start:idx_win_end].values
shoulder_x_series = coords_df[shoulder_x_col].iloc[idx_win_start:idx_win_end].values

# 1. Frames with sustained extension AND elbow behind shoulder
above = elbow_series > THRESHOLD_ANGLE
if direction == "→ Right":
    behind = elbow_x_series < shoulder_x_series
elif direction == "← Left":
    behind = elbow_x_series > shoulder_x_series
else:
    behind = np.ones_like(elbow_x_series, dtype=bool)  # fallback: don't restrict

above_and_behind = above & behind
num_above_and_behind = np.sum(above_and_behind)
duration_above_and_behind_ms = num_above_and_behind * 1000 / fps
sustained_extension_and_behind = duration_above_and_behind_ms >= REQUIRED_ABOVE_ANGLE_MS

# 2. Maximum angle reached (descriptive, not a pass/fail)
max_angle = np.nanmax(elbow_series)

# 3. (Optional/Descriptive) Did max angle persist for at least N frames?
from scipy.ndimage import label
tolerance = 5
max_mask = np.abs(elbow_series - max_angle) < tolerance
labeled, n_features = label(max_mask)
max_durations = [np.sum(labeled == i) for i in range(1, n_features + 1)]
max_angle_streak = max(max_durations) if max_durations else 0
max_angle_streak_ok = max_angle_streak >= MIN_MAX_DURATION_FRAMES

# Find indices for sustained extension (if it exists)
if sustained_extension_and_behind:
    above_indices = np.where(above_and_behind)[0]
    sustained_start_idx = idx_win_start + above_indices[0]
    sustained_end_idx = idx_win_start + above_indices[-1]
else:
    sustained_start_idx = sustained_end_idx = np.nan

# --- OUTPUT ---
st.write(f"Window analyzed: frames {idx_win_start} to {idx_win_end}, ({(idx_win_end-idx_win_start)/fps:.2f} s)")
st.write(f"Frames above {THRESHOLD_ANGLE}° and elbow behind shoulder: {num_above_and_behind} ({duration_above_and_behind_ms:.1f} ms)")
st.write(f"Maximum elbow angle: {max_angle:.1f}° (present for {max_angle_streak} frames)")

if sustained_extension_and_behind:
    st.write(
        f"✅ Sustained elbow extension: angle > {THRESHOLD_ANGLE}° AND elbow behind shoulder for at least {REQUIRED_ABOVE_ANGLE_MS} ms."
    )
else:
    st.write(
        f"❌ No sustained elbow extension (> {THRESHOLD_ANGLE}° AND elbow behind shoulder for {REQUIRED_ABOVE_ANGLE_MS} ms)."
    )

# (Optional, just info)
st.write(f"Longest streak at max angle: {max_angle_streak} frames (>= {MIN_MAX_DURATION_FRAMES}? {max_angle_streak_ok})")

# for i in range(idx_win_start, idx_win_end):
 #   print(f"Frame {i}: elbow_l={coords_df['elbow_l'][i]:.1f}, "
  #        f"elbow_x={coords_df['left_elbow_x'][i]:.1f}, "
   #       f"shoulder_x={coords_df['left_shoulder_x'][i]:.1f}, "
    #      f"above={coords_df['elbow_l'][i]>150}, "
     #     f"behind={coords_df['left_elbow_x'][i]>coords_df['left_shoulder_x'][i]}")



# Split session_name into components
def parse_session_name(session_name):
    parts = session_name.split("_")
    if len(parts) == 5:
        school, class_id, condition, subj_id, throw_nr = parts
    elif len(parts) == 4:
        school = None
        class_id, condition, subj_id, throw_nr = parts
    else:
        return None, None, None, None, None
    try:
        throw_nr = int(str(throw_nr).lstrip("0"))
    except Exception:
        throw_nr = None
    return school, class_id, condition, subj_id, throw_nr

school, class_id, condition, subj_id, throw_nr = parse_session_name(session_name)



all_results = []

if release_idx is None:
    st.warning("No release event detected for this throw. Skipping event-based analysis.")
    st.stop()
else:
    summary_row = {
        "session": session_name,
        "school": school, 
        "class_id": class_id,
        "condition": condition,
        "subject_id": subj_id,
        "throw_number": throw_nr,
        "release_idx": int(release_idx),
        "direction": direction,
        "fps": fps, 
        "slope": float(slope),
        "leg_length_px": float(leg_length_px),
        "num_steps": len(step_events),
        "step_from_frames": [x["from_frame"] for x in step_events],
        "step_to_frames": [x["to_frame"] for x in step_events],
        "step_from_sides": [x["from_side"] for x in step_events],
        "step_to_sides": [x["to_side"] for x in step_events],
        "step_lengths_px": [x["step_length_px"] for x in step_events],
        "step_types": [x["type"] for x in step_events],
        "step_sequence": [x["to_side"] for x in step_events], 
        # Real event-based kinematic fields (replace with your calculated variables)
        "throwing_side": throwing_side,
        "elbow_ahead_of_wrist_ms_pre": float(num_frames_ahead * 1000 / fps),
        "criterion_1_elbow_ahead": bool(criterion_1),
        "elbow_below_90_ms_pre": float(duration_min_angle_ms),
        "criterion_2_min_elbow": bool(criterion_2),
        "elbow_above_150_ms_post": float(duration_max_angle_ms),
        "criterion_3_max_elbow": bool(criterion_3),
        "elbow_angle_slope": float(slope_val) if not np.isnan(slope_val) else np.nan,
        "criterion_4_positive_slope": bool(criterion_4),
        "min_elbow_angle_pre": float(min_angle_valid),
        "max_elbow_angle_post": float(max_angle_valid),
        "max_elbow_angle_pre_release": float(max_angle),
        "max_angle_streak_frames": int(max_angle_streak),
        "sustained_extension": bool(sustained_extension_and_behind),
        "sustained_extension_start_idx": sustained_start_idx,
        "sustained_extension_end_idx": sustained_end_idx,
        "true_stop_after_release": bool(criterion_release_before_stop),
        "first_post_release_idx": int(first_post_release_idx) if first_post_release_idx is not None else None,
        "first_post_release_ms": float(first_post_release_ms) if first_post_release_ms is not None else None,
        "strong_stop": bool(strong_stop) if strong_stop is not None else None,
        "vel_before": float(vel_before) if not np.isnan(vel_before) else None,
        "vel_stop": float(vel_stop) if not np.isnan(vel_stop) else None,
        "stop_ratio": float(ratio) if not np.isnan(ratio) else None,
        "min_hip_acc": float(stop_accel_value) if stop_accel_value is not None else None,
        "stop_vel_after": float(stop_vel_after) if stop_vel_after is not None else None,
    }
    all_results.append(summary_row)

# Only show the table **once**
summary_df = pd.DataFrame(all_results)
st.dataframe(summary_df)   # Display in Streamlit

# Optional: Save to CSV if needed
# summary_df.to_csv("kinematic_summary.csv", index=False)


# Define Parts of the Javelin:
object_parts = ["Tail", "Handle", "Tip"]
wrist_sides = ["left", "right"]
object_dists = {f"{wrist}_{part}": [] for wrist in wrist_sides for part in object_parts}




    









current_frame_idx = st.session_state.frame_idx



if "playback_index" not in st.session_state:
    st.session_state.playback_index = st.session_state.frame_idx
elif st.session_state.just_pressed_play:
    # Only update if no slider change happened
    st.session_state.playback_index = st.session_state.frame_idx

st.session_state.just_pressed_play = False



current_index = st.session_state.frame_display if st.session_state.play else st.session_state.frame_idx

# === Safety check to prevent IndexError ===
if current_index >= len(visible_frames):
    current_index = max(0, len(visible_frames) - 1)
    st.session_state.frame_idx = current_index
    st.session_state.frame_display = current_index

selected_frame = visible_frames[current_index]



# === Fill object_dists with actual distance values ===
for wrist in ["left", "right"]:
    for part in ["Tail", "Handle", "Tip"]:
        wrist_x = coords_df[f"{wrist}_wrist_x"]
        wrist_y = coords_df[f"{wrist}_wrist_y"]
        obj_x = coords_df[f"{part}_center_x"]
        obj_y = coords_df[f"{part}_center_y"]
        
        # Calculate Euclidean distance for each frame
        distances = np.sqrt((wrist_x - obj_x)**2 + (wrist_y - obj_y)**2)
        object_dists[f"{wrist}_{part}"] = distances

        


plot_dists = [
    ("left", "Tail"),
    ("right", "Tail"),
    ("left", "Handle"),
    ("right", "Handle"),
    ("left", "Tip"),
    ("right", "Tip"),
    
]






if st.session_state.play:
    playback_index = st.session_state.playback_index

    while playback_index < len(visible_frames):
        selected_frame = visible_frames[playback_index]

        # Stop playback if user paused
        if not st.session_state.play:
            st.session_state.frame_idx = playback_index
            st.session_state.playback_index = playback_index
            break

        st.session_state.frame_display = playback_index

        
        # Set video to selected frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, selected_frame)
        ret, frame = cap.read()
        if not ret:
            break

        # Draw pose on frame
        frame_data = person_df[person_df.Frame == selected_frame]
        for _, row in frame_data.iterrows():
            frame = draw_pose_on_frame(frame, row,
                ["left_elbow", "right_elbow", "left_knee", "right_knee", "left_ankle", "right_ankle",
                 "left_hip", "right_hip", "left_shoulder", "right_shoulder", "left_wrist", "right_wrist"],
                [("left_shoulder", "right_shoulder"), ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
                 ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
                 ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
                 ("left_hip", "right_hip"), ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
                 ("right_hip", "right_knee"), ("right_knee", "right_ankle")],
                point_color=point_color, line_color=line_color
            )

        frame = draw_object_parts(frame, row, object_parts)
        

        # Show image
        _, buffer = cv2.imencode('.jpg', cv2.resize(frame, (960, 540)))
        frame_placeholder.image(buffer.tobytes(), caption=f"Frame {selected_frame}")

        # Plot graph (replace `i` with `playback_index`)
        half_window = window_size // 2
        center = frame_to_row.get(selected_frame, 0)
        
        start = max(0, center - half_window)
        end = min(len(coords_df), center + half_window)
        if end > len(plot_df):
            end = len(plot_df)
            start = max(0, end - window_size)
        fig, axs = plt.subplots(2, 5, figsize=(17, 4.5), dpi=600)
        axs = axs.flatten()
        window_df = coords_df.iloc[start:end]

        axs[0].plot(window_df["elbow_l"], label="Left Elbow", color="red", linewidth=0.8)
        axs[0].plot(window_df["elbow_r"], label="Right Elbow", color="blue", linewidth=0.8)
        axs[0].axvline(playback_index, color='black', linestyle='--', linewidth=0.6)
        axs[0].set_xlim(start, end)
        axs[0].legend()
        axs[0].set_title("Elbow Angles")

        axs[5].plot(window_df["knee_l"], label="Left Knee", color="purple", linewidth=0.8)
        axs[5].plot(window_df["knee_r"], label="Right Knee", color="seagreen", linewidth=0.8)
        axs[5].axvline(playback_index, color='black', linestyle='--', linewidth=0.6)
        axs[5].set_xlim(start, end)
        axs[5].legend()
        axs[5].set_title("Knee Angles")

        axs[1].plot(window_df["left_ankle_y"], label="Left Ankle Y", color="orange", linewidth=0.8)
        axs[1].plot(window_df["right_ankle_y"], label="Right Ankle Y", color="lightblue", linewidth=0.8)
        axs[1].invert_yaxis()
        axs[1].axvline(playback_index, color='black', linestyle='--', linewidth=0.6)
        axs[1].set_xlim(start, end)
        axs[1].legend()
        left_frames_ankley = [frame for frame in left_refined if start <= frame <= end]
        right_frames_ankley = [frame for frame in right_refined if start <= frame <= end]
        left_y_vals_ankley = [coords_df.loc[frame, "left_ankle_y"] for frame in left_frames_ankley]
        right_y_vals_ankley = [coords_df.loc[frame, "right_ankle_y"] for frame in right_frames_ankley]
        axs[1].scatter(left_frames_ankley, left_y_vals_ankley, color="orange", s=60, zorder=6, label="Refined L Contact")
        axs[1].scatter(right_frames_ankley, right_y_vals_ankley, color="brown", s=60, zorder=6, label="Refined R Contact")
        axs[1].set_title("Ankle Y Coordinates")

        axs[6].plot(window_df["left_hip_y"], label="Left Hip Y", color="brown", linewidth=0.8)
        axs[6].plot(window_df["right_hip_y"], label="Right Hip Y", color="springgreen", linewidth=0.8)
        axs[6].invert_yaxis()
        axs[6].axvline(playback_index, color='black', linestyle='--', linewidth=0.6)
        axs[6].set_xlim(start, end)
        axs[6].legend()
        axs[6].set_title("Hip Y Coordinates")

        axs[2].plot(window_df["left_ankle_x"], label="Left Ankle X", color="orange", linewidth=0.8)
        axs[2].plot(window_df["right_ankle_x"], label="Right Ankle X", color="lightblue", linewidth=0.8)
        axs[2].axvline(playback_index, color='black', linestyle='--', linewidth=0.6)
        axs[2].set_xlim(start, end)
        axs[2].legend()
        axs[2].set_title("Ankle X Coordinates")

        axs[7].plot(window_df["left_hip_x"], label="Left Hip X", color="brown", linewidth=0.8)
        axs[7].plot(window_df["right_hip_x"], label="Right Hip X", color="springgreen", linewidth=0.8)
        axs[7].axvline(playback_index, color='black', linestyle='--', linewidth=0.6)
        axs[7].set_xlim(start, end)
        axs[7].legend()

        # Annotate slope and direction
        slope_txt = f"Slope: {slope:.2f} ({direction})"
        axs[7].text(0.05, 0.95, slope_txt, transform=axs[7].transAxes,
                    fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        axs[7].set_title("Hip X Coordinates")

        
        axs[3].plot(window_df["left_ankle_rel_x"], label="left_hip_ank_dist", color="orange", linewidth=0.8)
        axs[3].plot(window_df["right_ankle_rel_x"], label="right_hip_ank_dist", color="brown", linewidth=0.8)
        axs[3].axvline(playback_index, color='black', linestyle='--', linewidth=0.6)
        axs[3].set_xlim(start, end)
        axs[3].legend()

        # Extract final contact frames and y-values
        left_frames = [frame for frame, _, _, side in contacts if side == "L" and start <= frame <= end]
        left_y_vals = [coords_df.loc[frame, "left_ankle_rel_x"] for frame in left_frames]
        
        right_frames = [frame for frame, _, _, side in contacts if side == "R" and start <= frame <= end]
        right_y_vals = [coords_df.loc[frame, "right_ankle_rel_x"] for frame in right_frames]
        
        axs[3].scatter(left_frames, left_y_vals, color="orange", s=60, zorder=6, label="Final Left Contact")
        axs[3].scatter(right_frames, right_y_vals, color="brown", s=60, zorder=6, label="Final Right Contact")

        for i, step in enumerate(step_events):
            # Calculate midpoint x value between contacts
            x_plot = (step["from_frame"] + step["to_frame"]) / 2
        
            # Get corresponding y values for plotting annotation
            y0 = left_diff[step["from_frame"]] if step["from_side"] == "L" else right_diff[step["from_frame"]]
            y1 = left_diff[step["to_frame"]]   if step["to_side"]   == "L" else right_diff[step["to_frame"]]
            y_plot = (y0 + y1) / 2
        
            # Annotate step type ("long" or "short")
            axs[3].text(
                x_plot, y_plot, step["type"],
                fontsize=11, color="black",
                ha="center", va="bottom", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none")
            )
        axs[3].set_title("Hip_Ankle_x_difference")


        # Plot wrist–object distances
        axs[4].plot(window_df["dist_javelin_to_left_wrist"], label="left_javelin_wrist", color="green", linewidth=0.8)
        axs[4].plot(window_df["dist_javelin_to_right_wrist"], label="right_javelin_wrist", color="darkblue", linewidth=0.8)
        axs[4].axvline(playback_index, color='black', linestyle='--', linewidth=0.6)
        axs[4].set_xlim(start, end)
        axs[4].legend()
        axs[4].set_title("Wrist–Object Distance")
        # Mark sharp increases (release candidates) in the plot
        for idx in left_sustained_indices:
            if start <= idx < end:
                axs[4].axvline(idx, color="green", linestyle=":", linewidth=1, label='Left release' if idx==left_sustained_indices[0] else "")
        
        for idx in right_sustained_indices:
            if start <= idx < end:
                axs[4].axvline(idx, color="darkblue", linestyle=":", linewidth=1, label='Right release' if idx==right_sustained_indices[0] else "")
        

        axs[8].text(
            0.01, 0.01, f"FPS: {fps:.2f}",
            transform=axs[8].transAxes, fontsize=9,
            color="black", ha="left", va="bottom",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.6)
        )
            
        # Visualize final computed release_idx (thick, red, labeled)
        if release_idx is not None and start <= release_idx < end:
            axs[4].axvline(release_idx, color="red", linestyle="-", linewidth=2.5, label='Final release')
            axs[4].legend()

            
        plt.tight_layout()
        fig.subplots_adjust(hspace=0.6)
        show_sharp_plot(fig, graph_placeholder)

        
        # time.sleep(0.01 / fps)
        playback_index += 1
        st.session_state.playback_index = playback_index



    # After playback ends or is paused, save current frame (do NOT reset)
    st.session_state.play = False
    st.session_state.frame_display = st.session_state.playback_index
    # Ensure frame_idx corresponds to the slider position (index in visible_frames)

    
else:
    selected_frame = visible_frames[st.session_state.frame_idx]
    cap.set(cv2.CAP_PROP_POS_FRAMES, selected_frame)
    ret, frame = cap.read()
    if ret:
        frame_data = person_df[person_df.Frame == selected_frame]
        for _, row in frame_data.iterrows():
            frame = draw_pose_on_frame(frame, row,
                ["left_elbow", "right_elbow", "left_knee", "right_knee", "left_ankle", "right_ankle",
                 "left_hip", "right_hip", "left_shoulder", "right_shoulder", "left_wrist", "right_wrist"],
                [("left_shoulder", "right_shoulder"), ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
                 ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
                 ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
                 ("left_hip", "right_hip"), ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
                 ("right_hip", "right_knee"), ("right_knee", "right_ankle")],
                point_color=point_color, line_color=line_color
            )
            frame = draw_object_parts(frame, row, object_parts)

        # --- High-res export button for current frame ---
        export_png = st.button("💾 Save Current Frame (High-Res PNG)")
        if export_png:
            export_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert for correct colors
            is_success, buffer_png = cv2.imencode(".png", export_frame)
            if is_success:
                b64 = base64.b64encode(buffer_png).decode()
                href = f'<a href="data:file/png;base64,{b64}" download="frame_{selected_frame}_id{person_id}.png">📥 Click here to download</a>'
                st.markdown(href, unsafe_allow_html=True)
            else:
                st.error("Could not encode frame for download.")

        
        _, buffer = cv2.imencode('.jpg', cv2.resize(frame, (960, 540)))


        frame_placeholder.image(buffer.tobytes(), caption=f"Frame {selected_frame}")

        
    
    half_window = window_size // 2
    center = frame_to_row.get(selected_frame, 0)
    
    start = max(0, center - half_window)
    end = min(len(coords_df), center + half_window)
    if end > len(plot_df):
        end = len(plot_df)
        start = max(0, end - window_size)
    fig, axs = plt.subplots(2, 5, figsize=(17, 4.5), dpi=600)
    axs = axs.flatten()  # Make it a flat list of 8 axes
    window_df = coords_df.iloc[start:end]

    # (Paste your full plotting code here but replace st.session_state.frame_idx with just i)
    axs[0].plot(window_df["elbow_l"], label="Left Elbow", color="red",linewidth=0.8)
    axs[0].plot(window_df["elbow_r"], label="Right Elbow", color="blue",linewidth=0.8)
    axs[0].axvline(st.session_state.frame_idx, color='black', linestyle='--', linewidth=0.6)
    axs[0].set_xlim(start, end)
    axs[0].legend()
    axs[0].set_title("Elbow Angles")

    axs[5].plot(window_df["knee_l"], label="Left Knee", color="purple", linewidth=0.8)
    axs[5].plot(window_df["knee_r"], label="Right Knee", color="seagreen", linewidth=0.8)
    axs[5].axvline(st.session_state.frame_idx, color='black', linestyle='--', linewidth=0.6)
    axs[5].set_xlim(start, end)
    axs[5].legend()
    axs[5].set_title("Knee Angles")

    axs[1].plot(window_df["left_ankle_y"], label="Left Ankle Y", color="orange", linewidth=0.8)
    axs[1].plot(window_df["right_ankle_y"], label="Right Ankle Y", color="lightblue", linewidth=0.8)
    axs[1].invert_yaxis()
    axs[1].axvline(st.session_state.frame_idx, color='black', linestyle='--', linewidth=0.6)
    axs[1].set_xlim(start, end)
    axs[1].legend()
    left_frames_ankley = [frame for frame in left_refined if start <= frame <= end]
    right_frames_ankley = [frame for frame in right_refined if start <= frame <= end]
    left_y_vals_ankley = [coords_df.loc[frame, "left_ankle_y"] for frame in left_frames_ankley]
    right_y_vals_ankley = [coords_df.loc[frame, "right_ankle_y"] for frame in right_frames_ankley]
    axs[1].scatter(left_frames_ankley, left_y_vals_ankley, color="orange", s=60, zorder=6, label="Refined L Contact")
    axs[1].scatter(right_frames_ankley, right_y_vals_ankley, color="brown", s=60, zorder=6, label="Refined R Contact")
    axs[1].set_title("Ankle Y Coordinates")

    axs[6].plot(window_df["left_hip_y"], label="Left Hip Y", color="brown", linewidth=0.8)
    axs[6].plot(window_df["right_hip_y"], label="Right Hip Y", color="springgreen", linewidth=0.8)
    axs[6].invert_yaxis()
    axs[6].axvline(st.session_state.frame_idx, color='black', linestyle='--', linewidth=0.6)
    axs[6].set_xlim(start, end)
    axs[6].legend()
    axs[6].set_title("Hip Y Coordinates")

    axs[2].plot(window_df["left_ankle_x"], label="Left Ankle X", color="orange", linewidth=0.8)
    axs[2].plot(window_df["right_ankle_x"], label="Right Ankle X", color="lightblue", linewidth=0.8)
    axs[2].axvline(st.session_state.frame_idx, color='black', linestyle='--', linewidth=0.6)
    axs[2].set_xlim(start, end)
    axs[2].legend()
    axs[2].set_title("Ankle X Coordinates")

    axs[7].plot(window_df["left_hip_x"], label="Left Hip X", color="brown", linewidth=0.8)
    axs[7].plot(window_df["right_hip_x"], label="Right Hip X", color="springgreen", linewidth=0.8)
    axs[7].axvline(st.session_state.frame_idx, color='black', linestyle='--', linewidth=0.6)
    axs[7].set_xlim(start, end)
    axs[7].legend()

    # Annotate slope and direction
    slope_txt = f"Slope: {slope:.2f} ({direction})"
    axs[7].text(0.05, 0.95, slope_txt, transform=axs[7].transAxes,
                fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    axs[7].set_title("Hip X Coordinates")


    axs[3].plot(window_df["left_ankle_rel_x"], label="left_hip_ank_dist", color="orange", linewidth=0.8)
    axs[3].plot(window_df["right_ankle_rel_x"], label="right_hip_ank_dist", color="brown", linewidth=0.8)
    axs[3].axvline(st.session_state.frame_idx, color='black', linestyle='--', linewidth=0.6)
    axs[3].set_xlim(start, end)
    axs[3].legend()

    # Extract final contact frames and y-values
    left_frames = [frame for frame, _, _, side in contacts if side == "L" and start <= frame <= end]
    left_y_vals = [coords_df.loc[frame, "left_ankle_rel_x"] for frame in left_frames]
    
    right_frames = [frame for frame, _, _, side in contacts if side == "R" and start <= frame <= end]
    right_y_vals = [coords_df.loc[frame, "right_ankle_rel_x"] for frame in right_frames]
    
    axs[3].scatter(left_frames, left_y_vals, color="orange", s=60, zorder=6, label="Final Left Contact")
    axs[3].scatter(right_frames, right_y_vals, color="brown", s=60, zorder=6, label="Final Right Contact")

    for i, step in enumerate(step_events):
        # Calculate midpoint x value between contacts
        x_plot = (step["from_frame"] + step["to_frame"]) / 2
    
        # Get corresponding y values for plotting annotation
        y0 = left_diff[step["from_frame"]] if step["from_side"] == "L" else right_diff[step["from_frame"]]
        y1 = left_diff[step["to_frame"]]   if step["to_side"]   == "L" else right_diff[step["to_frame"]]
        y_plot = (y0 + y1) / 2
    
        # Annotate step type ("long" or "short")
        axs[3].text(
            x_plot, y_plot, step["type"],
            fontsize=11, color="black",
            ha="center", va="bottom", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none")
        )
        
    axs[3].set_title("Hip_Ankle_x_difference")


    # Plot wrist–object distances
    axs[4].plot(window_df["dist_javelin_to_left_wrist"], label="left_javelin_wrist", color="green", linewidth=0.8)
    axs[4].plot(window_df["dist_javelin_to_right_wrist"], label="right_javelin_wrist", color="darkblue", linewidth=0.8)
    axs[4].axvline(st.session_state.frame_idx, color='black', linestyle='--', linewidth=0.6)
    axs[4].set_xlim(start, end)
    axs[4].legend()
    axs[4].set_title("Wrist–Object Distance")

    # Mark sharp increases (release candidates) in the plot
    for idx in left_sustained_indices:
        if start <= idx < end:
            axs[4].axvline(idx, color="green", linestyle=":", linewidth=1, label='Left release' if idx==left_sustained_indices[0] else "")
    
    for idx in right_sustained_indices:
        if start <= idx < end:
            axs[4].axvline(idx, color="darkblue", linestyle=":", linewidth=1, label='Right release' if idx==right_sustained_indices[0] else "")
    

    axs[8].text(
        0.01, 0.01, f"FPS: {fps:.2f}",
        transform=axs[8].transAxes, fontsize=9,
        color="black", ha="left", va="bottom",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.6)
    )

    # Visualize final computed release_idx (thick, red, labeled)
    if release_idx is not None and start <= release_idx < end:
        axs[4].axvline(release_idx, color="red", linestyle="-", linewidth=2.5, label='Final release')
        axs[4].legend()
    
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.6)
    show_sharp_plot(fig, graph_placeholder)

    download_graph = st.button("💾 Download Graph (High-Res PNG)")
    if download_graph:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=600, bbox_inches="tight", pad_inches=0.05)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode()
        href = f'<a href="data:file/png;base64,{b64}" download="graph_frame_{selected_frame}_id{person_id}.png">📥 Click here to download</a>'
        st.markdown(href, unsafe_allow_html=True)



