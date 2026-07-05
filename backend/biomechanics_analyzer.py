"""
biomechanics_analyzer.py

Class-based, multi-view (Front / Back / Side) cricket bowling biomechanics analyzer.

Pipeline per video:
    1. Run MediaPipe Pose over every frame.
    2. Draw the skeleton overlay and write it to an output video (mp4v codec).
    3. Detect the 5 classic bowling phases from the landmark time-series:
       Run-up -> Back Foot Contact (BFC) -> Front Foot Contact (FFC) ->
       Ball Release (BR) -> Follow-Through (FT)
    4. Compute joint-angle metrics at the biomechanically relevant phase for
       each metric (e.g. front knee flexion at FFC, elbow extension at BR).

Multi-view fusion:
    - SIDE view is preferred for sagittal-plane metrics (knee flexion, trunk
      forward flexion, elbow angle at release, run-up cadence).
    - FRONT view is preferred for frontal-plane metrics (trunk lateral flexion,
      shoulder-hip separation / counter-rotation, front foot landing angle).
    - BACK view is preferred for arm path / bowling-arm circle metrics.
    - If a preferred view is missing, the analyzer falls back to whichever
      view is available and lowers `analysis_confidence`.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "mediapipe is required. Install with: pip install mediapipe"
    ) from exc


VIEW_FRONT = "front"
VIEW_BACK = "back"
VIEW_SIDE = "side"
ALL_VIEWS = (VIEW_FRONT, VIEW_BACK, VIEW_SIDE)

# Preferred source view for each sagittal / frontal metric, in fallback order.
METRIC_VIEW_PRIORITY = {
    "front_knee_flexion_ffc_deg": [VIEW_SIDE, VIEW_FRONT, VIEW_BACK],
    "trunk_forward_flexion_br_deg": [VIEW_SIDE, VIEW_FRONT, VIEW_BACK],
    "elbow_extension_br_deg": [VIEW_SIDE, VIEW_BACK, VIEW_FRONT],
    "trunk_lateral_flexion_br_deg": [VIEW_FRONT, VIEW_BACK, VIEW_SIDE],
    "shoulder_hip_separation_deg": [VIEW_FRONT, VIEW_SIDE, VIEW_BACK],
    "front_foot_landing_angle_deg": [VIEW_FRONT, VIEW_SIDE, VIEW_BACK],
    "arm_path_deviation_deg": [VIEW_BACK, VIEW_SIDE, VIEW_FRONT],
    "run_up_cadence_steps_per_sec": [VIEW_SIDE, VIEW_FRONT, VIEW_BACK],
}


@dataclass
class PhaseFrames:
    """Frame indices for each detected bowling phase (None if undetected)."""
    run_up_start: Optional[int] = None
    back_foot_contact: Optional[int] = None
    front_foot_contact: Optional[int] = None
    ball_release: Optional[int] = None
    follow_through: Optional[int] = None


@dataclass
class ViewResult:
    view: str
    fps: float
    frame_count: int
    landmarks: List[Optional[np.ndarray]] = field(default_factory=list)  # per-frame (33,4) or None
    phases: PhaseFrames = field(default_factory=PhaseFrames)
    output_video_path: Optional[Path] = None
    pose_detection_rate: float = 0.0  # fraction of frames with a valid pose


class BiomechanicsAnalyzer:
    """
    Reusable, stateless-per-call analyzer. Instantiate once (e.g. as a
    FastAPI app-level singleton) and call `.analyze()` per request; MediaPipe
    Pose contexts are opened/closed per video so the instance is thread-safe
    across concurrent requests.
    """

    LANDMARK_CONNECTIONS = None  # set lazily from mediapipe

    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self._mp_pose = mp.solutions.pose
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_styles = mp.solutions.drawing_styles
        self._model_complexity = model_complexity
        self._min_det_conf = min_detection_confidence
        self._min_track_conf = min_tracking_confidence

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def analyze(
        self,
        video_paths: Dict[str, Path],
        output_dir: Path,
    ) -> Tuple[dict, Dict[str, Path]]:
        """
        Args:
            video_paths: mapping of view name ("front" | "back" | "side") to
                the input video file path. 1-3 entries.
            output_dir: directory (already created) where annotated output
                videos are written.

        Returns:
            (report_dict, output_video_paths) where output_video_paths maps
            view name -> annotated output video Path.
        """
        if not video_paths:
            raise ValueError("At least one video (front/back/side) is required.")
        for view in video_paths:
            if view not in ALL_VIEWS:
                raise ValueError(f"Unknown view '{view}'. Expected one of {ALL_VIEWS}.")

        started = time.monotonic()
        results: Dict[str, ViewResult] = {}
        for view, path in video_paths.items():
            out_path = output_dir / f"{view}_skeleton_{uuid.uuid4().hex[:8]}.mp4"
            results[view] = self._process_single_video(path, view, out_path)

        report = self._build_report(results, video_paths.keys())
        report["processing_time_sec"] = round(time.monotonic() - started, 2)

        output_video_paths = {
            view: r.output_video_path for view, r in results.items() if r.output_video_path
        }
        return report, output_video_paths

    # ------------------------------------------------------------------ #
    # Per-video processing
    # ------------------------------------------------------------------ #

    def _process_single_video(self, input_path: Path, view: str, output_path: Path) -> ViewResult:
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise IOError(f"Could not open video for view '{view}': {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # mp4v is the widely-supported, browser-compatible fourcc for .mp4 containers
        # in headless OpenCV builds (H.264 encoders are frequently unavailable/licensed).
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        landmarks_per_frame: List[Optional[np.ndarray]] = []
        detected_frames = 0
        frame_count = 0

        with self._mp_pose.Pose(
            static_image_mode=False,
            model_complexity=self._model_complexity,
            min_detection_confidence=self._min_det_conf,
            min_tracking_confidence=self._min_track_conf,
        ) as pose:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_count += 1

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                result = pose.process(rgb)
                rgb.flags.writeable = True

                if result.pose_landmarks:
                    detected_frames += 1
                    lm = np.array(
                        [[p.x, p.y, p.z, p.visibility] for p in result.pose_landmarks.landmark],
                        dtype=np.float32,
                    )
                    landmarks_per_frame.append(lm)

                    self._mp_drawing.draw_landmarks(
                        frame,
                        result.pose_landmarks,
                        self._mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=self._mp_styles.get_default_pose_landmarks_style(),
                    )
                else:
                    landmarks_per_frame.append(None)

                writer.write(frame)

        cap.release()
        writer.release()

        view_result = ViewResult(
            view=view,
            fps=fps,
            frame_count=frame_count,
            landmarks=landmarks_per_frame,
            output_video_path=output_path,
            pose_detection_rate=(detected_frames / frame_count) if frame_count else 0.0,
        )
        view_result.phases = self._detect_phases(view_result)
        return view_result

    # ------------------------------------------------------------------ #
    # Phase detection (heuristic, per-view)
    # ------------------------------------------------------------------ #

    def _detect_phases(self, vr: ViewResult) -> PhaseFrames:
        """
        Heuristics (deliberately simple / explainable, not ML-trained):

        - Ball release (BR): frame of maximum bowling-arm wrist speed
          (largest displacement between consecutive frames), restricted to
          the back half of the clip where the action typically happens.
        - Front Foot Contact (FFC): last local minimum in front-ankle vertical
          velocity *before* BR (foot plants just before release).
        - Back Foot Contact (BFC): last local minimum in back-ankle vertical
          velocity before FFC.
        - Run-up start: first frame with a valid pose detection.
        - Follow-through: first local minimum in wrist speed after BR
          (arm decelerating), or clip end if none found.
        """
        phases = PhaseFrames()
        valid_idxs = [i for i, lm in enumerate(vr.landmarks) if lm is not None]
        if len(valid_idxs) < 5:
            return phases  # not enough data to detect anything reliably

        phases.run_up_start = valid_idxs[0]

        mp_pose = self._mp_pose.PoseLandmark
        wrist_speed = self._series_speed(vr.landmarks, mp_pose.RIGHT_WRIST.value)
        # also try left wrist and take whichever has higher peak (handles left-arm bowlers)
        wrist_speed_l = self._series_speed(vr.landmarks, mp_pose.LEFT_WRIST.value)
        use_left = np.nanmax(wrist_speed_l) > np.nanmax(wrist_speed) if len(wrist_speed_l) else False
        bowling_wrist_speed = wrist_speed_l if use_left else wrist_speed

        if len(bowling_wrist_speed) == 0 or np.all(np.isnan(bowling_wrist_speed)):
            return phases

        search_start = int(len(bowling_wrist_speed) * 0.4)
        window = bowling_wrist_speed[search_start:]
        if len(window) == 0 or np.all(np.isnan(window)):
            return phases
        br_idx = search_start + int(np.nanargmax(window))
        phases.ball_release = br_idx

        front_ankle = mp_pose.LEFT_ANKLE.value if use_left else mp_pose.RIGHT_ANKLE.value
        back_ankle = mp_pose.RIGHT_ANKLE.value if use_left else mp_pose.LEFT_ANKLE.value

        ffc_idx = self._last_local_min_before(vr.landmarks, front_ankle, br_idx)
        phases.front_foot_contact = ffc_idx if ffc_idx is not None else max(br_idx - 3, 0)

        bfc_search_end = phases.front_foot_contact
        bfc_idx = self._last_local_min_before(vr.landmarks, back_ankle, bfc_search_end)
        phases.back_foot_contact = bfc_idx if bfc_idx is not None else max(bfc_search_end - 5, 0)

        ft_idx = None
        for i in range(br_idx + 1, len(bowling_wrist_speed) - 1):
            if np.isnan(bowling_wrist_speed[i]):
                continue
            if bowling_wrist_speed[i] <= bowling_wrist_speed[br_idx] * 0.3:
                ft_idx = i
                break
        phases.follow_through = ft_idx if ft_idx is not None else valid_idxs[-1]

        return phases

    @staticmethod
    def _series_speed(landmarks: List[Optional[np.ndarray]], landmark_idx: int) -> np.ndarray:
        pts = []
        for lm in landmarks:
            pts.append(lm[landmark_idx, :2] if lm is not None else np.array([np.nan, np.nan]))
        pts = np.array(pts)
        speed = np.full(len(pts), np.nan)
        for i in range(1, len(pts)):
            if np.any(np.isnan(pts[i])) or np.any(np.isnan(pts[i - 1])):
                continue
            speed[i] = np.linalg.norm(pts[i] - pts[i - 1])
        return speed

    @staticmethod
    def _last_local_min_before(
        landmarks: List[Optional[np.ndarray]], landmark_idx: int, before_frame: int
    ) -> Optional[int]:
        ys = []
        for lm in landmarks[:before_frame]:
            ys.append(lm[landmark_idx, 1] if lm is not None else np.nan)
        ys = np.array(ys)
        if len(ys) < 3:
            return None
        candidates = [
            i for i in range(1, len(ys) - 1)
            if not np.isnan(ys[i - 1]) and not np.isnan(ys[i]) and not np.isnan(ys[i + 1])
            and ys[i] >= ys[i - 1] and ys[i] >= ys[i + 1]  # local max in y == foot lowest on screen
        ]
        return candidates[-1] if candidates else None

    # ------------------------------------------------------------------ #
    # Angle geometry helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """Angle at vertex b formed by points a-b-c, in degrees (2D, image plane)."""
        ba = a[:2] - b[:2]
        bc = c[:2] - b[:2]
        denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) or 1e-8
        cosang = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
        return float(np.degrees(np.arccos(cosang)))

    def _landmarks_at(self, vr: ViewResult, frame_idx: Optional[int]) -> Optional[np.ndarray]:
        if frame_idx is None or frame_idx >= len(vr.landmarks):
            return None
        return vr.landmarks[frame_idx]

    # ------------------------------------------------------------------ #
    # Metric computation per view, then fusion
    # ------------------------------------------------------------------ #

    def _metrics_for_view(self, vr: ViewResult) -> Dict[str, Optional[float]]:
        mp_pose = self._mp_pose.PoseLandmark
        metrics: Dict[str, Optional[float]] = {k: None for k in METRIC_VIEW_PRIORITY}

        ffc_lm = self._landmarks_at(vr, vr.phases.front_foot_contact)
        br_lm = self._landmarks_at(vr, vr.phases.ball_release)

        if ffc_lm is not None:
            # Try right leg first, fall back to left (front leg depends on bowler handedness/orientation)
            for hip_i, knee_i, ank_i in (
                (mp_pose.RIGHT_HIP.value, mp_pose.RIGHT_KNEE.value, mp_pose.RIGHT_ANKLE.value),
                (mp_pose.LEFT_HIP.value, mp_pose.LEFT_KNEE.value, mp_pose.LEFT_ANKLE.value),
            ):
                if ffc_lm[knee_i, 3] > 0.5:
                    metrics["front_knee_flexion_ffc_deg"] = self._angle_deg(
                        ffc_lm[hip_i], ffc_lm[knee_i], ffc_lm[ank_i]
                    )
                    break

            metrics["front_foot_landing_angle_deg"] = self._foot_landing_angle(ffc_lm, mp_pose)

        if br_lm is not None:
            shoulder = br_lm[mp_pose.RIGHT_SHOULDER.value]
            hip_mid = (br_lm[mp_pose.LEFT_HIP.value] + br_lm[mp_pose.RIGHT_HIP.value]) / 2
            knee_mid = (br_lm[mp_pose.LEFT_KNEE.value] + br_lm[mp_pose.RIGHT_KNEE.value]) / 2
            metrics["trunk_forward_flexion_br_deg"] = self._angle_deg(shoulder, hip_mid, knee_mid)

            l_sh, r_sh = br_lm[mp_pose.LEFT_SHOULDER.value], br_lm[mp_pose.RIGHT_SHOULDER.value]
            l_hip, r_hip = br_lm[mp_pose.LEFT_HIP.value], br_lm[mp_pose.RIGHT_HIP.value]
            trunk_top = (l_sh + r_sh) / 2
            trunk_bottom = (l_hip + r_hip) / 2
            vertical_ref = trunk_bottom.copy()
            vertical_ref[1] -= 0.3
            metrics["trunk_lateral_flexion_br_deg"] = self._angle_deg(trunk_top, trunk_bottom, vertical_ref)

            shoulder_vec = r_sh[:2] - l_sh[:2]
            hip_vec = r_hip[:2] - l_hip[:2]
            denom = (np.linalg.norm(shoulder_vec) * np.linalg.norm(hip_vec)) or 1e-8
            cosang = np.clip(np.dot(shoulder_vec, hip_vec) / denom, -1.0, 1.0)
            metrics["shoulder_hip_separation_deg"] = float(np.degrees(np.arccos(cosang)))

            for sh_i, el_i, wr_i in (
                (mp_pose.RIGHT_SHOULDER.value, mp_pose.RIGHT_ELBOW.value, mp_pose.RIGHT_WRIST.value),
                (mp_pose.LEFT_SHOULDER.value, mp_pose.LEFT_ELBOW.value, mp_pose.LEFT_WRIST.value),
            ):
                if br_lm[el_i, 3] > 0.5:
                    metrics["elbow_extension_br_deg"] = self._angle_deg(
                        br_lm[sh_i], br_lm[el_i], br_lm[wr_i]
                    )
                    break

        metrics["arm_path_deviation_deg"] = self._arm_path_deviation(vr, mp_pose)
        metrics["run_up_cadence_steps_per_sec"] = self._run_up_cadence(vr, mp_pose)

        return metrics

    def _foot_landing_angle(self, lm: np.ndarray, mp_pose) -> Optional[float]:
        for heel_i, toe_i in (
            (mp_pose.RIGHT_HEEL.value, mp_pose.RIGHT_FOOT_INDEX.value),
            (mp_pose.LEFT_HEEL.value, mp_pose.LEFT_FOOT_INDEX.value),
        ):
            heel, toe = lm[heel_i], lm[toe_i]
            if heel[3] > 0.5 and toe[3] > 0.5:
                vec = toe[:2] - heel[:2]
                return float(np.degrees(np.arctan2(vec[1], vec[0])))
        return None

    def _arm_path_deviation(self, vr: ViewResult, mp_pose) -> Optional[float]:
        """Std-dev of bowling wrist's lateral (x) position across the delivery
        stride -> release window, as a proxy for arm-path consistency/deviation."""
        start = vr.phases.back_foot_contact
        end = vr.phases.ball_release
        if start is None or end is None or end <= start:
            return None
        xs = []
        for lm in vr.landmarks[start:end + 1]:
            if lm is None:
                continue
            r_wrist, l_wrist = lm[mp_pose.RIGHT_WRIST.value], lm[mp_pose.LEFT_WRIST.value]
            wrist = r_wrist if r_wrist[3] >= l_wrist[3] else l_wrist
            xs.append(wrist[0])
        if len(xs) < 3:
            return None
        return float(np.degrees(np.std(xs) * np.pi))  # scaled proxy, not a true angle

    def _run_up_cadence(self, vr: ViewResult, mp_pose) -> Optional[float]:
        """Approximate steps/sec using ankle-crossing frequency during run-up."""
        start = vr.phases.run_up_start
        end = vr.phases.back_foot_contact
        if start is None or end is None or end - start < vr.fps:  # need >=1s of data
            return None
        crossings = 0
        prev_diff = None
        for lm in vr.landmarks[start:end + 1]:
            if lm is None:
                continue
            diff = lm[mp_pose.LEFT_ANKLE.value, 0] - lm[mp_pose.RIGHT_ANKLE.value, 0]
            if prev_diff is not None and (diff * prev_diff) < 0:
                crossings += 1
            prev_diff = diff
        duration_sec = (end - start) / vr.fps
        if duration_sec <= 0:
            return None
        return round((crossings / 2) / duration_sec, 2)  # 2 crossings ~= 1 step cycle

    # ------------------------------------------------------------------ #
    # Report assembly
    # ------------------------------------------------------------------ #

    def _build_report(self, results: Dict[str, ViewResult], uploaded_views) -> dict:
        uploaded_views = set(uploaded_views)
        per_view_metrics = {view: self._metrics_for_view(vr) for view, vr in results.items()}

        fused: Dict[str, Optional[float]] = {}
        source_view: Dict[str, Optional[str]] = {}
        for metric_name, priority in METRIC_VIEW_PRIORITY.items():
            value, chosen_view = None, None
            for view in priority:
                if view in per_view_metrics and per_view_metrics[view].get(metric_name) is not None:
                    value = per_view_metrics[view][metric_name]
                    chosen_view = view
                    break
            fused[metric_name] = round(value, 2) if isinstance(value, float) else value
            source_view[metric_name] = chosen_view

        num_views = len(uploaded_views)
        avg_detection_rate = (
            sum(vr.pose_detection_rate for vr in results.values()) / len(results) if results else 0.0
        )
        low_confidence = num_views < 3 or avg_detection_rate < 0.6

        report = {
            "views_analyzed": sorted(uploaded_views),
            "views_missing": sorted(set(ALL_VIEWS) - uploaded_views),
            "phase_frames": {
                view: {
                    "run_up_start": vr.phases.run_up_start,
                    "back_foot_contact": vr.phases.back_foot_contact,
                    "front_foot_contact": vr.phases.front_foot_contact,
                    "ball_release": vr.phases.ball_release,
                    "follow_through": vr.phases.follow_through,
                    "fps": round(vr.fps, 2),
                    "pose_detection_rate": round(vr.pose_detection_rate, 3),
                }
                for view, vr in results.items()
            },
            "metrics": fused,
            "metric_source_view": source_view,
            "analysis_confidence": "low" if low_confidence else "high",
        }

        if num_views < 3:
            report["warning"] = "Partial data: Results are approximated"
        elif avg_detection_rate < 0.6:
            report["warning"] = (
                "Pose detection was unreliable in parts of the footage "
                "(occlusion, poor lighting, or fast motion blur): Results are approximated"
            )

        return report
