import { useState } from "react";
import { Loader2, TriangleAlert, PlayCircle } from "lucide-react";
import UploadSlot from "./UploadSlot";
import { analyzeBowling } from "../services/api";

const VIEW_CONFIG = [
  { key: "front", label: "Front View", hint: "Facing the bowler head-on" },
  { key: "back", label: "Back View", hint: "Behind the bowler, run-up angle" },
  { key: "side", label: "Side View", hint: "Perpendicular, best for phases" },
];

const METRIC_LABELS = {
  front_knee_flexion_ffc_deg: "Front Knee Flexion (FFC)",
  trunk_forward_flexion_br_deg: "Trunk Forward Flexion (Release)",
  elbow_extension_br_deg: "Elbow Extension (Release)",
  trunk_lateral_flexion_br_deg: "Trunk Lateral Flexion (Release)",
  shoulder_hip_separation_deg: "Shoulder-Hip Separation",
  front_foot_landing_angle_deg: "Front Foot Landing Angle",
  arm_path_deviation_deg: "Arm Path Deviation",
  run_up_cadence_steps_per_sec: "Run-up Cadence (steps/sec)",
};

const PHASE_LABELS = ["Run-up", "Back Foot Contact", "Front Foot Contact", "Ball Release", "Follow-through"];

export default function BowlingAnalyzer({ onReportReady }) {
  const [files, setFiles] = useState({ front: null, back: null, side: null });
  const [status, setStatus] = useState("idle"); // idle | loading | done | error
  const [error, setError] = useState(null);
  const [report, setReport] = useState(null);

  const uploadedCount = Object.values(files).filter(Boolean).length;

  const handleStart = async () => {
    if (uploadedCount === 0) return;
    setStatus("loading");
    setError(null);
    try {
      const result = await analyzeBowling(files);
      setReport(result);
      onReportReady?.(result);
      setStatus("done");
    } catch (e) {
      setError(e.message);
      setStatus("error");
    }
  };

  return (
    <div className="flex-1 overflow-y-auto px-8 py-8">
      <header className="mb-8">
        <p className="font-condensed text-3xl uppercase tracking-wide text-white">Bowling Analyzer</p>
        <p className="text-sm text-white/50 mt-1">
          Upload up to three camera angles. One is enough to run — more views sharpen the read.
        </p>
      </header>

      <section className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-6">
        {VIEW_CONFIG.map(({ key, label, hint }) => (
          <UploadSlot
            key={key}
            label={label}
            hint={hint}
            file={files[key]}
            onChange={(f) => setFiles((prev) => ({ ...prev, [key]: f }))}
          />
        ))}
      </section>

      <div className="flex items-center gap-4 mb-10">
        <button
          onClick={handleStart}
          disabled={uploadedCount === 0 || status === "loading"}
          className={[
            "flex items-center gap-2 rounded-md px-5 py-2.5 font-condensed uppercase tracking-wide text-base",
            "transition-colors",
            uploadedCount === 0
              ? "bg-white/5 text-white/25 cursor-not-allowed"
              : "bg-[#B33A3A] text-white hover:bg-[#B33A3A]/85",
          ].join(" ")}
        >
          {status === "loading" ? (
            <Loader2 size={18} className="animate-spin" />
          ) : (
            <PlayCircle size={18} />
          )}
          {status === "loading" ? "Analyzing…" : "Start Analysis"}
        </button>
        {uploadedCount > 0 && uploadedCount < 3 && status !== "loading" && (
          <p className="text-xs text-white/40 font-mono flex items-center gap-1.5">
            <TriangleAlert size={13} className="text-amber-500" />
            {uploadedCount} of 3 views — result will be approximated
          </p>
        )}
      </div>

      {status === "error" && (
        <div className="rounded-md border border-[#B33A3A]/40 bg-[#B33A3A]/10 px-4 py-3 text-sm text-[#e08a8a] mb-8">
          {error}
        </div>
      )}

      {report && <ReportView report={report} />}
    </div>
  );
}

function ReportView({ report }) {
  const isLow = report.analysis_confidence === "low";
  return (
    <section>
      <div className="flex items-center gap-3 mb-6">
        <p className="font-condensed text-xl uppercase tracking-wide text-white">Report</p>
        <span
          className={[
            "text-[10px] uppercase tracking-wider font-mono px-2 py-1 rounded",
            isLow ? "bg-amber-500/15 text-amber-400" : "bg-[#4C7A5E]/20 text-[#4C7A5E]",
          ].join(" ")}
        >
          confidence: {report.analysis_confidence}
        </span>
        {report.warning && (
          <span className="text-xs text-white/40 font-mono">{report.warning}</span>
        )}
      </div>

      {/* Phase timeline — signature element */}
      <div className="mb-8">
        <p className="text-xs uppercase tracking-wide text-white/40 font-mono mb-2">Delivery phases</p>
        <div className="flex rounded-md overflow-hidden border border-white/10">
          {PHASE_LABELS.map((label, i) => (
            <div
              key={label}
              className="flex-1 px-3 py-3 text-center border-r border-white/10 last:border-r-0 bg-white/[0.02]"
            >
              <p className="font-mono text-[10px] text-white/30 mb-1">0{i + 1}</p>
              <p className="text-xs text-white/70">{label}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Metrics grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-10">
        {Object.entries(report.metrics || {}).map(([key, value]) => (
          <div key={key} className="rounded-md border border-white/10 bg-white/[0.02] px-4 py-3">
            <p className="text-[10px] uppercase tracking-wide text-white/40 font-mono mb-1">
              {METRIC_LABELS[key] || key}
            </p>
            <p className="font-condensed text-2xl text-white">
              {value !== null && value !== undefined ? value : "—"}
              {value !== null && value !== undefined && key.endsWith("_deg") && (
                <span className="text-sm text-white/40">°</span>
              )}
            </p>
          </div>
        ))}
      </div>

      {/* Processed videos */}
      <div>
        <p className="text-xs uppercase tracking-wide text-white/40 font-mono mb-3">
          Skeleton overlay playback
        </p>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          {Object.entries(report.video_urls || {}).map(([view, url]) => (
            <div key={view} className="rounded-md overflow-hidden border border-white/10 bg-black">
              <video src={url} controls className="w-full aspect-video bg-black" />
              <p className="px-3 py-2 text-xs font-mono text-white/50 uppercase tracking-wide">
                {view} view
              </p>
            </div>
          ))}
          {Object.keys(report.video_urls || {}).length === 0 && (
            <p className="text-sm text-white/30 col-span-3">No processed video available.</p>
          )}
        </div>
      </div>
    </section>
  );
}
