import { useRef } from "react";
import { UploadCloud, CheckCircle2, XCircle } from "lucide-react";

export default function UploadSlot({ label, hint, file, onChange }) {
  const inputRef = useRef(null);

  const handleFile = (e) => {
    const f = e.target.files?.[0];
    if (f) onChange(f);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const f = e.dataTransfer.files?.[0];
    if (f && f.type.startsWith("video/")) onChange(f);
  };

  return (
    <div
      onDragOver={(e) => e.preventDefault()}
      onDrop={handleDrop}
      onClick={() => inputRef.current?.click()}
      className={[
        "relative flex flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed",
        "px-4 py-8 cursor-pointer transition-colors min-h-[160px]",
        file ? "border-[#4C7A5E]/60 bg-[#4C7A5E]/[0.06]" : "border-white/10 hover:border-white/25 bg-white/[0.02]",
      ].join(" ")}
    >
      <input
        ref={inputRef}
        type="file"
        accept="video/*"
        className="hidden"
        onChange={handleFile}
      />
      {file ? (
        <CheckCircle2 size={26} className="text-[#4C7A5E]" strokeWidth={1.5} />
      ) : (
        <UploadCloud size={26} className="text-white/30" strokeWidth={1.5} />
      )}
      <p className="font-condensed text-lg tracking-wide text-white uppercase">{label}</p>
      <p className="text-xs text-white/40 text-center font-mono">
        {file ? file.name : hint}
      </p>
      {file && (
        <button
          onClick={(e) => {
            e.stopPropagation();
            onChange(null);
          }}
          className="absolute top-2 right-2 text-white/30 hover:text-[#B33A3A] transition-colors"
        >
          <XCircle size={18} />
        </button>
      )}
    </div>
  );
}
