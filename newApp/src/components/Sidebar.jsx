import { Activity, Apple, Gauge } from "lucide-react";

const NAV_ITEMS = [
  { id: "bowling", label: "Bowling Analyzer", icon: Activity, enabled: true },
  { id: "dietitian", label: "Dietitian", icon: Apple, enabled: false },
  { id: "workload", label: "Workload Management", icon: Gauge, enabled: false },
];

export default function Sidebar({ activeId, onSelect }) {
  return (
    <aside className="w-60 shrink-0 bg-[#12161C] border-r border-white/5 flex flex-col">
      <div className="px-5 py-6 border-b border-white/5">
        <p className="font-condensed text-2xl tracking-wide text-white uppercase">
          Pace<span className="text-[#B33A3A]">Lab</span>
        </p>
        <p className="text-[11px] text-white/40 font-mono mt-1">biomechanics console</p>
      </div>

      <nav className="flex-1 px-3 py-4 space-y-1">
        {NAV_ITEMS.map(({ id, label, icon: Icon, enabled }) => {
          const isActive = activeId === id;
          return (
            <button
              key={id}
              disabled={!enabled}
              onClick={() => enabled && onSelect(id)}
              className={[
                "w-full flex items-center gap-3 rounded-md px-3 py-2.5 text-sm transition-colors",
                enabled
                  ? isActive
                    ? "bg-[#4C7A5E]/15 text-white"
                    : "text-white/70 hover:bg-white/5 hover:text-white"
                  : "text-white/25 cursor-not-allowed",
              ].join(" ")}
            >
              <Icon size={17} strokeWidth={1.75} className={isActive ? "text-[#4C7A5E]" : ""} />
              <span className="flex-1 text-left">{label}</span>
              {!enabled && (
                <span className="text-[9px] uppercase tracking-wide font-mono bg-white/5 text-white/40 px-1.5 py-0.5 rounded">
                  Soon
                </span>
              )}
            </button>
          );
        })}
      </nav>

      <div className="px-5 py-4 border-t border-white/5 text-[11px] text-white/30 font-mono">
        session-safe · files auto-purged
      </div>
    </aside>
  );
}
