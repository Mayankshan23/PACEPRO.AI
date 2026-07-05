import { useState } from "react";
import Sidebar from "./components/Sidebar";
import BowlingAnalyzer from "./components/BowlingAnalyzer";
import ChatWindow from "./components/ChatWindow";

export default function App() {
  const [activeSection, setActiveSection] = useState("bowling");
  const [report, setReport] = useState(null);

  return (
    <div className="flex h-screen w-screen bg-[#0D1016] text-white font-sans">
      <Sidebar activeId={activeSection} onSelect={setActiveSection} />

      {activeSection === "bowling" && (
        <BowlingAnalyzer onReportReady={setReport} />
      )}

      <ChatWindow report={report} />
    </div>
  );
}
