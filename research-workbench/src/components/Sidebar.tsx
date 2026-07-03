"use client";

import React from "react";
import {
  Home,
  BookOpen,
  Database,
  Play,
  BarChart2,
  Sliders,
  User,
  ArrowRightLeft,
  ShieldAlert,
  Activity,
  Award,
  FileText,
  Copy,
  ChevronRight
} from "lucide-react";

interface SidebarProps {
  currentSection: string;
  setSection: (section: string) => void;
}

export default function Sidebar({ currentSection, setSection }: SidebarProps) {
  const menuItems = [
    { id: "home", label: "Overview", icon: Home },
    { id: "narrative", label: "Research Narrative", icon: BookOpen },
    { id: "datasets", label: "Datasets Profiler", icon: Database },
    { id: "experiments", label: "Experiment Console", icon: Play },
    { id: "performance", label: "Performance & CIs", icon: BarChart2 },
    { id: "calibration", label: "Calibration Repair", icon: Sliders },
    { id: "patient_explorer", label: "Patient SHAP Explorer", icon: User },
    { id: "explanation_drift", label: "Explanation Drift", icon: ArrowRightLeft },
    { id: "robustness", label: "Stress Testing", icon: ShieldAlert },
    { id: "clinical_utility", label: "Clinical Utility (DCA)", icon: Activity },
    { id: "validation_radar", label: "Validation Audit", icon: Award },
    { id: "publication", label: "Publication Mode", icon: Copy },
    { id: "reports", label: "Reports & Export", icon: FileText },
    { id: "deployment", label: "Deployment Readiness", icon: ChevronRight }
  ];

  return (
    <div className="w-64 bg-[#111827] border-r border-[#1E293B] flex flex-col h-screen text-slate-300">
      {/* Brand Header */}
      <div className="p-6 border-b border-[#1E293B] flex items-center gap-3">
        <div className="w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center font-bold text-white shadow-md shadow-blue-900/50">
          Æ
        </div>
        <div>
          <h1 className="font-bold text-white tracking-wider text-sm">AETHEL Studio</h1>
          <p className="text-[10px] text-slate-500 font-mono">v3.0.0 · AUDITING ENGINE</p>
        </div>
      </div>

      {/* Main Menu */}
      <div className="flex-1 py-4 px-4 space-y-1 overflow-y-auto">
        <p className="px-3 text-[9px] font-bold text-slate-500 tracking-wider uppercase mb-2">
          Auditing Modules
        </p>
        {menuItems.map((item) => {
          const Icon = item.icon;
          const isActive = currentSection === item.id;
          return (
            <button
              key={item.id}
              onClick={() => setSection(item.id)}
              className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-[11px] font-medium transition-all duration-150 ${
                isActive
                  ? "bg-blue-600/10 text-blue-400 border border-blue-500/20 shadow-[inset_0_1px_0_0_rgba(255,255,255,0.05)]"
                  : "hover:bg-slate-800/50 text-slate-400 border border-transparent hover:text-slate-200"
              }`}
            >
              <Icon className={`w-3.5 h-3.5 ${isActive ? "text-blue-400" : "text-slate-500"}`} />
              {item.label}
            </button>
          );
        })}
      </div>

      {/* Secondary Indicators */}
      <div className="p-4 border-t border-[#1E293B] space-y-2.5 font-mono text-[9px] text-slate-500">
        <div className="flex justify-between items-center">
          <span>SEED</span>
          <span className="text-slate-400">42 / 123</span>
        </div>
        <div className="flex justify-between items-center">
          <span>REPRODUCIBILITY</span>
          <span className="text-emerald-500 flex items-center gap-1">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></span>
            VERIFIED
          </span>
        </div>
      </div>
    </div>
  );
}
