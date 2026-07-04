"use client";

import React from "react";
import {
  BookOpen,
  Database,
  BarChart2,
  User,
  ArrowRightLeft,
  ShieldAlert,
  Activity,
  Copy,
  ToggleLeft,
  ToggleRight,
  CheckCircle
} from "lucide-react";

interface SidebarProps {
  currentSection: string;
  setSection: (section: string) => void;
  isPresentationMode: boolean;
  setIsPresentationMode: (val: boolean) => void;
}

export default function Sidebar({
  currentSection,
  setSection,
  isPresentationMode,
  setIsPresentationMode
}: SidebarProps) {
  
  // 5 steps for Guided Research Tour
  const presentationItems = [
    { id: "narrative", label: "Research Overview", icon: BookOpen },
    { id: "contributions", label: "Scientific Contributions", icon: CheckCircle },
    { id: "explorer", label: "Clinical Inspectability", icon: User },
    { id: "evidence", label: "Scientific Evidence", icon: BarChart2 },
    { id: "gallery", label: "Publication Figure Gallery", icon: Copy }
  ];

  // 9 steps for Research Workspace
  const researchItems = [
    { id: "narrative", label: "1. Research Overview", icon: BookOpen },
    { id: "contributions", label: "2. Scientific Contributions", icon: CheckCircle },
    { id: "cohorts", label: "3. Cohort Characteristics", icon: Database },
    { id: "evidence", label: "4. Scientific Evidence", icon: BarChart2 },
    { id: "explorer", label: "5. Clinical Inspectability", icon: User },
    { id: "drift", label: "6. Explanation Drift", icon: ArrowRightLeft },
    { id: "stress", label: "7. Robustness Stress Testing", icon: ShieldAlert },
    { id: "gallery", label: "8. Publication Figure Gallery", icon: Copy },
    { id: "publication", label: "9. Publication Assets & Runs", icon: Activity }
  ];

  const menuItems = isPresentationMode ? presentationItems : researchItems;

  return (
    <div className="w-64 bg-[#111827] border-r border-[#1E293B] flex flex-col h-screen text-slate-300 select-none">
      
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

      {/* Mode Switcher Toggle */}
      <div className="p-4 border-b border-[#1E293B] bg-slate-950/40">
        <div className="flex items-center justify-between bg-slate-900/80 p-2.5 rounded-lg border border-[#1E293B]">
          <span className="text-[10px] font-bold text-slate-400 font-mono tracking-wide">
            {isPresentationMode ? "GUIDED TOUR" : "WORKSPACE"}
          </span>
          <button
            onClick={() => {
              const newMode = !isPresentationMode;
              setIsPresentationMode(newMode);
              // Safe fallback section selection when toggling modes
              setSection("narrative");
            }}
            className="text-blue-400 hover:text-blue-300 focus:outline-none transition-colors"
            title="Toggle View Mode"
          >
            {isPresentationMode ? (
              <ToggleRight className="w-6 h-6 text-blue-500" />
            ) : (
              <ToggleLeft className="w-6 h-6 text-slate-500" />
            )}
          </button>
        </div>
      </div>

      {/* Main Menu Links */}
      <div className="flex-1 py-4 px-4 space-y-1 overflow-y-auto">
        <p className="px-3 text-[9px] font-bold text-slate-500 tracking-wider uppercase mb-2">
          {isPresentationMode ? "Guided Research Tour" : "Research Workspace"}
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

      {/* Footer Info */}
      <div className="p-4 border-t border-[#1E293B] space-y-2 font-mono text-[9px] text-slate-500 bg-slate-950/20">
        <div className="flex justify-between items-center">
          <span>SEED</span>
          <span className="text-slate-400">42 / 123</span>
        </div>
        <div className="flex justify-between items-center">
          <span>REPRODUCIBILITY</span>
          <span className="text-emerald-500 flex items-center gap-1 font-bold">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></span>
            VERIFIED
          </span>
        </div>
      </div>

    </div>
  );
}
