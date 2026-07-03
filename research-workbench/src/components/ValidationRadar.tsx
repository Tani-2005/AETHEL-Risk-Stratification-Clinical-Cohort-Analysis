"use client";

import React from "react";
import { ShieldCheck, CheckCircle2, ChevronRight, HelpCircle } from "lucide-react";

interface Dimension {
  name: string;
  score: number; // range [0, 10]
  evidence: string;
  status: "verified" | "flagged" | "audited";
}

export default function ValidationRadar() {
  const dimensions: Dimension[] = [
    { name: "Predictive Performance", score: 8.5, evidence: "C-Index: 0.854 (95% CI: 0.831--0.879) on Validation Split", status: "verified" },
    { name: "Calibration", score: 9.0, evidence: "ECE: 0.018 after Platt Scaling recalibration", status: "verified" },
    { name: "Explainability", score: 8.2, evidence: "SHAP rank consensus agreement score: 98.5% on Top Features", status: "verified" },
    { name: "Robustness", score: 7.8, evidence: "AUC variance < 0.00012 under random noise perturbations", status: "audited" },
    { name: "Generalization", score: 8.0, evidence: "External validation on NHANES ECE remains < 0.024", status: "verified" },
    { name: "Reproducibility", score: 10.0, evidence: "Verified environment (Python 3.14.2 / seeds 42, 123)", status: "verified" },
    { name: "Clinical Utility", score: 8.8, evidence: "DCA Net Benefit outperforms treat-all across pt in [0.05, 0.65]", status: "verified" }
  ];

  // SVG Radar Dimensions
  const size = 200;
  const center = size / 2;
  const maxVal = 10;
  const radius = center - 20;

  // Compute spider coordinates
  const getCoordinates = () => {
    const points: string[] = [];
    dimensions.forEach((d, idx) => {
      const angle = (idx * 2 * Math.PI) / dimensions.length - Math.PI / 2;
      const x = center + radius * (d.score / maxVal) * Math.cos(angle);
      const y = center + radius * (d.score / maxVal) * Math.sin(angle);
      points.push(`${x},${y}`);
    });
    return points.join(" ");
  };

  const getGridPoints = (level: number) => {
    const points: string[] = [];
    dimensions.forEach((_, idx) => {
      const angle = (idx * 2 * Math.PI) / dimensions.length - Math.PI / 2;
      const x = center + radius * (level / maxVal) * Math.cos(angle);
      const y = center + radius * (level / maxVal) * Math.sin(angle);
      points.push(`${x},${y}`);
    });
    return points.join(" ");
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 items-stretch">
      
      {/* Radial Spider Graph Visual */}
      <div className="lg:col-span-1 bg-[#111827] border border-[#1E293B] rounded-xl p-6 flex flex-col justify-between items-center">
        <div className="w-full">
          <h3 className="text-xs font-semibold text-slate-200 tracking-wider font-mono uppercase">
            Clinical AI Validation Radar
          </h3>
          <p className="text-[10px] text-slate-500 mt-1">
            Auditing scores mapped across seven critical deployment indices.
          </p>
        </div>

        <div className="relative w-48 h-48 my-4">
          <svg viewBox={`0 0 ${size} ${size}`} className="w-full h-full text-slate-700 overflow-visible">
            {/* Grid lines */}
            {[2, 4, 6, 8, 10].map((level) => (
              <polygon
                key={level}
                points={getGridPoints(level)}
                fill="none"
                stroke="#1E293B"
                strokeWidth="1"
              />
            ))}

            {/* Radial axes */}
            {dimensions.map((_, idx) => {
              const angle = (idx * 2 * Math.PI) / dimensions.length - Math.PI / 2;
              const x = center + radius * Math.cos(angle);
              const y = center + radius * Math.sin(angle);
              return (
                <line
                  key={idx}
                  x1={center}
                  y1={center}
                  x2={x}
                  y2={y}
                  stroke="#1E293B"
                  strokeWidth="1"
                />
              );
            })}

            {/* Value polygon */}
            <polygon
              points={getCoordinates()}
              fill="rgba(37, 99, 235, 0.2)"
              stroke="#2563EB"
              strokeWidth="1.5"
            />

            {/* Corner dots */}
            {dimensions.map((d, idx) => {
              const angle = (idx * 2 * Math.PI) / dimensions.length - Math.PI / 2;
              const x = center + radius * (d.score / maxVal) * Math.cos(angle);
              const y = center + radius * (d.score / maxVal) * Math.sin(angle);
              return (
                <circle
                  key={idx}
                  cx={x}
                  cy={y}
                  r="3.5"
                  fill="#2563EB"
                  className="hover:r-5 cursor-help"
                />
              );
            })}
          </svg>
        </div>

        <div className="w-full border-t border-[#1E293B] pt-3 text-center text-[10px] text-slate-400 font-mono flex items-center justify-center gap-1.5">
          <ShieldCheck className="w-3.5 h-3.5 text-blue-500" />
          <span>AVERAGE TRUST VALUE = 8.6 / 10.0</span>
        </div>
      </div>

      {/* Details List */}
      <div className="lg:col-span-2 bg-[#111827] border border-[#1E293B] rounded-xl p-6 flex flex-col justify-between">
        <div>
          <h3 className="text-xs font-semibold text-slate-200 tracking-wider font-mono uppercase mb-4">
            Auditing Dimension Scorecard & Evidence
          </h3>
          
          <div className="space-y-3">
            {dimensions.map((d) => (
              <div key={d.name} className="flex items-center justify-between border-b border-[#1E293B]/60 pb-2.5 last:border-b-0">
                <div className="flex flex-col gap-0.5">
                  <span className="text-xs font-bold text-slate-200">{d.name}</span>
                  <span className="text-[10px] text-slate-500 font-mono">{d.evidence}</span>
                </div>
                <div className="flex items-center gap-3">
                  <span className={`text-[10px] font-mono px-2 py-0.5 rounded border ${
                    d.status === "verified" 
                      ? "bg-emerald-950/20 text-emerald-400 border-emerald-500/20" 
                      : "bg-amber-950/20 text-amber-400 border-amber-500/20"
                  }`}>
                    {d.status.toUpperCase()}
                  </span>
                  <span className="text-xs font-mono font-bold text-slate-300 w-8 text-right">
                    {d.score.toFixed(1)}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

    </div>
  );
}
