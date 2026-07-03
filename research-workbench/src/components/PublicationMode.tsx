"use client";

import React, { useState } from "react";
import { Download, FileText, Check, Copy } from "lucide-react";

export default function PublicationMode() {
  const [generating, setGenerating] = useState(false);
  const [progress, setProgress] = useState(0);
  const [generated, setGenerated] = useState(false);
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null);

  const startGeneration = () => {
    setGenerating(true);
    setGenerated(false);
    setProgress(0);
    const interval = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 100) {
          clearInterval(interval);
          setGenerating(false);
          setGenerated(true);
          return 100;
        }
        return prev + 10;
      });
    }, 150);
  };

  const copyToClipboard = (text: string, index: number) => {
    navigator.clipboard.writeText(text);
    setCopiedIndex(index);
    setTimeout(() => setCopiedIndex(null), 2000);
  };

  const latexTable = `\\begin{table}[h]
\\centering
\\caption{Model performance comparisons with 95% Bootstrap Confidence Intervals.}
\\label{tab:results}
\\begin{tabular}{lccc}
\\hline
\\textbf{Model} & \\textbf{C-Index (95% CI)} & \\textbf{ECE} & \\textbf{Brier Score} \\\\
\\hline
XGBoost (Champion)  & \\textbf{0.854 (0.831--0.879)} & 0.018 & 0.119 \\\\
Logistic Regression & 0.812 (0.785--0.838)          & 0.024 & 0.142 \\\\
Random Forest       & 0.845 (0.820--0.871)          & 0.031 & 0.128 \\\\
Cox PH (Survival)   & 0.801 (0.771--0.830)          & ---   & ---   \\\\
\\hline
\\end{tabular}
\\end{table}`;

  const captionText = "Figure 3. Decision Curve Analysis (DCA) of the XGBoost classifier compared to alternative clinical strategies across decision thresholds pt in [0.01, 0.99]. The model provides positive Net Benefit over standard strategies (Treat All/None) for thresholds up to 65%, with peak clinical utility between 10% and 40%.";

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 items-stretch">
      
      {/* Generation Panel */}
      <div className="lg:col-span-1 bg-[#111827] border border-[#1E293B] rounded-xl p-6 flex flex-col justify-between">
        <div>
          <h3 className="text-xs font-semibold text-slate-200 tracking-wider font-mono uppercase">
            Paper Asset Pipeline
          </h3>
          <p className="text-[10px] text-slate-500 mt-1 mb-6">
            Compile all model outputs, graphs, and statistical tables into journal-compliant LaTeX and high-res vector files.
          </p>

          {!generating && !generated && (
            <button
              onClick={startGeneration}
              className="w-full bg-blue-600 hover:bg-blue-500 text-white text-xs font-semibold font-mono py-2.5 px-4 rounded-lg flex items-center justify-center gap-2 transition-all duration-150"
            >
              <FileText className="w-4 h-4" />
              Generate Publication Assets
            </button>
          )}

          {generating && (
            <div className="flex flex-col gap-2 font-mono text-[10px]">
              <div className="flex justify-between text-slate-400">
                <span>Compiling metrics, SVGs & LaTeX...</span>
                <span>{progress}%</span>
              </div>
              <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                <div 
                  className="bg-blue-500 h-full transition-all duration-150" 
                  style={{ width: `${progress}%` }}
                ></div>
              </div>
            </div>
          )}

          {generated && (
            <div className="flex flex-col gap-3">
              <div className="bg-emerald-950/20 text-emerald-400 border border-emerald-500/20 p-3 rounded-lg text-[10px] font-mono leading-relaxed">
                Assets generated successfully. High-res files and supplementary tables are prepared in the repository output folders.
              </div>
              <button
                onClick={startGeneration}
                className="w-full bg-slate-900 border border-[#1E293B] text-slate-300 hover:text-slate-200 text-xs font-mono py-2 px-4 rounded-lg transition-all duration-150"
              >
                Re-generate Assets
              </button>
            </div>
          )}
        </div>

        <div className="border-t border-[#1E293B] pt-4 mt-6">
          <div className="flex flex-col gap-2.5 font-mono text-[10px] text-slate-400">
            <div className="flex justify-between items-center">
              <span>ROC_curves_val.svg</span>
              <a href="#" className="text-blue-400 hover:text-blue-300 flex items-center gap-1">
                <Download className="w-3 h-3" /> Vector SVG
              </a>
            </div>
            <div className="flex justify-between items-center">
              <span>calibration_curves.pdf</span>
              <a href="#" className="text-blue-400 hover:text-blue-300 flex items-center gap-1">
                <Download className="w-3 h-3" /> Vector PDF
              </a>
            </div>
            <div className="flex justify-between items-center">
              <span>supplementary_tables.csv</span>
              <a href="#" className="text-blue-400 hover:text-blue-300 flex items-center gap-1">
                <Download className="w-3 h-3" /> CSV Table
              </a>
            </div>
          </div>
        </div>
      </div>

      {/* LaTeX & Caption Panels */}
      <div className="lg:col-span-2 bg-[#111827] border border-[#1E293B] rounded-xl p-6 flex flex-col justify-between">
        <div className="flex flex-col gap-6">
          {/* LaTeX Block */}
          <div>
            <div className="flex justify-between items-center mb-2">
              <span className="text-xs font-semibold text-slate-200 font-mono">LaTeX Table Definition (tab:results)</span>
              <button 
                onClick={() => copyToClipboard(latexTable, 1)}
                className="text-[10px] text-blue-400 hover:text-blue-300 flex items-center gap-1 font-mono cursor-pointer"
              >
                {copiedIndex === 1 ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                {copiedIndex === 1 ? "Copied" : "Copy Code"}
              </button>
            </div>
            <pre className="bg-slate-950 p-4 rounded-lg text-[9px] font-mono text-slate-400 leading-relaxed overflow-x-auto border border-[#1E293B]">
              {latexTable}
            </pre>
          </div>

          {/* Figure Caption Block */}
          <div>
            <div className="flex justify-between items-center mb-2">
              <span className="text-xs font-semibold text-slate-200 font-mono">Figure Caption Template</span>
              <button 
                onClick={() => copyToClipboard(captionText, 2)}
                className="text-[10px] text-blue-400 hover:text-blue-300 flex items-center gap-1 font-mono cursor-pointer"
              >
                {copiedIndex === 2 ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                {copiedIndex === 2 ? "Copied" : "Copy Caption"}
              </button>
            </div>
            <div className="bg-slate-950/70 p-4 rounded-lg text-[10px] font-mono text-slate-400 leading-relaxed border border-[#1E293B]">
              {captionText}
            </div>
          </div>
        </div>
      </div>

    </div>
  );
}
