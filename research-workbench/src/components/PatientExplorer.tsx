"use client";

import React, { useState, useEffect } from "react";
import { Info, UserCheck, ShieldAlert } from "lucide-react";

export default function PatientExplorer() {
  const [age, setAge] = useState(55);
  const [bmi, setBmi] = useState(28);
  const [isSmoker, setIsSmoker] = useState(0);
  const [explanationData, setExplanationData] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchExplanation();
  }, [age, bmi, isSmoker]);

  const fetchExplanation = async () => {
    setLoading(true);
    try {
      const res = await fetch(`http://127.0.0.1:8000/api/explain/patient?age=${age}&bmi=${bmi}&is_smoker=${isSmoker}`);
      const data = await res.json();
      setExplanationData(data);
    } catch (e) {
      // Mock fallback if API offline
      const mockProb = 0.12 + (age * 0.003) + (bmi * 0.004) + (isSmoker * 0.15);
      setExplanationData({
        prediction_probability: mockProb,
        base_probability: 0.224,
        features: {
          h_age: { value: age, contribution: (age - 54.2) * 0.008 },
          h_bmi: { value: bmi, contribution: (bmi - 27.8) * 0.012 },
          h_is_smoker: { value: isSmoker, contribution: isSmoker ? 0.28 : -0.06 }
        },
        explanation: `Simulated Patient has a predicted risk probability of ${(mockProb * 100).toFixed(1)}%.`
      });
    } finally {
      setLoading(false);
    }
  };

  const waterfallBars = () => {
    if (!explanationData) return null;
    const base = explanationData.base_probability;
    const ageCont = explanationData.features.h_age.contribution;
    const bmiCont = explanationData.features.h_bmi.contribution;
    const smokerCont = explanationData.features.h_is_smoker.contribution;
    
    // Scale contributions to look nice in SVG coordinate grid
    const totalWidth = 360;
    const startX = 60;
    const scale = (val: number) => val * 120; // 1 unit = 120 pixels
    
    const basePos = startX + scale(base);
    const ageEnd = basePos + scale(ageCont);
    const bmiEnd = ageEnd + scale(bmiCont);
    const smokerEnd = bmiEnd + scale(smokerCont);

    return (
      <svg viewBox="0 0 400 240" className="w-full h-full text-slate-500 overflow-visible">
        {/* Axes lines */}
        <line x1={startX} y1="10" x2={startX} y2="210" stroke="#1E293B" strokeWidth="1" />
        <line x1={startX} y1="210" x2="380" y2="210" stroke="#1E293B" strokeWidth="1" />
        
        {/* Base Prob Grid Line */}
        <line x1={basePos} y1="10" x2={basePos} y2="210" stroke="#475569" strokeWidth="1" strokeDasharray="2,2" />
        <text x={basePos} y="225" textAnchor="middle" className="text-[9px] fill-slate-500 font-mono">
          E[Y] = {base.toFixed(2)}
        </text>

        {/* Final Prob Grid Line */}
        <line x1={smokerEnd} y1="10" x2={smokerEnd} y2="210" stroke="#E11D48" strokeWidth="1.2" />
        <text x={smokerEnd} y="238" textAnchor="middle" className="text-[9px] fill-rose-500 font-bold font-mono">
          f(x) = {explanationData.prediction_probability.toFixed(2)}
        </text>

        {/* 1. Base Probability Block */}
        <rect x={startX} y="20" width={scale(base)} height="25" fill="#334155" rx="3" />
        <text x={startX + 5} y="35" className="text-[9px] fill-slate-200 font-mono">Base Risk</text>
        
        {/* 2. Age Contribution Block */}
        <rect 
          x={ageCont > 0 ? basePos : ageEnd} 
          y="65" 
          width={Math.abs(scale(ageCont))} 
          height="25" 
          fill={ageCont > 0 ? "#10B981" : "#EF4444"} 
          rx="3" 
        />
        <text x={startX + 5} y="80" className="text-[9px] fill-slate-400 font-mono">Age ({age} yrs)</text>
        <text x={(basePos + ageEnd)/2} y="80" textAnchor="middle" className="text-[9px] fill-slate-900 font-bold font-mono">
          {ageCont > 0 ? "+" : ""}{ageCont.toFixed(3)}
        </text>

        {/* 3. BMI Contribution Block */}
        <rect 
          x={bmiCont > 0 ? ageEnd : bmiEnd} 
          y="110" 
          width={Math.abs(scale(bmiCont))} 
          height="25" 
          fill={bmiCont > 0 ? "#10B981" : "#EF4444"} 
          rx="3" 
        />
        <text x={startX + 5} y="125" className="text-[9px] fill-slate-400 font-mono">BMI ({bmi})</text>
        <text x={(ageEnd + bmiEnd)/2} y="125" textAnchor="middle" className="text-[9px] fill-slate-900 font-bold font-mono">
          {bmiCont > 0 ? "+" : ""}{bmiCont.toFixed(3)}
        </text>

        {/* 4. Smoking Contribution Block */}
        <rect 
          x={smokerCont > 0 ? bmiEnd : smokerEnd} 
          y="155" 
          width={Math.abs(scale(smokerCont))} 
          height="25" 
          fill={smokerCont > 0 ? "#10B981" : "#EF4444"} 
          rx="3" 
        />
        <text x={startX + 5} y="170" className="text-[9px] fill-slate-400 font-mono">Smoking ({isSmoker ? "Yes" : "No"})</text>
        <text x={(bmiEnd + smokerEnd)/2} y="170" textAnchor="middle" className="text-[9px] fill-slate-900 font-bold font-mono">
          {smokerCont > 0 ? "+" : ""}{smokerCont.toFixed(3)}
        </text>
      </svg>
    );
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
      
      {/* Input Sliders */}
      <div className="bg-[#111827] border border-[#1E293B] rounded-xl p-6 flex flex-col gap-6">
        <div>
          <h3 className="text-xs font-semibold text-slate-200 tracking-wider font-mono uppercase">
            Interactive Patient Profile Simulator
          </h3>
          <p className="text-[10px] text-slate-400 mt-1">
            Simulate patient risk attributes to test model decision boundaries.
          </p>
        </div>

        {/* Age */}
        <div className="flex flex-col gap-2">
          <div className="flex justify-between text-xs font-mono">
            <span className="text-slate-400">h_age (Years)</span>
            <span className="text-blue-400 font-bold">{age}</span>
          </div>
          <input 
            type="range" 
            min="20" 
            max="95" 
            value={age} 
            onChange={(e) => setAge(parseInt(e.target.value))} 
            className="w-full accent-blue-500 bg-slate-800 h-1 rounded-lg cursor-pointer"
          />
        </div>

        {/* BMI */}
        <div className="flex flex-col gap-2">
          <div className="flex justify-between text-xs font-mono">
            <span className="text-slate-400">h_bmi (Body Mass Index)</span>
            <span className="text-blue-400 font-bold">{bmi}</span>
          </div>
          <input 
            type="range" 
            min="15" 
            max="50" 
            value={bmi} 
            onChange={(e) => setBmi(parseInt(e.target.value))} 
            className="w-full accent-blue-500 bg-slate-800 h-1 rounded-lg cursor-pointer"
          />
        </div>

        {/* Smoking Toggle */}
        <div className="flex items-center justify-between border-t border-[#1E293B] pt-4">
          <div className="text-xs font-mono">
            <span className="text-slate-400 block">h_is_smoker</span>
            <span className="text-[10px] text-slate-500">Active nicotine tobacco history</span>
          </div>
          <button
            onClick={() => setIsSmoker(prev => prev === 1 ? 0 : 1)}
            className={`w-12 h-6 rounded-full p-1 transition-all duration-200 ${
              isSmoker ? "bg-blue-600 justify-end" : "bg-slate-800 justify-start"
            } flex items-center`}
          >
            <span className="w-4 h-4 rounded-full bg-white shadow-md"></span>
          </button>
        </div>

        <div className="bg-slate-900 border border-[#1E293B] p-4 rounded-lg flex items-start gap-3 text-xs leading-relaxed text-slate-400 font-mono">
          <Info className="w-4.5 h-4.5 text-blue-500 mt-0.5 flex-shrink-0" />
          <div>
            <span className="font-semibold text-slate-300 block mb-1">Decision Support Context</span>
            Waterfall diagrams show feature impact on predicted risk relative to base prevalence. 
            This visualization decomposes contributions to enhance model inspectability, not to make direct treatment recommendations.
          </div>
        </div>
      </div>

      {/* Waterfall Visualization */}
      <div className="bg-[#111827] border border-[#1E293B] rounded-xl p-6 flex flex-col gap-4">
        <h3 className="text-xs font-semibold text-slate-200 tracking-wider font-mono uppercase">
          Local SHAP Waterfall Decomposition
        </h3>
        
        <div className="border border-[#1E293B] p-4 rounded-lg bg-slate-950/70 h-[240px] flex items-center justify-center">
          {loading ? (
            <div className="text-xs text-slate-500 font-mono animate-pulse">Computing SHAP values...</div>
          ) : (
            waterfallBars()
          )}
        </div>

        {explanationData && (
          <div className="bg-slate-900/50 border border-[#1E293B] p-4 rounded-lg flex items-start gap-3 text-xs font-mono leading-relaxed text-slate-300">
            <UserCheck className="w-5 h-5 text-emerald-400 flex-shrink-0" />
            <div>
              <span className="font-bold text-slate-200 block mb-1">Audit Insights</span>
              {explanationData.explanation}
            </div>
          </div>
        )}
      </div>

    </div>
  );
}
