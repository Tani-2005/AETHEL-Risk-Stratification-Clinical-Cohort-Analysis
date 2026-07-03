"use client";

import React, { useState } from "react";

interface Point {
  x: number;
  y: number;
  label?: string;
}

interface Series {
  name: string;
  points: Point[];
  color: string;
  strokeDash?: string;
}

interface InteractiveChartProps {
  title: string;
  xAxisLabel: string;
  yAxisLabel: string;
  series: Series[];
  sliderValue?: number;
  onSliderChange?: (val: number) => void;
  sliderLabel?: string;
}

export default function InteractiveChart({
  title,
  xAxisLabel,
  yAxisLabel,
  series,
  sliderValue,
  onSliderChange,
  sliderLabel
}: InteractiveChartProps) {
  const [hoveredPoint, setHoveredPoint] = useState<{
    seriesName: string;
    point: Point;
    cx: number;
    cy: number;
  } | null>(null);

  // SVG dimensions
  const width = 500;
  const height = 300;
  const paddingLeft = 50;
  const paddingRight = 20;
  const paddingTop = 20;
  const paddingBottom = 40;

  const graphWidth = width - paddingLeft - paddingRight;
  const graphHeight = height - paddingTop - paddingBottom;

  // Coordinate projection helper
  const project = (x: number, y: number) => {
    // assume x, y range is [0, 1] (typical for ROC, PR, Calibration, DCA etc.)
    const px = paddingLeft + x * graphWidth;
    const py = paddingTop + (1 - y) * graphHeight;
    return { x: px, y: py };
  };

  return (
    <div className="bg-[#111827] border border-[#1E293B] rounded-xl p-6 flex flex-col gap-4 relative">
      <div className="flex justify-between items-center">
        <h3 className="text-xs font-semibold text-slate-200 tracking-wide font-mono uppercase">
          {title}
        </h3>
        
        {/* Legends */}
        <div className="flex gap-4 flex-wrap">
          {series.map((s, idx) => (
            <div key={idx} className="flex items-center gap-2 text-[10px] text-slate-400">
              <span
                className="w-3 h-0.5"
                style={{
                  backgroundColor: s.color,
                  borderTop: s.strokeDash ? `1px dashed ${s.color}` : "none"
                }}
              />
              <span>{s.name}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="relative w-full h-[300px]">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="w-full h-full text-slate-500 overflow-visible"
        >
          {/* Grid lines */}
          {[0, 0.2, 0.4, 0.6, 0.8, 1.0].map((val, idx) => {
            const hLine = project(0, val);
            const vLine = project(val, 0);
            return (
              <React.Fragment key={idx}>
                {/* Horizontal grid line */}
                <line
                  x1={paddingLeft}
                  y1={hLine.y}
                  x2={width - paddingRight}
                  y2={hLine.y}
                  stroke="#1E293B"
                  strokeWidth="1"
                />
                <text
                  x={paddingLeft - 8}
                  y={hLine.y + 3}
                  textAnchor="end"
                  className="text-[9px] fill-slate-500 font-mono"
                >
                  {val.toFixed(1)}
                </text>

                {/* Vertical grid line */}
                <line
                  x1={vLine.x}
                  y1={paddingTop}
                  x2={vLine.x}
                  y2={height - paddingBottom}
                  stroke="#1E293B"
                  strokeWidth="1"
                />
                <text
                  x={vLine.x}
                  y={height - paddingBottom + 12}
                  textAnchor="middle"
                  className="text-[9px] fill-slate-500 font-mono"
                >
                  {val.toFixed(1)}
                </text>
              </React.Fragment>
            );
          })}

          {/* Axes Labels */}
          <text
            x={paddingLeft + graphWidth / 2}
            y={height - 8}
            textAnchor="middle"
            className="text-[10px] fill-slate-400 font-mono uppercase tracking-wider"
          >
            {xAxisLabel}
          </text>
          <text
            transform={`rotate(-90) translate(${-paddingTop - graphHeight / 2}, 12)`}
            textAnchor="middle"
            className="text-[10px] fill-slate-400 font-mono uppercase tracking-wider"
          >
            {yAxisLabel}
          </text>

          {/* Render Lines */}
          {series.map((s, idx) => {
            const pathData = s.points
              .map((p, pIdx) => {
                const projected = project(p.x, p.y);
                return `${pIdx === 0 ? "M" : "L"} ${projected.x} ${projected.y}`;
              })
              .join(" ");

            return (
              <path
                key={idx}
                d={pathData}
                fill="none"
                stroke={s.color}
                strokeWidth="1.5"
                strokeDasharray={s.strokeDash ? "4,4" : "none"}
                className="transition-all duration-300"
              />
            );
          })}

          {/* Render Interactive Dots for hover */}
          {series.map((s, sIdx) =>
            s.points.map((p, pIdx) => {
              const proj = project(p.x, p.y);
              return (
                <circle
                  key={`${sIdx}-${pIdx}`}
                  cx={proj.x}
                  cy={proj.y}
                  r="4"
                  fill="transparent"
                  className="cursor-crosshair hover:fill-blue-500/50 hover:r-6"
                  onMouseEnter={() =>
                    setHoveredPoint({
                      seriesName: s.name,
                      point: p,
                      cx: proj.x,
                      cy: proj.y
                    })
                  }
                  onMouseLeave={() => setHoveredPoint(null)}
                />
              );
            })
          )}

          {/* Render Vertical Slider Guideline (if sliderValue provided) */}
          {sliderValue !== undefined && (
            <line
              x1={project(sliderValue, 0).x}
              y1={paddingTop}
              x2={project(sliderValue, 0).x}
              y2={height - paddingBottom}
              stroke="#E11D48"
              strokeWidth="1.5"
              strokeDasharray="2,2"
            />
          )}
        </svg>

        {/* Hover Tooltip */}
        {hoveredPoint && (
          <div
            className="absolute bg-[#1E293B] border border-[#334155] text-[10px] p-2 rounded shadow-lg text-slate-200 pointer-events-none font-mono"
            style={{
              left: hoveredPoint.cx + 10,
              top: hoveredPoint.cy - 10
            }}
          >
            <div className="font-bold text-blue-400">{hoveredPoint.seriesName}</div>
            <div>
              X: {hoveredPoint.point.x.toFixed(3)}
            </div>
            <div>
              Y: {hoveredPoint.point.y.toFixed(3)}
            </div>
          </div>
        )}
      </div>

      {/* Threshold Slider Interface */}
      {sliderValue !== undefined && onSliderChange && (
        <div className="border-t border-[#1E293B] pt-4 mt-2 flex flex-col gap-2">
          <div className="flex justify-between items-center text-xs font-mono">
            <span className="text-slate-400 uppercase tracking-wider">{sliderLabel || "Decision Threshold ($p_t$)"}</span>
            <span className="text-rose-400 font-bold">{sliderValue.toFixed(2)}</span>
          </div>
          <input
            type="range"
            min="0.01"
            max="0.99"
            step="0.01"
            value={sliderValue}
            onChange={(e) => onSliderChange(parseFloat(e.target.value))}
            className="w-full accent-rose-500 bg-slate-800 h-1 rounded-lg cursor-pointer"
          />
        </div>
      )}
    </div>
  );
}
