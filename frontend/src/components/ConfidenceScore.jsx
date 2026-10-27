import React from "react";

export default function ConfidenceScore({ probabilities = { good: 0, defect: 0 } }) {
  const good = probabilities.good || 0;
  const defect = probabilities.defect || 0;

  return (
    <div className="flex flex-col gap-4">
      <div>
        <div className="flex justify-between text-sm text-slate-700">
          <span>Good</span>
          <span>{(good * 100).toFixed(1)}%</span>
        </div>
        <div className="mt-2 bg-slate-100 rounded-full h-2 overflow-hidden">
          <div className="bg-emerald-500 h-2 rounded-full" style={{ width: `${good * 100}%` }} />
        </div>
      </div>
      <div>
        <div className="flex justify-between text-sm text-slate-700">
          <span>Defect</span>
          <span>{(defect * 100).toFixed(1)}%</span>
        </div>
        <div className="mt-2 bg-slate-100 rounded-full h-2 overflow-hidden">
          <div className="bg-rose-500 h-2 rounded-full" style={{ width: `${defect * 100}%` }} />
        </div>
      </div>
    </div>
  );
}
