import React from "react";

export default function ConfidenceScore({ probabilities = { good: 0, defect: 0 } }) {
  const good = probabilities.good || 0;
  const defect = probabilities.defect || 0;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "12px", marginTop: "16px" }}>
      <div>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <span>Good</span>
          <span>{(good * 100).toFixed(1)}%</span>
        </div>
        <div style={{ height: "12px", borderRadius: "9999px", background: "#e5e7eb", margin: "4px 0" }}>
          <div
            style={{
              height: "12px",
              borderRadius: "9999px",
              background: "#16a34a",
              width: `${good * 100}%`,
            }}
          />
        </div>
      </div>

      <div>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <span>Defect</span>
          <span>{(defect * 100).toFixed(1)}%</span>
        </div>
        <div style={{ height: "12px", borderRadius: "9999px", background: "#e5e7eb", margin: "4px 0" }}>
          <div
            style={{
              height: "12px",
              borderRadius: "9999px",
              background: "#dc2626",
              width: `${defect * 100}%`,
            }}
          />
        </div>
      </div>
    </div>
  );
}
