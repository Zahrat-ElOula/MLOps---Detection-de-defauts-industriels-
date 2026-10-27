import React from "react";

export default function ResultOverlay({ imageUrl, result }) {
  if (!result) {
    return null;
  }

  const isGood = result.label === "good";
  const bannerColor = isGood ? "#16a34a" : "#dc2626";
  const bannerText = isGood ? "Pièce conforme" : "Défaut détecté";
  const confidencePercent = (result.confidence * 100).toFixed(1);
  const borderColor = isGood ? "#16a34a" : "#dc2626";

  return (
    <div style={{ position: "relative", textAlign: "center" }}>
      <div
        style={{
          background: bannerColor,
          color: "#ffffff",
          fontWeight: "bold",
          padding: "10px",
          textAlign: "center",
          borderRadius: "6px 6px 0 0",
        }}
      >
        {bannerText}
      </div>
      {imageUrl && (
        <img
          src={imageUrl}
          alt="Résultat"
          style={{
            borderLeft: `6px solid ${borderColor}`,
            borderRight: `6px solid ${borderColor}`,
            borderBottom: `6px solid ${borderColor}`,
            display: "block",
            maxWidth: "100%",
          }}
        />
      )}
      <div style={{ marginTop: "8px", fontWeight: "bold" }}>
        Confiance : {confidencePercent}%
      </div>
    </div>
  );
}
