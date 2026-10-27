import React from "react";

export default function ResultOverlay({ imageUrl, result }) {
  if (!result) {
    return null;
  }

  const isGood = result.label === "good";
  const bannerColor = isGood ? "#16a34a" : "#dc2626";
  const bannerText = isGood ? "Pièce conforme" : "Défaut détecté";
  const confidencePercent = (result.confidence * 100).toFixed(1);

  return (
    <div>
      <div
        style={{
          background: bannerColor,
          color: "#ffffff",
          fontWeight: "bold",
          padding: "8px",
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
            maxWidth: "100%",
            borderRadius: "0 0 6px 6px",
            display: "block",
          }}
        />
      )}
      <div style={{ marginTop: "8px" }}>
        Confiance : {confidencePercent}%
      </div>
    </div>
  );
}
