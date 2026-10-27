import React, { useState } from "react";
import UploadImage from "./components/UploadImage";
import ResultOverlay from "./components/ResultOverlay";
import ConfidenceScore from "./components/ConfidenceScore";
import { predictImage } from "./services/api";

function App() {
  const [file, setFile] = useState(null);
  const [imageUrl, setImageUrl] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const onFileSelected = (f) => {
    setFile(f);
    if (imageUrl) {
      URL.revokeObjectURL(imageUrl);
    }
    setImageUrl(URL.createObjectURL(f));
    setResult(null);
    setError(null);
  };

  const onAnalyze = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const r = await predictImage(file);
      setResult(r);
    } catch (e) {
      const detail = e.response?.data?.detail;
      setError(detail || e.message);
    } finally {
      setLoading(false);
    }
  };

  const onReset = () => {
    if (imageUrl) {
      URL.revokeObjectURL(imageUrl);
    }
    setFile(null);
    setImageUrl(null);
    setResult(null);
    setError(null);
    setLoading(false);
  };

  return (
    <div
      style={{
        maxWidth: "900px",
        margin: "auto",
        padding: "32px 16px",
        fontFamily: "system-ui",
      }}
    >
      <header style={{ marginBottom: "24px" }}>
        <h1 style={{ margin: 0 }}>Détection de défauts industriels</h1>
        <p style={{ margin: "4px 0", color: "#334155" }}>MVTec AD — catégorie bottle</p>
      </header>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "24px" }}>
        <div>
          <UploadImage onFileSelected={onFileSelected} disabled={loading} />
          <div style={{ display: "flex", gap: "12px", marginTop: "12px" }}>
            <button
              onClick={onAnalyze}
              disabled={!file || loading}
              style={{
                background: "#2563eb",
                color: "#ffffff",
                padding: "10px 20px",
                borderRadius: "6px",
                border: "none",
                cursor: !file || loading ? "not-allowed" : "pointer",
              }}
            >
              Analyser
            </button>
            <button
              onClick={onReset}
              style={{
                background: "#e2e8f0",
                color: "#000000",
                padding: "10px 20px",
                borderRadius: "6px",
                border: "none",
                cursor: "pointer",
              }}
            >
              Réinitialiser
            </button>
          </div>
          {loading && <p>Analyse en cours…</p>}
          {error && <p style={{ color: "red" }}>{error}</p>}
        </div>

        <div>
          {result ? (
            <div>
              <ResultOverlay imageUrl={imageUrl} result={result} />
              <ConfidenceScore probabilities={result.probabilities} />
            </div>
          ) : (
            <p style={{ color: "#64748b" }}>Aucune analyse pour le moment</p>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
