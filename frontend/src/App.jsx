import React, { useState } from "react";
import { useTranslation } from "react-i18next";
import UploadImage from "./components/UploadImage";
import CameraCapture from "./components/CameraCapture";
import ResultOverlay from "./components/ResultOverlay";
import ConfidenceScore from "./components/ConfidenceScore";
import { predictImage, sendFeedback } from "./services/api";

function App() {
  const { t, i18n } = useTranslation();
  const [activeTab, setActiveTab] = useState("upload");
  const [file, setFile] = useState(null);
  const [imageUrl, setImageUrl] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [feedbackSent, setFeedbackSent] = useState(false);
  const [showFeedbackOptions, setShowFeedbackOptions] = useState(false);
  const [feedbackThanks, setFeedbackThanks] = useState(false);

  const clearPreview = () => {
    if (imageUrl) {
      URL.revokeObjectURL(imageUrl);
    }
  };

  const resetFeedback = () => {
    setFeedbackSent(false);
    setShowFeedbackOptions(false);
    setFeedbackThanks(false);
  };

  const onFileSelected = (f) => {
    clearPreview();
    setFile(f);
    setImageUrl(URL.createObjectURL(f));
    setResult(null);
    setError(null);
    resetFeedback();
  };

  const onCapture = (f) => {
    clearPreview();
    setFile(f);
    setImageUrl(URL.createObjectURL(f));
    setResult(null);
    setError(null);
    resetFeedback();
  };

  const onAnalyze = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    resetFeedback();
    try {
      const r = await predictImage(file);
      setResult(r);
    } catch (e) {
      const detail = e.response?.data?.detail;
      setError(detail || e.message || t("error_generic"));
    } finally {
      setLoading(false);
    }
  };

  const onFeedback = async (label) => {
    if (!file) return;
    try {
      await sendFeedback(file, label);
      setFeedbackSent(true);
      setShowFeedbackOptions(false);
      setFeedbackThanks(true);
    } catch (e) {
      const detail = e.response?.data?.detail;
      setError(detail || e.message || t("error_generic"));
    }
  };

  const onReset = () => {
    clearPreview();
    setFile(null);
    setImageUrl(null);
    setResult(null);
    setError(null);
    setLoading(false);
    resetFeedback();
  };

  const toggleLang = () => {
    i18n.changeLanguage(i18n.language === "fr" ? "en" : "fr");
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800">
      <div className="max-w-6xl mx-auto px-4 py-10">
        <header className="flex items-center justify-between mb-10">
          <div>
            <h1 className="text-3xl font-bold text-slate-900">{t("title")}</h1>
            <p className="text-slate-500 mt-1">{t("subtitle")}</p>
          </div>
          <button
            onClick={toggleLang}
            className="bg-white border border-slate-300 rounded-xl px-4 py-2 text-sm"
          >
            {i18n.language === "fr" ? "EN" : "FR"}
          </button>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <div className="flex flex-col gap-6">
            <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-1 flex">
              <button
                onClick={() => setActiveTab("upload")}
                className={`flex-1 rounded-xl py-2 ${activeTab === "upload" ? "bg-blue-600 text-white" : "text-slate-600"}`}
              >
                {t("upload_mode")}
              </button>
              <button
                onClick={() => setActiveTab("camera")}
                className={`flex-1 rounded-xl py-2 ${activeTab === "camera" ? "bg-blue-600 text-white" : "text-slate-600"}`}
              >
                {t("camera_mode")}
              </button>
            </div>

            {activeTab === "upload" && <UploadImage onFileSelected={onFileSelected} disabled={loading} />}
            {activeTab === "camera" && <CameraCapture onCapture={onCapture} />}

            {imageUrl && (
              <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-4">
                <img src={imageUrl} alt="preview" className="w-full max-h-[320px] object-contain rounded-xl" />
              </div>
            )}

            {result && !feedbackSent && (
              <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-4 flex flex-col gap-3">
                {showFeedbackOptions ? (
                  <div className="flex gap-3">
                    <button
                      onClick={() => onFeedback("good")}
                      className="bg-emerald-600 hover:bg-emerald-700 text-white rounded-xl px-4 py-2"
                    >
                      C'est Good
                    </button>
                    <button
                      onClick={() => onFeedback("defect")}
                      className="bg-rose-600 hover:bg-rose-700 text-white rounded-xl px-4 py-2"
                    >
                      C'est Defect
                    </button>
                  </div>
                ) : (
                  <div className="flex gap-3">
                    <button
                      onClick={() => setFeedbackSent(true)}
                      className="bg-emerald-600 hover:bg-emerald-700 text-white rounded-xl px-4 py-2"
                    >
                      Confirmer
                    </button>
                    <button
                      onClick={() => setShowFeedbackOptions(true)}
                      className="bg-rose-600 hover:bg-rose-700 text-white rounded-xl px-4 py-2"
                    >
                      Signaler une erreur
                    </button>
                  </div>
                )}
              </div>
            )}

            {feedbackSent && feedbackThanks && (
              <p className="text-emerald-600 font-medium">Merci, feedback enregistré</p>
            )}

            <div className="flex gap-3">
              <button
                onClick={onAnalyze}
                disabled={!file || loading}
                className="bg-blue-600 hover:bg-blue-700 disabled:opacity-60 text-white rounded-xl px-6 py-3"
              >
                {t("analyze_btn")}
              </button>
              <button onClick={onReset} className="bg-white border border-slate-300 rounded-xl px-6 py-3">
                {t("reset_btn")}
              </button>
            </div>

            {loading && <p className="text-slate-600">{t("loading")}</p>}
            {error && <p className="text-rose-600">{error}</p>}
          </div>

          <div>
            {result ? (
              <div className="flex flex-col gap-4">
                <ResultOverlay imageUrl={imageUrl} result={result} boxes={result.boxes || []} />
                <ConfidenceScore probabilities={result.probabilities} />
              </div>
            ) : (
              <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-8 text-center text-slate-500">
                {t("no_analysis")}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
