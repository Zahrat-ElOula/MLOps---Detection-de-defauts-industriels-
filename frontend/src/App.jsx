import React, { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import UploadImage from "./components/UploadImage";
import CameraCapture from "./components/CameraCapture";
import ResultOverlay from "./components/ResultOverlay";
import ConfidenceScore from "./components/ConfidenceScore";
import { predictImage, sendFeedback } from "./services/api";

const KNOWN_DEFECT_TYPES = ["broken_large", "broken_small", "contamination"];

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
  const [selectedDefect, setSelectedDefect] = useState("");

  const clearPreview = () => {
    if (imageUrl) URL.revokeObjectURL(imageUrl);
  };

  const resetFeedback = () => {
    setFeedbackSent(false);
    setShowFeedbackOptions(false);
    setFeedbackThanks(false);
    setSelectedDefect("");
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

  useEffect(() => {
    document.title = t("title");
  }, [t]);

  const availableDefectTypes = Array.from(
    new Set([
      ...KNOWN_DEFECT_TYPES,
      ...Object.keys(result?.probabilities || {}).filter(
        (name) => name !== "good" && name !== "defect"
      ),
    ])
  );

  return (
    <div className="min-h-screen bg-slate-50/50 text-slate-800">
      <div className="max-w-7xl mx-auto bg-white/80 backdrop-blur-sm rounded-3xl shadow-xl border border-slate-100 p-8">
        <header className="flex items-center justify-between border-b border-slate-100 pb-6 mb-8">
          <div>
            <h1 className="text-3xl font-bold text-slate-800">{t("title")}</h1>
            <p className="text-slate-500 text-lg mt-1">{t("subtitle")}</p>
          </div>
          <button
            onClick={toggleLang}
            className="bg-white border border-slate-300 rounded-xl px-4 py-2 text-sm hover:bg-slate-100 transition"
          >
            {i18n.language === "fr" ? "EN" : "FR"}
          </button>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12">
          {/* Colonne gauche : Upload + boutons */}
          <div className="flex flex-col gap-6">
            <div className="bg-slate-100 p-1 rounded-xl flex">
              <button
                onClick={() => setActiveTab("upload")}
                className={`flex-1 py-2 rounded-lg text-sm font-medium transition ${activeTab === "upload" ? "bg-white text-slate-900 shadow-sm" : "text-slate-500 hover:text-slate-700"}`}
              >
                {t("upload_mode")}
              </button>
              <button
                onClick={() => setActiveTab("camera")}
                className={`flex-1 py-2 rounded-lg text-sm font-medium transition ${activeTab === "camera" ? "bg-white text-slate-900 shadow-sm" : "text-slate-500 hover:text-slate-700"}`}
              >
                {t("camera_mode")}
              </button>
            </div>

            {activeTab === "upload" && <UploadImage onFileSelected={onFileSelected} disabled={loading} />}
            {activeTab === "camera" && <CameraCapture onCapture={onCapture} />}

            {imageUrl && (
              <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6 flex justify-center">
                <img src={imageUrl} alt="preview" className="max-h-[320px] object-contain rounded-xl" />
              </div>
            )}

            {/* Feedback buttons (petits, discrets) */}
            {result && !feedbackSent && (
              <div className="flex flex-col gap-2 items-start">
                {showFeedbackOptions ? (
                  <div className="flex flex-wrap items-center gap-2">
                    <button
                      onClick={() => onFeedback("good")}
                      className="text-xs font-medium px-3 py-1.5 rounded-lg border bg-white border-emerald-200 text-emerald-700 hover:bg-emerald-50 transition"
                    >
                      {t("feedback_good")}
                    </button>
                    <select
                      value={selectedDefect}
                      onChange={(e) => setSelectedDefect(e.target.value)}
                      className="text-xs font-medium px-3 py-1.5 rounded-lg border border-rose-200 bg-white text-rose-700 focus:outline-none focus:ring-2 focus:ring-rose-200"
                    >
                      <option value="">{t("feedback_select_defect")}</option>
                      {availableDefectTypes.map((name) => (
                        <option key={name} value={name}>
                          {t(`defect_${name}`, { defaultValue: name })}
                        </option>
                      ))}
                    </select>
                    <button
                      onClick={() => selectedDefect && onFeedback(selectedDefect)}
                      disabled={!selectedDefect}
                      className="text-xs font-medium px-3 py-1.5 rounded-lg border bg-white border-rose-200 text-rose-700 hover:bg-rose-50 disabled:opacity-50 disabled:cursor-not-allowed transition"
                    >
                      {t("feedback_send")}
                    </button>
                  </div>
                ) : (
                  <div className="flex gap-2">
                    <button
                      onClick={() => setFeedbackSent(true)}
                      className="text-xs font-medium px-3 py-1.5 rounded-lg border bg-white border-emerald-200 text-emerald-700 hover:bg-emerald-50 transition"
                    >
                      {t("feedback_confirm")}
                    </button>
                    <button
                      onClick={() => setShowFeedbackOptions(true)}
                      className="text-xs font-medium px-3 py-1.5 rounded-lg border bg-white border-rose-200 text-rose-700 hover:bg-rose-50 transition"
                    >
                      {t("feedback_report")}
                    </button>
                  </div>
                )}
              </div>
            )}

            {feedbackSent && feedbackThanks && (
              <p className="text-emerald-600 font-medium">{t("feedback_thanks")}</p>
            )}

            {/* Boutons principaux */}
            <div className="flex gap-3 mt-2">
              <button
                onClick={onAnalyze}
                disabled={!file || loading}
                className="bg-indigo-600 hover:bg-indigo-700 disabled:opacity-60 text-white font-semibold px-8 py-3 rounded-xl shadow-lg shadow-indigo-200 transition-all"
              >
                {loading ? t("analyzing") : t("analyze_btn")}
              </button>
              <button
                onClick={onReset}
                className="bg-white border border-slate-200 text-slate-700 hover:bg-slate-50 px-6 py-3 rounded-xl shadow-sm"
              >
                {t("reset_btn")}
              </button>
            </div>

            {error && <p className="text-rose-600">{error}</p>}
          </div>

          {/* Colonne droite : Résultat */}
          <div>
            {result ? (
              <div className="flex flex-col gap-4">
                <ResultOverlay imageUrl={imageUrl} result={result} boxes={result.boxes || []} />
                <ConfidenceScore probabilities={result.probabilities} />
              </div>
            ) : (
              <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-12 text-center text-slate-400">
                <p className="text-lg font-medium">{t("no_analysis")}</p>
                <p className="text-sm mt-2">{t("empty_hint")}</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;