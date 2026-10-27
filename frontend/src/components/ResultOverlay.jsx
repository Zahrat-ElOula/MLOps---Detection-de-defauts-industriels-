import React from "react";
import { useTranslation } from "react-i18next";

export default function ResultOverlay({ imageUrl, result }) {
  const { t } = useTranslation();
  if (!result) return null;
  const isGood = result.label === "good";

  return (
    <div className="bg-white rounded-2xl shadow-lg overflow-hidden">
      <div className={`px-4 py-3 text-center font-semibold ${isGood ? "bg-emerald-50 text-emerald-700" : "bg-rose-50 text-rose-700"}`}>
        {isGood ? t("result_good") : t("result_defect")}
      </div>
      {imageUrl && <img src={imageUrl} alt="result" className="w-full object-cover" />}
      <div className="px-4 py-3 text-center font-medium">
        {t("confidence")} : {(result.confidence * 100).toFixed(1)}%
      </div>
    </div>
  );
}
