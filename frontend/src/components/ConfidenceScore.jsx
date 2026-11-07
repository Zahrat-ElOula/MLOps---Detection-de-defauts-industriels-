import React from "react";
import { useTranslation } from "react-i18next";

export default function ConfidenceScore({ probabilities = { good: 0, defect: 0 } }) {
  const { t } = useTranslation();
  const good = probabilities.good || 0;
  const defect = probabilities.defect || 0;

  return (
    <div className="bg-white rounded-2xl shadow-lg border border-slate-100 p-6">
      <h3 className="text-sm font-semibold text-slate-500 uppercase tracking-wider mb-4">
        {t("scores_title")}
      </h3>
      <div className="flex flex-col gap-5">
        <div>
          <div className="flex justify-between text-sm font-medium text-slate-700 mb-1">
            <span>{t("label_good")}</span>
            <span>{(good * 100).toFixed(1)}%</span>
          </div>
          <div className="h-3 rounded-full bg-slate-100 overflow-hidden">
            <div className="h-3 rounded-full bg-gradient-to-r from-emerald-400 to-emerald-500 transition-all duration-700 ease-out" style={{ width: `${good * 100}%` }} />
          </div>
        </div>
        <div>
          <div className="flex justify-between text-sm font-medium text-slate-700 mb-1">
            <span>{t("label_defect")}</span>
            <span>{(defect * 100).toFixed(1)}%</span>
          </div>
          <div className="h-3 rounded-full bg-slate-100 overflow-hidden">
            <div className="h-3 rounded-full bg-gradient-to-r from-rose-400 to-rose-500 transition-all duration-700 ease-out" style={{ width: `${defect * 100}%` }} />
          </div>
        </div>
      </div>
    </div>
  );
}