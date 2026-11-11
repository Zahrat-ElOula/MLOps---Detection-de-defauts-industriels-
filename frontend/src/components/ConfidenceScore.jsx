import React from "react";
import { useTranslation } from "react-i18next";

export default function ConfidenceScore({ probabilities = {} }) {
  const { t } = useTranslation();
  const entries = Object.entries(probabilities || {});
  const classLabel = (name) => {
    if (name === "good") return t("label_good");
    if (name === "defect") return t("label_defect");
    return t(`defect_${name}`, { defaultValue: name });
  };

  return (
    <div className="bg-white rounded-2xl shadow-lg border border-slate-100 p-6">
      <h3 className="text-sm font-semibold text-slate-500 uppercase tracking-wider mb-4">
        {t("probabilities_title")}
      </h3>
      <div className="flex flex-col gap-5">
        {entries.map(([name, value]) => {
          const probability = Number(value) || 0;
          const isGood = name === "good";
          return (
            <div key={name}>
              <div className="flex justify-between text-sm font-medium text-slate-700 mb-1">
                <span>{classLabel(name)}</span>
                <span>{(probability * 100).toFixed(1)}%</span>
              </div>
              <div className="h-3 rounded-full bg-slate-100 overflow-hidden">
                <div
                  className={`h-3 rounded-full transition-all duration-700 ease-out ${
                    isGood
                      ? "bg-gradient-to-r from-emerald-400 to-emerald-500"
                      : "bg-gradient-to-r from-rose-400 to-rose-500"
                  }`}
                  style={{ width: `${Math.min(100, Math.max(0, probability * 100))}%` }}
                />
              </div>
            </div>
          );
        })}
        {entries.length === 0 && (
          <p className="text-sm text-slate-400">{t("no_analysis")}</p>
        )}
      </div>
    </div>
  );
}
