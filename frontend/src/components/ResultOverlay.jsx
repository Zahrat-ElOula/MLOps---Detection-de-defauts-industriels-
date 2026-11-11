import React, { useEffect, useRef } from "react";
import { useTranslation } from "react-i18next";

export default function ResultOverlay({ imageUrl, result, boxes = [] }) {
  const { t } = useTranslation();
  const imgRef = useRef(null);
  const canvasRef = useRef(null);
  const boxList = boxes || result?.boxes || [];

  useEffect(() => {
    const img = imgRef.current;
    const canvas = canvasRef.current;
    if (!img || !canvas) return;

    const draw = () => {
      const ctx = canvas.getContext("2d");
      canvas.width = img.naturalWidth || img.width;
      canvas.height = img.naturalHeight || img.height;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.strokeStyle = "#dc2626";
      ctx.lineWidth = 4;
      ctx.font = "bold 16px sans-serif";
      for (let i = 0; i < boxList.length; i++) {
        const b = boxList[i];
        const x1 = b.x1;
        const y1 = b.y1;
        const x2 = b.x2;
        const y2 = b.y2;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        // Label avec fond rouge
        ctx.fillStyle = "#dc2626";
        ctx.fillRect(x1, Math.max(0, y1 - 25), 130, 25);
        ctx.fillStyle = "white";
        ctx.fillText(`${t("label_defect")} ${((b.confidence || 0) * 100).toFixed(0)}%`, x1 + 6, Math.max(12, y1 - 7));
      }
    };

    if (img.complete) draw();
    else img.onload = draw;
  }, [imageUrl, boxList, t]);

  if (!result) return null;
  const isGood = result.label === "good";
  const classLabel = (name) => {
    if (!name || name === "none") return t("defect_none");
    if (name === "good") return t("label_good");
    if (name === "defect") return t("label_defect");
    return t(`defect_${name}`, { defaultValue: name });
  };

  return (
    <div className="bg-white rounded-2xl shadow-xl border border-slate-100 overflow-hidden">
      <div className={`px-6 py-4 text-center text-lg font-semibold tracking-wide ${isGood ? "bg-emerald-50 text-emerald-700 border-b border-emerald-100" : "bg-rose-50 text-rose-700 border-b border-rose-100"}`}>
        {isGood ? t("result_good") : t("result_defect")}
      </div>
      <div className="px-6 py-2 text-sm text-center bg-slate-50 border-b border-slate-100">
        <span className="text-slate-500">{t("defect_type_label")} : </span>
        <span className={`font-semibold ${isGood ? "text-emerald-700" : "text-rose-700"}`}>
          {isGood ? t("defect_none") : classLabel(result.defect_type)}
        </span>
      </div>
      <div className="relative bg-slate-50/50">
        {imageUrl && (
          <img
            ref={imgRef}
            src={imageUrl}
            alt="result"
            className="max-h-96 object-contain w-full bg-slate-50/50"
          />
        )}
        <canvas ref={canvasRef} style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", pointerEvents: "none" }} />
      </div>
      <div className="py-4 text-center text-xl font-semibold text-slate-800 bg-white border-t border-slate-100">
        {t("confidence")} : {(result.confidence * 100).toFixed(1)}%
      </div>
    </div>
  );
}