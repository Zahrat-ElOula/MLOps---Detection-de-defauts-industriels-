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
      ctx.font = "16px sans-serif";
      ctx.fillStyle = "#dc2626";
      for (let i = 0; i < boxList.length; i++) {
        const b = boxList[i];
        const x1 = b.x1;
        const y1 = b.y1;
        const x2 = b.x2;
        const y2 = b.y2;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        ctx.fillRect(x1, Math.max(0, y1 - 25), 120, 25);
        ctx.fillStyle = "white";
        ctx.fillText(`Defect ${((b.confidence || 0) * 100).toFixed(0)}%`, x1 + 4, Math.max(12, y1 - 7));
        ctx.fillStyle = "#dc2626";
      }
    };

    if (img.complete) {
      draw();
    } else {
      img.onload = draw;
    }
  }, [imageUrl, boxList]);

  if (!result) return null;
  const isGood = result.label === "good";

  return (
    <div className="bg-white rounded-2xl shadow-lg overflow-hidden">
      <div className={`px-4 py-3 text-center font-semibold ${isGood ? "bg-emerald-50 text-emerald-700" : "bg-rose-50 text-rose-700"}`}>
        {isGood ? t("result_good") : t("result_defect")}
      </div>
      <div style={{ position: "relative" }}>
        {imageUrl && <img ref={imgRef} src={imageUrl} alt="result" className="w-full object-cover" />}
        <canvas ref={canvasRef} style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%" }} />
      </div>
      <div className="px-4 py-3 text-center font-medium">
        {t("confidence")} : {(result.confidence * 100).toFixed(1)}%
      </div>
    </div>
  );
}

