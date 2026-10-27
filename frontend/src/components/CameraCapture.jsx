import React, { useRef, useState } from "react";
import { useTranslation } from "react-i18next";

export default function CameraCapture({ onCapture }) {
  const { t } = useTranslation();
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const [active, setActive] = useState(false);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setActive(true);
    } catch (e) {
      // ignore
    }
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setActive(false);
  };

  const capture = () => {
    if (!videoRef.current || !canvasRef.current) return;
    const video = videoRef.current;
    const canvas = canvasRef.current;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob((blob) => {
      if (blob && onCapture) {
        const file = new File([blob], "capture.jpg", { type: "image/jpeg" });
        onCapture(file);
      }
    }, "image/jpeg", 0.9);
  };

  return (
    <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6 text-center">
      <video ref={videoRef} className="w-full rounded-xl bg-slate-100" playsInline muted />
      <canvas ref={canvasRef} className="hidden" />
      <div className="flex gap-3 justify-center mt-4">
        {!active ? (
          <button onClick={startCamera} className="bg-blue-600 text-white rounded-xl px-4 py-2">
            {t("camera_activate")}
          </button>
        ) : (
          <>
            <button onClick={capture} className="bg-emerald-600 text-white rounded-xl px-4 py-2">
              {t("camera_capture")}
            </button>
            <button onClick={stopCamera} className="bg-white border border-slate-300 rounded-xl px-4 py-2">
              {t("camera_stop")}
            </button>
          </>
        )}
      </div>
    </div>
  );
}
