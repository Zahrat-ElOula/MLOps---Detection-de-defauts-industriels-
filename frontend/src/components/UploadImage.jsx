import React, { useRef } from "react";
import { useTranslation } from "react-i18next";

export default function UploadImage({ onFileSelected, disabled = false }) {
  const { t } = useTranslation();
  const inputRef = useRef(null);

  const handleClick = () => {
    if (disabled) return;
    inputRef.current?.click();
  };

  const handleChange = (e) => {
    const file = e.target.files?.[0];
    if (file && !disabled && onFileSelected) {
      onFileSelected(file);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    if (disabled) return;
    const file = e.dataTransfer.files?.[0];
    if (file && onFileSelected) {
      onFileSelected(file);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  return (
    <div
      onClick={handleClick}
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      className={`bg-white rounded-2xl shadow-sm border-2 border-dashed border-slate-300 p-10 text-center transition-colors ${
        disabled ? "opacity-60 cursor-not-allowed" : "cursor-pointer hover:border-blue-400"
      }`}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={handleChange}
        disabled={disabled}
      />
      <p className="text-slate-600">{t("upload_drag")}</p>
    </div>
  );
}
