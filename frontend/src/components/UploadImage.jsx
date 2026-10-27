import React, { useRef, useState } from "react";

export default function UploadImage({ onFileSelected, disabled = false }) {
  const inputRef = useRef(null);
  const [preview, setPreview] = useState(null);

  const handleClick = () => {
    if (disabled) return;
    if (inputRef.current) {
      inputRef.current.click();
    }
  };

  const handleFile = (file) => {
    if (!file) return;
    if (disabled) return;
    if (preview) {
      URL.revokeObjectURL(preview);
    }
    const url = URL.createObjectURL(file);
    setPreview(url);
    if (onFileSelected) {
      onFileSelected(file);
    }
  };

  const handleChange = (e) => {
    const file = e.target.files && e.target.files[0];
    handleFile(file);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    if (disabled) return;
    const file = e.dataTransfer.files && e.dataTransfer.files[0];
    handleFile(file);
  };

  return (
    <div>
      <div
        onClick={handleClick}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
        style={{
          padding: "24px",
          border: "2px dashed #cbd5e1",
          borderRadius: "8px",
          textAlign: "center",
          cursor: disabled ? "not-allowed" : "pointer",
          background: "#f8fafc",
        }}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          style={{ display: "none" }}
          onChange={handleChange}
          disabled={disabled}
        />
        <div>Glissez-déposez une image ici ou cliquez pour choisir</div>
      </div>
      {preview && (
        <img
          src={preview}
          alt="Aperçu"
          style={{
            maxWidth: "100%",
            maxHeight: "300px",
            objectFit: "contain",
            marginTop: "12px",
            borderRadius: "4px",
          }}
        />
      )}
    </div>
  );
}
