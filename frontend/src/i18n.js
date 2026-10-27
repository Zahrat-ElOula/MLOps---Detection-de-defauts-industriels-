import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';

const resources = {
  fr: {
    translation: {
      title: "Détection de défauts industriels",
      subtitle: "MVTec AD — catégorie bottle",
      upload_mode: "Téléverser",
      camera_mode: "Caméra",
      upload_drag: "Glissez-déposez une image ici ou cliquez pour choisir",
      camera_activate: "Activer la caméra",
      camera_capture: "Capturer",
      camera_stop: "Arrêter",
      analyze_btn: "Analyser",
      reset_btn: "Réinitialiser",
      result_good: "Pièce conforme",
      result_defect: "Défaut détecté",
      confidence: "Confiance",
      loading: "Analyse en cours…",
      error_generic: "Une erreur est survenue",
      no_analysis: "Aucune analyse pour le moment"
    }
  },
  en: {
    translation: {
      title: "Industrial Defect Detection",
      subtitle: "MVTec AD — bottle category",
      upload_mode: "Upload",
      camera_mode: "Camera",
      upload_drag: "Drag & drop an image here or click to choose",
      camera_activate: "Activate camera",
      camera_capture: "Capture",
      camera_stop: "Stop",
      analyze_btn: "Analyze",
      reset_btn: "Reset",
      result_good: "Conforming part",
      result_defect: "Defect detected",
      confidence: "Confidence",
      loading: "Analysis in progress…",
      error_generic: "An error occurred",
      no_analysis: "No analysis yet"
    }
  }
};

i18n
  .use(initReactI18next)
  .init({
    resources,
    lng: "fr",
    fallbackLng: "fr",
    interpolation: { escapeValue: false }
  });

export default i18n;
