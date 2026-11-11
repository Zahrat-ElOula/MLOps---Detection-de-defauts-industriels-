import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';

const resources = {
  fr: {
    translation: {
      title: "Détection de défauts industriels",
      subtitle: "MVTec AD",
      upload_mode: "Téléverser",
      camera_mode: "Caméra",
      upload_drag: "Glissez-déposez une image ici ou cliquez pour choisir",
      camera_activate: "Activer la caméra",
      camera_capture: "Capturer",
      camera_stop: "Arrêter",
      analyze_btn: "Analyser",
      reset_btn: "Réinitialiser",
      analyzing: "Analyse en cours...",
      feedback_good: "C'est conforme",
      feedback_defect: "C'est défectueux",
      feedback_confirm: "Confirmer",
      feedback_report: "Signaler une erreur",
      feedback_thanks: "Merci, feedback enregistré",
      result_good: "Pièce conforme",
      result_defect: "Défaut détecté",
      confidence: "Confiance",
      loading: "Analyse en cours…",
      error_generic: "Une erreur est survenue",
      no_analysis: "Aucune analyse pour le moment",
      empty_hint: "Ajoutez une image et cliquez sur « Analyser »",
      scores_title: "Scores de confiance",
      probabilities_title: "Probabilités par classe",
      label_good: "Conforme",
      label_defect: "Défaut",
      defect_type_label: "Type de défaut",
      defect_none: "Aucun",
      defect_broken_large: "Cassure importante",
      defect_broken_small: "Fissure / petite cassure",
      defect_contamination: "Contamination",
      defect_defect: "Défaut",
      feedback_select_defect: "Choisir un type de défaut",
      feedback_send: "Envoyer"
    }
  },
  en: {
    translation: {
      title: "Industrial Defect Detection",
      subtitle: "MVTec AD",
      upload_mode: "Upload",
      camera_mode: "Camera",
      upload_drag: "Drag & drop an image here or click to choose",
      camera_activate: "Activate camera",
      camera_capture: "Capture",
      camera_stop: "Stop",
      analyze_btn: "Analyze",
      reset_btn: "Reset",
      analyzing: "Analysis in progress...",
      feedback_good: "It's conforming",
      feedback_defect: "It's defective",
      feedback_confirm: "Confirm",
      feedback_report: "Report an error",
      feedback_thanks: "Thank you, feedback saved",
      result_good: "Conforming part",
      result_defect: "Defect detected",
      confidence: "Confidence",
      loading: "Analysis in progress…",
      error_generic: "An error occurred",
      no_analysis: "No analysis yet",
      empty_hint: "Add an image and click \"Analyze\"",
      scores_title: "Confidence scores",
      probabilities_title: "Per-class probabilities",
      label_good: "Good",
      label_defect: "Defect",
      defect_type_label: "Defect type",
      defect_none: "None",
      defect_broken_large: "Large break",
      defect_broken_small: "Small crack",
      defect_contamination: "Contamination",
      defect_defect: "Defect",
      feedback_select_defect: "Select a defect type",
      feedback_send: "Send"
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
