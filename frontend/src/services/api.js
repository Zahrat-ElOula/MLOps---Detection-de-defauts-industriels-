import axios from "axios";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

export async function predictImage(file) {
  const form = new FormData();
  form.append("file", file);
  const { data } = await axios.post(`${API_BASE}/predict`, form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
}

export async function sendFeedback(file, label) {
  const form = new FormData();
  form.append("file", file);
  form.append("label", label);
  const { data } = await axios.post(`${API_BASE}/feedback`, form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
}

export async function health() {
  const { data } = await axios.get(`${API_BASE}/health`);
  return data;
}
