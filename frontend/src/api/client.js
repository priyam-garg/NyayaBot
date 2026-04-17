import axios from "axios";

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || "http://localhost:8000",
});

api.interceptors.request.use((config) => {
  const token = localStorage.getItem("nyayabot_token");
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

api.interceptors.response.use(
  (r) => r,
  (err) => {
    if (err?.response?.status === 401) {
      localStorage.removeItem("nyayabot_token");
      localStorage.removeItem("nyayabot_user");
      if (!location.pathname.startsWith("/login") && !location.pathname.startsWith("/signup")) {
        location.href = "/login";
      }
    }
    return Promise.reject(err);
  }
);

export default api;
