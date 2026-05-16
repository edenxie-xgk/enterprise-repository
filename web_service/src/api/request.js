import axios from "axios";
import { ElLoading, ElMessage } from "element-plus";

import router from "../router";

const configuredBaseURL = (import.meta.env.VITE_API_BASE_URL || "").trim();

export const baseURL = configuredBaseURL || "http://" + window.location.hostname + ":1016";

const clearAuthSession = () => {
  localStorage.removeItem("token");
  localStorage.removeItem("userInfo");
};

let loadingInstance;
let loadingRequestCount = 0;
const request = axios.create({
  timeout: 300000,
  baseURL,
  headers: {
    "Content-Type": "application/json",
  },
});

const whitelist = ["/user/login"];

const openGlobalLoading = () => {
  loadingRequestCount += 1;
  if (loadingInstance) return;

  loadingInstance = ElLoading.service({
    lock: true,
    text: "加载中...",
    background: "rgba(0, 0, 0, 0.7)",
  });
};

const closeGlobalLoading = (config = {}) => {
  if (!config._showLoading) return;

  loadingRequestCount = Math.max(loadingRequestCount - 1, 0);
  if (loadingRequestCount > 0 || !loadingInstance) return;

  loadingInstance.close();
  loadingInstance = null;
};

request.interceptors.request.use(
  function (config) {
    if (!whitelist.includes(config.url) && !config.headers.Authorization) {
      config.headers.Authorization = `Bearer ${localStorage.getItem("token") || ""}`;
    }

    config._showLoading = config.showLoading !== false;
    if (config._showLoading) {
      openGlobalLoading();
    }
    return config;
  },
  function (error) {
    closeGlobalLoading(error.config);
    return Promise.reject(error);
  }
);

let errorTime;
const errorDebounceTime = 1000;

const getResponseErrorMessage = (error, fallback) =>
  error.response?.data?.msg ||
  error.response?.data?.message ||
  error.response?.data?.detail ||
  fallback;

request.interceptors.response.use(
  function (response) {
    closeGlobalLoading(response.config);
    return response.data;
  },
  function (error) {
    closeGlobalLoading(error.config);
    const now = Date.now();
    const shouldShowError = !error.config?.silentError || [401, 403].includes(error.response?.status);
    if (shouldShowError && (!errorTime || now - errorTime > errorDebounceTime)) {
      errorTime = now;
      if (!error.response) {
        ElMessage.error("网络错误");
      } else if (error.response.status === 401) {
        if (error.config?.url === "/user/login") {
          ElMessage.error(getResponseErrorMessage(error, "用户名或密码错误"));
          return Promise.reject(error);
        }
        clearAuthSession();
        ElMessage.error("登录已过期");
        router.replace({ path: "/chat" });
      } else if (error.response.status === 403) {
        ElMessage.error("权限不足");
        router.replace({ path: "/chat" });
      } else if (error.response.status === 404) {
        ElMessage.error("请求地址错误");
      } else if (error.response.status === 500) {
        ElMessage.error("服务器错误");
      } else {
        ElMessage.error(getResponseErrorMessage(error, "请求失败"));
      }
    }
    return Promise.reject(error);
  }
);

export default request;
