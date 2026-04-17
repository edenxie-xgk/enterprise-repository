import axios from "axios";
import { ElLoading, ElMessage } from "element-plus";

import router from "../router";

const configuredBaseURL = (import.meta.env.VITE_API_BASE_URL || "").trim();

export const baseURL = configuredBaseURL || "http://" + window.location.hostname + ":1016";

let loadingInstance;
const request = axios.create({
  timeout: 300000,
  baseURL,
  headers: {
    "Content-Type": "application/json",
  },
});

const whitelist = ["/user/login"];

request.interceptors.request.use(
  function (config) {
    if (!whitelist.includes(config.url) && !config.headers.Authorization) {
      config.headers.Authorization = `Bearer ${localStorage.getItem("token") || ""}`;
    }

    loadingInstance = ElLoading.service({
      lock: true,
      text: "Loading...",
      background: "rgba(0, 0, 0, 0.7)",
    });
    return config;
  },
  function (error) {
    loadingInstance && loadingInstance.close();
    return Promise.reject(error);
  }
);

let errorTime;
const errorDebounceTime = 1000;

request.interceptors.response.use(
  function (response) {
    loadingInstance && loadingInstance.close();
    return response.data;
  },
  function (error) {
    loadingInstance && loadingInstance.close();
    const now = Date.now();
    if (!errorTime || now - errorTime > errorDebounceTime) {
      errorTime = now;
      if (!error.response) {
        ElMessage.error("网络错误");
      } else if (error.response.status === 401) {
        ElMessage.error("登录已过期");
        router.replace({ path: "/" });
      } else if (error.response.status === 403) {
        ElMessage.error("权限不足");
        router.replace({ path: "/" });
      } else if (error.response.status === 404) {
        ElMessage.error("请求地址错误");
      } else if (error.response.status === 500) {
        ElMessage.error("服务器错误");
      } else {
        ElMessage.error(error.response.data?.msg || error.response.data?.detail || "请求失败");
      }
    }
    return Promise.reject(error);
  }
);

export default request;
