import axios from "axios";

import { ElMessage, ElLoading } from "element-plus";
export const baseURL = "http://" + window.location.hostname + ":1016";

let loadingInstance;
const unLodingList = [];
const request = axios.create({
  timeout: 300000,
  baseURL,
  headers: {
    "Content-Type": "application/json",
  },
  withCredentials: true,
});

const whitelist = ["/user/login"];

// 添加请求拦截器
request.interceptors.request.use(
  function (config) {
    // 在发送请求之前做些什么
    if (!whitelist.includes(config.url) && !config.headers["Authorization"]) {
      config.headers["Authorization"] = `Bearer ${
        localStorage.getItem("token") || ""
      }`;
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
    // 对请求错误做些什么
    return Promise.reject(error);
  }
);
let errorTime;
const errorDebounceTime = 1000; // 2秒内相同的错误只显示一次

// 添加响应拦截器
request.interceptors.response.use(
  function (response) {
    loadingInstance && loadingInstance.close();
    // 2xx 范围内的状态码都会触发该函数。
    // 对响应数据做点什么
    return response.data;
  },
  function (error) {
    loadingInstance && loadingInstance.close();
    const now = Date.now();
    if (!errorTime || now - errorTime > errorDebounceTime) {
      errorTime = now;
      if (!error.response) {
        //网络请求
        ElMessage.error(t("网络错误"));
      } else {
        if (error.response.status == 401) {
          ElMessage.error("登录过期");
          router.replace({ path: "/" });
        } else if (error.response.status == 403) {
          ElMessage.error("权限不足");
          router.replace({ path: "/" });
        } else if (error.response.status == 404) {
          ElMessage.error("请求地址错误");
        } else if (error.response.status == 500) {
          ElMessage.error("服务器错误");
        } else {
          ElMessage.error(error.response.data?.msg || error.response.data);
        }
      }
    }
    return Promise.reject(error);
  }
);

export default request;
