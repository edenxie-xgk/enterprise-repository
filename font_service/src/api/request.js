import axios from "axios";


export const baseURL = "http://"+window.location.hostname+':1016';


const request = axios.create({
  timeout: 30000,
  baseURL,
  headers: {
    "Content-Type": "application/json",
  },
  withCredentials: true,
});

const whitelist = ["/Authentication"];

// 添加请求拦截器
request.interceptors.request.use(
  function (config) {
    // 在发送请求之前做些什么
    if (!whitelist.includes(config.url) && !config.headers["Authorization"]) {
      config.headers["Authorization"] = `Bearer ${localStorage.getItem("token")}`;
    }
    return config;
  },
  function (error) {
    // 对请求错误做些什么
    return Promise.reject(error);
  }
);

// 添加响应拦截器
request.interceptors.response.use(
  function (response) {
    // 2xx 范围内的状态码都会触发该函数。
    // 对响应数据做点什么
    return response.data;
  },
  function (error) {
    
    
    return Promise.reject(error);
  }
);

export default request;
