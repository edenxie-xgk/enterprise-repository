import request from "./request";

export const user_login = (data) => {
  return request({
    url: "/user/login",
    method: "post",
    data,
  });
};