import request from "./request";

export const upload_file = (data) => {
  return request({
    url: "/file/upload",
    method: "post",
    data,
    headers: {
      "Content-Type": "multipart/form-data",
    },
  });
};


export const get_file = () => {
  return request({
    url: "/file/query_file",
    method: "get",
  });
};