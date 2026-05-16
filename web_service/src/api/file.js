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

export const get_upload_departments = () => {
  return request({
    url: "/file/departments",
    method: "get",
  });
};

export const get_file = (config = {}) => {
  return request({
    url: "/file/query_file",
    method: "get",
    ...config,
  });
};

export const list_admin_files = () => {
  return request({
    url: "/file/admin/files",
    method: "get",
  });
};

export const delete_admin_file = (fileId) => {
  return request({
    url: `/file/admin/files/${fileId}`,
    method: "delete",
  });
};

export const download_file = async (downloadUrl, fileName) => {
  const blob = await request({
    url: downloadUrl,
    method: "get",
    responseType: "blob",
  });

  const objectUrl = window.URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = objectUrl;
  link.download = fileName || "下载文件";
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  window.URL.revokeObjectURL(objectUrl);
};
