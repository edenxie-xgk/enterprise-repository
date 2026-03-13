import request from "./request";


export const get_department_role = () => {
    return request({
        url: "/role/department_role",
        method: "get",
    })
}