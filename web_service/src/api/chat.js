import request, { baseURL } from "./request";

export const list_chat_sessions = (config = {}) =>
  request({
    url: "/agent/sessions",
    method: "get",
    ...config,
  });

export const get_chat_messages = (sessionId) =>
  request({
    url: `/agent/sessions/${sessionId}/messages`,
    method: "get",
  });

export const delete_chat_session = (sessionId) =>
  request({
    url: `/agent/sessions/${sessionId}`,
    method: "delete",
  });

const parseSseChunk = (block) => {
  const lines = block.split("\n");
  let event = "message";
  const dataLines = [];

  lines.forEach((line) => {
    if (line.startsWith("event:")) {
      event = line.slice(6).trim();
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trim());
    }
  });

  if (!dataLines.length) {
    return null;
  }

  return {
    event,
    data: JSON.parse(dataLines.join("\n")),
  };
};

const parseErrorMessage = (errorText, fallback) => {
  if (!errorText) return fallback;

  try {
    const payload = JSON.parse(errorText);
    if (typeof payload.detail === "string") return payload.detail;
    if (typeof payload.message === "string") return payload.message;
    if (typeof payload.msg === "string") return payload.msg;
    if (Array.isArray(payload.detail)) {
      return payload.detail.map((item) => item.msg || item.message || String(item)).join("；");
    }
  } catch {
    return errorText;
  }

  return fallback;
};

export const stream_agent_chat = async ({ query, sessionId, outputLevel, onEvent }) => {
  const response = await fetch(`${baseURL}/agent/chat/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${localStorage.getItem("token") || ""}`,
    },
    body: JSON.stringify({
      query,
      session_id: sessionId || null,
      output_level: outputLevel || "standard",
    }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    const error = new Error(parseErrorMessage(errorText, "对话请求失败"));
    error.status = response.status;
    throw error;
  }

  if (!response.body) {
    throw new Error("当前浏览器不支持流式响应");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }

    buffer += decoder.decode(value, { stream: true });
    const blocks = buffer.split("\n\n");
    buffer = blocks.pop() || "";

    blocks.forEach((block) => {
      const parsed = parseSseChunk(block.trim());
      if (parsed && typeof onEvent === "function") {
        onEvent(parsed);
      }
    });
  }

  if (buffer.trim()) {
    const parsed = parseSseChunk(buffer.trim());
    if (parsed && typeof onEvent === "function") {
      onEvent(parsed);
    }
  }
};
