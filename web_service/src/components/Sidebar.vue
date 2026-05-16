<template>
  <div class="w-80 bg-white border-r border-gray-200 flex flex-col">
    <div class="p-4 border-b border-gray-200">
      <h2 class="text-lg font-semibold text-primary">RAG 检索助手</h2>
    </div>

    <div class="flex-1 overflow-y-auto p-2 sidebar-height">
      <div class="mb-2">
        <div
          class="accordion-header flex items-center justify-between p-3 bg-secondary rounded-lg cursor-pointer hover:bg-gray-200 transition-colors"
          @click="toggleAccordion('files')"
        >
          <div class="flex items-center">
            <i class="fas fa-folder-open mr-2 text-primary"></i>
            <span class="font-medium">上传文件</span>
          </div>
          <i
            class="fas fa-chevron-down text-xs transition-transform duration-300"
            :class="{ 'rotate-180': isFilesExpanded }"
          ></i>
        </div>

        <div v-if="isFilesExpanded" class="mt-2 pl-2">
          <div v-if="isLogin" class="mb-3 px-2">
            <div class="text-xs text-gray-500 mb-1">上传到部门</div>
            <select
              v-model="selectedDeptId"
              class="w-full px-3 py-2 text-sm border border-gray-300 rounded-lg bg-white focus:outline-none focus:ring-2 focus:ring-primary/40"
            >
              <option :value="null" disabled>请选择部门</option>
              <option
                v-for="department in uploadDepartments"
                :key="department.dept_id"
                :value="department.dept_id"
              >
                {{ department.dept_name }}
              </option>
            </select>
            <div v-if="!uploadDepartments.length" class="mt-2 text-xs text-amber-600">
              当前账号没有可上传的部门权限
            </div>
          </div>

          <div id="fileList" class="space-y-2">
            <div v-if="!fileList.length" class="text-gray-500 text-sm italic pl-6">暂无上传文件</div>

            <div v-else class="space-y-2">
              <div v-for="folder in fileList" :key="folder.dept_id" class="border-l-2 border-gray-200 pl-4">
                <div
                  class="flex items-center justify-between p-2 rounded hover:bg-gray-100 cursor-pointer"
                  @click="toggleFolder(folder.dept_id)"
                >
                  <div class="flex items-center">
                    <i
                      class="fas mr-2 text-accent"
                      :class="expandedFolders.includes(folder.dept_id) ? 'fa-folder-open' : 'fa-folder'"
                    ></i>
                    <span class="text-sm font-medium">{{ folder.dept_name || `部门 ${folder.dept_id}` }}</span>
                    <span class="ml-2 text-xs text-gray-500">（{{ folder.children.length }} 个文件）</span>
                  </div>
                  <i
                    class="fas fa-chevron-down text-xs transition-transform duration-300"
                    :class="{ 'rotate-180': expandedFolders.includes(folder.dept_id) }"
                  ></i>
                </div>

                <div
                  v-if="expandedFolders.includes(folder.dept_id)"
                  class="space-y-1 mt-1 pl-4 border-l-2 border-gray-100"
                >
                  <button
                    v-for="file in folder.children"
                    :key="file.file_id || file.file_name"
                    type="button"
                    class="file-row"
                    :class="{ 'file-row-disabled': !isFileReady(file) }"
                    @click="handleFileClick(file)"
                  >
                    <i class="fas mr-2 text-accent shrink-0" :class="getFileIcon(file.file_type)"></i>
                    <div class="min-w-0 flex-1">
                      <div class="flex items-start justify-between gap-2">
                        <div class="text-sm font-medium truncate">{{ file.file_name }}</div>
                        <span class="file-state-pill" :class="getFileStateMeta(file.state).tone">
                          <i
                            class="fas mr-1"
                            :class="[
                              getFileStateMeta(file.state).icon,
                              { 'fa-spin': file.state === '2' },
                            ]"
                          ></i>
                          {{ getFileStateMeta(file.state).label }}
                        </span>
                      </div>
                      <div class="file-progress">
                        <div
                          class="file-progress-bar"
                          :class="getFileStateMeta(file.state).tone"
                          :style="{ width: `${getFileStateMeta(file.state).progress}%` }"
                        ></div>
                      </div>
                      <div class="mt-1 text-xs text-gray-500">
                        {{ getFileStateMeta(file.state).description }}
                      </div>
                    </div>
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div class="mb-2">
        <div class="flex items-center gap-2">
          <div
            class="accordion-header flex-1 flex items-center justify-between p-3 bg-secondary rounded-lg cursor-pointer hover:bg-gray-200 transition-colors"
            @click="toggleAccordion('history')"
          >
            <div class="flex items-center">
              <i class="fas fa-history mr-2 text-primary"></i>
              <span class="font-medium">聊天历史</span>
            </div>
            <i
              class="fas fa-chevron-down text-xs transition-transform duration-300"
              :class="{ 'rotate-180': isHistoryExpanded }"
            ></i>
          </div>

          <button
            class="px-3 py-3 rounded-lg bg-primary text-white hover:bg-primary/90 transition-colors"
            :class="{ 'opacity-50 cursor-not-allowed': isNewChatDisabled }"
            :disabled="isNewChatDisabled"
            title="新建会话"
            @click="$emit('new-chat')"
          >
            <i class="fas fa-plus"></i>
          </button>
        </div>

        <div v-if="isHistoryExpanded" class="mt-2 pl-2">
          <div v-if="!sessions.length" class="text-gray-500 text-sm italic pl-3">暂无历史会话</div>
          <div v-else class="space-y-2">
            <div
              v-for="history in sessions"
              :key="history.session_id"
              class="history-row"
              :class="{
                active: currentSessionId === history.session_id,
                disabled: isHistoryActionDisabled(history),
              }"
            >
              <button
                type="button"
                class="history-main"
                :disabled="isHistoryActionDisabled(history)"
                @click="$emit('select-session', history.session_id)"
              >
                <i
                  class="fas mr-2 text-accent shrink-0"
                  :class="selectingSessionId === history.session_id ? 'fa-spinner fa-spin' : 'fa-comment-dots'"
                ></i>
                <span class="min-w-0 flex-1">
                  <span class="history-title">{{ history.title || "未命名会话" }}</span>
                  <span class="history-preview">{{ getSessionPreview(history) }}</span>
                  <span class="history-meta">
                    <span>{{ formatSessionTime(history.updated_at || history.last_message_at || history.created_at) }}</span>
                    <span>{{ Number(history.message_count || 0) }} 条消息</span>
                  </span>
                </span>
              </button>
              <button
                type="button"
                class="history-delete"
                :disabled="isHistoryActionDisabled(history)"
                title="删除会话"
                @click.stop="confirmDelete(history)"
              >
                <i
                  class="fas"
                  :class="deletingSessionId === history.session_id ? 'fa-spinner fa-spin' : 'fa-trash-alt'"
                ></i>
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>

    <div class="p-4 border-t border-gray-200 bg-white">
      <div
        v-if="uploadStatus.message"
        class="upload-status"
        :class="uploadStatus.type"
      >
        {{ uploadStatus.message }}
      </div>
      <div class="grid grid-cols-2 gap-3">
        <label
          class="flex flex-col items-center justify-center p-4 border-2 border-dashed border-gray-300 rounded-lg hover:border-primary hover:bg-primary/5 transition-colors cursor-pointer"
          :class="{ 'opacity-50 cursor-not-allowed': !isLogin || !uploadDepartments.length }"
        >
          <input
            type="file"
            @change="handleFolderUpload"
            webkitdirectory
            directory
            multiple
            class="hidden"
            :disabled="!isLogin || !uploadDepartments.length"
          />
          <i class="fas fa-folder-plus text-2xl text-primary mb-2"></i>
          <span class="text-sm font-medium">上传文件夹</span>
        </label>

        <label
          class="flex flex-col items-center justify-center p-4 border-2 border-dashed border-gray-300 rounded-lg hover:border-primary hover:bg-primary/5 transition-colors cursor-pointer"
          :class="{ 'opacity-50 cursor-not-allowed': !isLogin || !uploadDepartments.length }"
        >
          <input
            type="file"
            @change="handleFileUpload"
            multiple
            class="hidden"
            :disabled="!isLogin || !uploadDepartments.length"
          />
          <i class="fas fa-file-upload text-2xl text-primary mb-2"></i>
          <span class="text-sm font-medium">上传文件</span>
        </label>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, onUnmounted, ref, watch } from "vue";
import { ElMessage, ElMessageBox } from "element-plus";

import { download_file, get_file, get_upload_departments, upload_file } from "../api/file";

const props = defineProps({
  sessions: {
    type: Array,
    default: () => [],
  },
  currentSessionId: {
    type: String,
    default: "",
  },
  isLogin: {
    type: Boolean,
    default: false,
  },
  isStreaming: {
    type: Boolean,
    default: false,
  },
  selectingSessionId: {
    type: String,
    default: "",
  },
  deletingSessionId: {
    type: String,
    default: "",
  },
});

const emit = defineEmits(["new-chat", "select-session", "delete-session"]);

const isFilesExpanded = ref(true);
const isHistoryExpanded = ref(false);
const fileList = ref([]);
const expandedFolders = ref([]);
const uploadDepartments = ref([]);
const selectedDeptId = ref(null);
const statusPollingTimer = ref(null);
const uploadStatus = ref({
  type: "",
  message: "",
});

const ACTIVE_FILE_STATES = new Set(["2", "3"]);
const FILE_STATE_META = {
  "1": {
    label: "已就绪",
    description: "可用于问答和下载",
    tone: "ready",
    icon: "fa-circle-check",
    progress: 100,
  },
  "2": {
    label: "处理中",
    description: "正在解析、切分并写入知识库",
    tone: "processing",
    icon: "fa-spinner",
    progress: 68,
  },
  "3": {
    label: "待处理",
    description: "已提交，等待后台处理",
    tone: "pending",
    icon: "fa-clock",
    progress: 32,
  },
  "4": {
    label: "失败",
    description: "处理失败，请重新上传或联系管理员",
    tone: "failed",
    icon: "fa-triangle-exclamation",
    progress: 100,
  },
};

const toggleAccordion = (type) => {
  if (type === "files") {
    isFilesExpanded.value = !isFilesExpanded.value;
  } else {
    isHistoryExpanded.value = !isHistoryExpanded.value;
  }
};

const toggleFolder = (deptId) => {
  const index = expandedFolders.value.indexOf(deptId);
  if (index > -1) {
    expandedFolders.value.splice(index, 1);
  } else {
    expandedFolders.value.push(deptId);
  }
};

const getFileIcon = (fileType) => {
  if (!fileType) return "fa-file";
  if (fileType.includes("pdf")) return "fa-file-pdf";
  if (fileType.includes("doc")) return "fa-file-word";
  if (fileType.includes("xls") || fileType.includes("csv")) return "fa-file-excel";
  if (["png", "jpg", "jpeg", "bmp", "gif"].some((item) => fileType.includes(item))) return "fa-file-image";
  return "fa-file";
};

const getFileStateMeta = (state) =>
  FILE_STATE_META[String(state || "")] || {
    label: "未知",
    description: "状态未知，请刷新后重试",
    tone: "unknown",
    icon: "fa-circle-question",
    progress: 0,
  };

const isFileReady = (file) => file?.is_ready === true || String(file?.state || "") === "1";

const getSessionPreview = (history) => {
  const preview = String(history?.preview || "").trim();
  if (preview) return preview;
  return "暂无摘要";
};

const formatSessionTime = (value) => {
  if (!value) return "时间未知";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "时间未知";

  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  if (diffMs < 0) return "刚刚";

  const diffMinutes = Math.floor(diffMs / 60000);
  if (diffMinutes >= 0 && diffMinutes < 1) return "刚刚";
  if (diffMinutes < 60) return `${diffMinutes} 分钟前`;

  const diffHours = Math.floor(diffMinutes / 60);
  if (diffHours < 24 && date.toDateString() === now.toDateString()) {
    return `${diffHours} 小时前`;
  }

  const sameYear = date.getFullYear() === now.getFullYear();
  return date.toLocaleDateString("zh-CN", {
    month: "numeric",
    day: "numeric",
    ...(sameYear ? {} : { year: "numeric" }),
  });
};

const isHistoryActionDisabled = (history) =>
  props.isStreaming ||
  Boolean(props.selectingSessionId) ||
  Boolean(props.deletingSessionId) ||
  props.selectingSessionId === history?.session_id ||
  props.deletingSessionId === history?.session_id;

const isNewChatDisabled = computed(
  () => !props.isLogin || props.isStreaming || Boolean(props.selectingSessionId) || Boolean(props.deletingSessionId)
);

const hasActiveFiles = () =>
  fileList.value.some((folder) =>
    folder.children.some((file) => ACTIVE_FILE_STATES.has(String(file.state || "")))
  );

const stopStatusPolling = () => {
  if (!statusPollingTimer.value) return;
  clearInterval(statusPollingTimer.value);
  statusPollingTimer.value = null;
};

const updateStatusPolling = () => {
  if (!props.isLogin || !hasActiveFiles()) {
    stopStatusPolling();
    return;
  }
  if (statusPollingTimer.value) return;
  statusPollingTimer.value = setInterval(() => {
    queryFileList({ silent: true });
  }, 4000);
};

const queryFileList = async ({ silent = false } = {}) => {
  if (!props.isLogin) {
    fileList.value = [];
    stopStatusPolling();
    return;
  }

  const res = await get_file({
    showLoading: !silent,
    silentError: silent,
  });
  if (res.code !== 200) return;

  const grouped = [];
  (res.data || []).forEach((file) => {
    const index = grouped.findIndex((item) => item.dept_id === file.dept_id);
    if (index >= 0) {
      grouped[index].children.push(file);
    } else {
      grouped.push({
        dept_id: file.dept_id,
        dept_name: file.dept_name,
        children: [file],
      });
    }
  });
  fileList.value = grouped;
  updateStatusPolling();
};

const queryUploadDepartments = async () => {
  if (!props.isLogin) {
    uploadDepartments.value = [];
    selectedDeptId.value = null;
    return;
  }

  const res = await get_upload_departments();
  if (res.code !== 200) return;

  uploadDepartments.value = res.data || [];
  if (!uploadDepartments.value.some((department) => department.dept_id === selectedDeptId.value)) {
    selectedDeptId.value = uploadDepartments.value[0]?.dept_id ?? null;
  }
};

const handleFilesUpload = async (files) => {
  if (!props.isLogin) {
    ElMessage.warning("请先登录");
    return;
  }
  const selectedFiles = Array.from(files || []);
  if (!selectedFiles.length) return;
  if (!selectedDeptId.value) {
    ElMessage.warning("请选择上传部门");
    return;
  }

  uploadStatus.value = {
    type: "working",
    message: `正在提交 ${selectedFiles.length} 个文件...`,
  };

  const requests = selectedFiles.map((file) => {
    const formData = new FormData();
    formData.append("file", file, file.name);
    formData.append("dept_id", selectedDeptId.value);
    return upload_file(formData);
  });

  const results = await Promise.allSettled(requests);
  const succeeded = results.filter((item) => item.status === "fulfilled").length;
  const failed = results.length - succeeded;

  if (succeeded) {
    ElMessage.success(`${succeeded} 个文件已提交处理`);
  }
  if (failed) {
    ElMessage.warning(`${failed} 个文件提交失败`);
    results
      .filter((item) => item.status === "rejected")
      .forEach((item) => console.error("上传文件失败:", item.reason));
  }

  uploadStatus.value = {
    type: failed ? "warning" : "success",
    message: failed
      ? `${succeeded} 个已提交，${failed} 个提交失败`
      : "上传任务已提交，后台正在处理文件",
  };
  await queryFileList();
};

const handleFolderUpload = (event) => {
  handleFilesUpload(event.target.files);
  event.target.value = "";
};

const handleFileUpload = (event) => {
  handleFilesUpload(event.target.files);
  event.target.value = "";
};

const handleFileClick = async (file) => {
  const stateMeta = getFileStateMeta(file?.state);
  if (!isFileReady(file)) {
    ElMessage.info(`${file?.file_name || "文件"}：${stateMeta.description}`);
    return;
  }
  if (!file?.download_url && !file?.file_path) {
    ElMessage.warning("该文件暂时没有可用的下载地址");
    return;
  }
  try {
    await download_file(file.download_url || file.file_path, file.file_name);
  } catch (error) {
    console.error("下载文件失败:", error);
  }
};

const confirmDelete = async (history) => {
  if (!history?.session_id || isHistoryActionDisabled(history)) return;
  try {
    await ElMessageBox.confirm(
      `确定删除会话“${history.title || "未命名会话"}”吗？删除后将不再显示在历史列表中。`,
      "删除会话",
      {
        type: "warning",
        confirmButtonText: "删除",
        cancelButtonText: "取消",
        confirmButtonClass: "el-button--danger",
      }
    );
    emit("delete-session", history.session_id);
  } catch {
    // 用户取消删除时不需要提示。
  }
};

watch(
  () => props.isLogin,
  async (loggedIn) => {
    if (!loggedIn) {
      fileList.value = [];
      uploadDepartments.value = [];
      selectedDeptId.value = null;
      stopStatusPolling();
      return;
    }
    await queryUploadDepartments();
    await queryFileList();
  },
  { immediate: true }
);

onUnmounted(() => {
  stopStatusPolling();
});
</script>

<style scoped>
.history-row {
  display: flex;
  align-items: stretch;
  gap: 6px;
  border: 1px solid transparent;
  border-radius: 8px;
  transition:
    background 0.16s ease,
    border-color 0.16s ease,
    opacity 0.16s ease;
}

.history-row:not(.active):hover {
  background: #f3f4f6;
}

.history-row.active {
  border-color: #bfdbfe;
  background: #eff6ff;
}

.history-row.disabled {
  opacity: 0.72;
}

.history-main {
  display: flex;
  min-width: 0;
  flex: 1;
  align-items: flex-start;
  border-radius: 8px;
  padding: 8px;
  text-align: left;
}

.history-main:disabled {
  cursor: not-allowed;
}

.history-title {
  display: block;
  overflow: hidden;
  color: #1f2937;
  font-size: 13px;
  font-weight: 600;
  line-height: 1.35;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.history-preview {
  display: block;
  overflow: hidden;
  margin-top: 3px;
  color: #64748b;
  font-size: 12px;
  line-height: 1.3;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.history-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 5px;
  color: #94a3b8;
  font-size: 11px;
  line-height: 1.3;
}

.history-delete {
  display: inline-flex;
  width: 30px;
  flex-shrink: 0;
  align-items: center;
  justify-content: center;
  border-radius: 8px;
  color: #94a3b8;
  font-size: 12px;
  transition:
    background 0.16s ease,
    color 0.16s ease;
}

.history-delete:hover:not(:disabled) {
  background: #fee2e2;
  color: #dc2626;
}

.history-delete:disabled {
  cursor: not-allowed;
}

.file-row {
  display: flex;
  width: 100%;
  border-radius: 8px;
  padding: 8px;
  text-align: left;
  transition:
    background 0.16s ease,
    opacity 0.16s ease;
}

.file-row:hover {
  background: #f3f4f6;
}

.file-row-disabled {
  cursor: default;
}

.file-state-pill {
  display: inline-flex;
  flex-shrink: 0;
  align-items: center;
  border-radius: 999px;
  padding: 2px 7px;
  font-size: 11px;
  font-weight: 600;
  line-height: 1.3;
}

.file-state-pill.ready {
  background: #dcfce7;
  color: #166534;
}

.file-state-pill.processing,
.file-state-pill.pending {
  background: #fef3c7;
  color: #92400e;
}

.file-state-pill.failed {
  background: #fee2e2;
  color: #991b1b;
}

.file-state-pill.unknown {
  background: #e5e7eb;
  color: #4b5563;
}

.file-progress {
  height: 4px;
  margin-top: 7px;
  overflow: hidden;
  border-radius: 999px;
  background: #e5e7eb;
}

.file-progress-bar {
  height: 100%;
  border-radius: inherit;
  transition: width 0.24s ease;
}

.file-progress-bar.ready {
  background: #22c55e;
}

.file-progress-bar.processing {
  background: #3b82f6;
}

.file-progress-bar.pending {
  background: #f59e0b;
}

.file-progress-bar.failed {
  background: #ef4444;
}

.upload-status {
  margin-bottom: 12px;
  border-radius: 8px;
  padding: 8px 10px;
  font-size: 12px;
  line-height: 1.4;
}

.upload-status.working {
  background: #eff6ff;
  color: #1d4ed8;
}

.upload-status.success {
  background: #ecfdf5;
  color: #047857;
}

.upload-status.warning {
  background: #fff7ed;
  color: #c2410c;
}
</style>
