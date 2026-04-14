<template>
  <div class="flex-1 flex flex-col bg-gray-50">
    <div class="p-4 border-b border-gray-200 bg-white flex items-center justify-between gap-4">
      <div class="flex items-center gap-3 min-w-0">
        <div>
          <h3 class="font-semibold text-lg truncate">{{ sessionTitle || "Agent Chat" }}</h3>
          <div class="flex items-center gap-2 mt-1 text-xs text-gray-500 flex-wrap">
            <span class="px-2 py-1 bg-green-100 text-green-800 rounded-full">Connected</span>
            <span class="px-2 py-1 bg-slate-100 text-slate-700 rounded-full">
              Current output: {{ outputLevelLabel }}
            </span>
            <span v-if="isLogin" class="px-2 py-1 bg-sky-100 text-sky-700 rounded-full">
              Profile default: {{ profileAnswerStyleLabel }}
            </span>
            <span v-if="isLogin" class="px-2 py-1 bg-amber-100 text-amber-700 rounded-full">
              Web search: {{ userProfile.allow_web_search ? "On" : "Off" }}
            </span>
          </div>
        </div>
      </div>

      <div class="flex items-center space-x-3 shrink-0">
        <button
          v-if="!isLogin"
          class="px-4 py-1.5 bg-primary text-white rounded-lg hover:bg-primary/90 transition-colors flex items-center"
          @click="showLoginModal = true"
        >
          <i class="fas fa-user-circle mr-2"></i>
          <span>Login</span>
        </button>

        <div v-else class="flex items-center space-x-2">
          <span class="text-sm font-medium text-gray-700 truncate max-w-[120px]">
            {{ userInfo.username || "Unknown user" }}
          </span>
          <div class="relative">
            <button class="text-gray-500 hover:text-primary transition-colors" @click="showUserMenu = !showUserMenu">
              <i class="fas fa-caret-down"></i>
            </button>
            <div
              v-if="showUserMenu"
              class="absolute right-0 top-full mt-2 w-48 bg-white rounded-lg shadow-lg border border-gray-100 z-10"
            >
              <div class="py-2">
                <button
                  class="w-full text-left px-4 py-2 text-sm text-gray-700 hover:bg-gray-100"
                  @click="openProfilePanel"
                >
                  <i class="fas fa-sliders-h mr-2"></i>Profile settings
                </button>
                <button class="w-full text-left px-4 py-2 text-sm text-red-500 hover:bg-gray-100" @click="logout">
                  <i class="fas fa-sign-out-alt mr-2"></i>Logout
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <div ref="chatContainer" class="flex-1 p-6 overflow-y-auto chat-height space-y-6">
      <div class="flex flex-col items-start">
        <div class="flex items-center mb-1">
          <div class="w-8 h-8 rounded-full bg-primary flex items-center justify-center text-white">
            <i class="fas fa-robot"></i>
          </div>
          <span class="ml-2 text-sm font-medium">Assistant</span>
        </div>
        <div class="bg-white p-4 rounded-lg rounded-tl-none shadow-sm max-w-3xl">
          <p class="text-gray-700">
            Ask about the documents you can access. The agent will route between direct answer, RAG,
            structured data queries, and web search when allowed.
          </p>
        </div>
      </div>

      <div
        v-for="message in messages"
        :key="message.message_id"
        class="flex flex-col"
        :class="{ 'items-end': isUserMessage(message), 'items-start': !isUserMessage(message) }"
      >
        <div class="flex items-center mb-1 gap-2">
          <template v-if="isUserMessage(message)">
            <span class="text-sm font-medium">You</span>
            <div class="w-8 h-8 rounded-full bg-gray-200 flex items-center justify-center text-gray-700">
              <i class="fas fa-user"></i>
            </div>
          </template>
          <template v-else>
            <div class="w-8 h-8 rounded-full bg-primary flex items-center justify-center text-white">
              <i class="fas fa-robot"></i>
            </div>
            <span class="text-sm font-medium">Assistant</span>
          </template>
        </div>

        <div
          class="p-4 rounded-lg shadow-sm max-w-3xl"
          :class="{
            'rounded-tr-none bg-primary/10': isUserMessage(message),
            'rounded-tl-none bg-white': !isUserMessage(message),
          }"
        >
          <p class="text-gray-700 whitespace-pre-wrap">{{ message.content }}</p>

          <div
            v-if="!isUserMessage(message) && message.report_summary"
            class="mt-3 pt-3 border-t border-gray-100 space-y-3"
          >
            <div class="flex flex-wrap gap-2 text-xs">
              <span class="px-2 py-1 bg-slate-100 text-slate-700 rounded-full">
                Status: {{ message.report_summary.status || message.status || "-" }}
              </span>
              <span v-if="message.report_summary.action" class="px-2 py-1 bg-blue-100 text-blue-700 rounded-full">
                Action: {{ message.report_summary.action }}
              </span>
              <span
                v-if="message.report_summary.output_level"
                class="px-2 py-1 bg-indigo-100 text-indigo-700 rounded-full"
              >
                Output: {{ outputLevelLabelMap[message.report_summary.output_level] || message.report_summary.output_level }}
              </span>
              <span
                v-if="message.report_summary.fail_reason"
                class="px-2 py-1 bg-red-100 text-red-700 rounded-full"
              >
                Fail reason: {{ message.report_summary.fail_reason }}
              </span>
            </div>

            <div v-if="message.report_summary.reason" class="text-xs text-gray-500">
              Route note: {{ message.report_summary.reason }}
            </div>

            <div v-if="message.citations && message.citations.length">
              <div class="text-xs font-medium text-gray-500 mb-2">Citations</div>
              <div class="flex flex-wrap gap-2">
                <span
                  v-for="citation in message.citations"
                  :key="citation"
                  class="px-2 py-1 bg-gray-100 rounded text-xs text-gray-600"
                >
                  {{ citation }}
                </span>
              </div>
            </div>

            <button
              class="text-xs text-primary hover:text-primary/80 transition-colors"
              @click="toggleDetails(message.message_id)"
            >
              {{ isDetailOpen(message.message_id) ? "Hide run details" : "Show run details" }}
            </button>

            <div v-if="isDetailOpen(message.message_id)" class="detail-panel">
              <div class="detail-row">
                <span class="detail-label">Trace steps</span>
                <span>{{ message.report_summary.trace?.length || 0 }}</span>
              </div>
              <div class="detail-row">
                <span class="detail-label">Action history</span>
                <span>{{ message.report_summary.action_history?.length || 0 }}</span>
              </div>
              <div class="detail-row">
                <span class="detail-label">Long-term memory used</span>
                <span>{{ message.report_summary.long_term_memory_used ? "Yes" : "No" }}</span>
              </div>
              <div v-if="message.report_summary.user_profile?.preferred_topics?.length" class="detail-column">
                <span class="detail-label">Preferred topics</span>
                <div class="flex flex-wrap gap-2 mt-2">
                  <span
                    v-for="topic in message.report_summary.user_profile.preferred_topics"
                    :key="topic"
                    class="px-2 py-1 bg-amber-50 text-amber-700 rounded-full text-xs"
                  >
                    {{ topic }}
                  </span>
                </div>
              </div>
              <div
                v-if="message.report_summary.preferred_topics_usage?.available_topics?.length"
                class="detail-column"
              >
                <span class="detail-label">Preferred topic impact</span>
                <div class="space-y-2 mt-2">
                  <div class="detail-row">
                    <span>Used in this run</span>
                    <span>{{ message.report_summary.preferred_topics_usage.used ? "Yes" : "No" }}</span>
                  </div>
                  <div class="detail-row">
                    <span>Guidance queries</span>
                    <span>{{ message.report_summary.preferred_topics_usage.guidance_query_count || 0 }}</span>
                  </div>
                  <div
                    v-if="message.report_summary.preferred_topics_usage.applied_steps?.length"
                    class="flex flex-wrap gap-2 pt-1"
                  >
                    <span
                      v-for="step in message.report_summary.preferred_topics_usage.applied_steps"
                      :key="step"
                      class="px-2 py-1 bg-emerald-50 text-emerald-700 rounded-full text-xs"
                    >
                      {{ step }}
                    </span>
                  </div>
                </div>
              </div>
              <div v-if="message.report_summary.long_term_memory_context" class="detail-column">
                <span class="detail-label">Long-term memory context</span>
                <div class="detail-card mt-2 whitespace-pre-wrap text-sm text-gray-600">
                  {{ message.report_summary.long_term_memory_context }}
                </div>
              </div>
              <div
                v-if="message.report_summary.memory_write_summary && (message.report_summary.memory_write_summary.written_count || message.report_summary.memory_write_summary.diagnostics?.length)"
                class="detail-column"
              >
                <span class="detail-label">Memory writeback</span>
                <div class="space-y-2 mt-2">
                  <div class="detail-row">
                    <span>Written</span>
                    <span>{{ message.report_summary.memory_write_summary.written_count || 0 }}</span>
                  </div>
                  <div class="detail-row">
                    <span>Skipped</span>
                    <span>{{ message.report_summary.memory_write_summary.skipped_count || 0 }}</span>
                  </div>
                  <div
                    v-if="message.report_summary.memory_write_summary.diagnostics?.length"
                    class="detail-card text-xs text-gray-500 whitespace-pre-wrap"
                  >
                    {{ message.report_summary.memory_write_summary.diagnostics.join("\n") }}
                  </div>
                </div>
              </div>
              <div
                v-if="message.report_summary.citation_details && message.report_summary.citation_details.length"
                class="detail-column"
              >
                <span class="detail-label">Citation details</span>
                <div class="space-y-2 mt-2">
                  <div
                    v-for="detail in message.report_summary.citation_details"
                    :key="detail.node_id"
                    class="detail-card"
                  >
                    <div class="font-medium text-gray-700">{{ detail.label || detail.node_id }}</div>
                    <div class="text-xs text-gray-500 mt-1">
                      {{ detail.metadata?.file_path || detail.metadata?.file_name || detail.node_id }}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div v-if="message.status === 'streaming'" class="mt-2 text-xs text-primary flex items-center gap-2">
            <span class="inline-block w-2 h-2 rounded-full bg-primary animate-pulse"></span>
            Streaming
          </div>
        </div>
      </div>
    </div>

    <div class="p-4 border-t border-gray-200 bg-white">
      <div class="flex flex-col space-y-3">
        <div class="relative">
          <textarea
            v-model="inputMessage"
            class="w-full p-3 pr-10 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary resize-none h-20"
            :placeholder="isLogin ? 'Ask a question...' : 'Please login to start chatting'"
            :disabled="!isLogin || isStreaming"
            @keydown.enter.exact.prevent="emitSend"
          ></textarea>
        </div>

        <div class="flex justify-between items-center gap-3">
          <div class="flex items-center gap-3 flex-wrap">
            <div class="flex items-center gap-2">
              <label class="text-xs text-gray-500">Output level</label>
              <select
                class="px-2 py-1 text-sm border border-gray-300 rounded-md bg-white focus:outline-none focus:ring-2 focus:ring-primary/40"
                :value="outputLevel"
                :disabled="isStreaming"
                @change="handleOutputLevelChange"
              >
                <option v-for="option in outputLevelOptions" :key="option.value" :value="option.value">
                  {{ option.label }}
                </option>
              </select>
            </div>
            <div class="text-xs text-gray-400">
              {{ isStreaming ? "Agent is working..." : "Streaming answers, profile controls, and session replay are enabled." }}
            </div>
          </div>
          <button
            class="px-6 py-2 bg-primary text-white rounded-lg hover:bg-primary/90 transition-colors flex items-center disabled:opacity-50 disabled:cursor-not-allowed"
            :disabled="!isLogin || isStreaming || !inputMessage.trim()"
            @click="emitSend"
          >
            <span>{{ isStreaming ? "Running" : "Send" }}</span>
            <i class="fas fa-paper-plane ml-2"></i>
          </button>
        </div>
      </div>
    </div>

    <teleport to="body">
      <div v-if="showLoginModal" class="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
        <div class="bg-white rounded-lg w-full max-w-sm p-6">
          <h3 class="text-lg font-semibold text-center mb-4">User Login</h3>

          <div class="space-y-4">
            <input
              v-model="loginForm.username"
              type="text"
              class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary/50"
              placeholder="Username"
            />
            <input
              v-model="loginForm.password"
              type="password"
              class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary/50"
              placeholder="Password"
            />
          </div>

          <div class="flex space-x-2 mt-4">
            <button
              class="flex-1 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition-colors"
              @click="showLoginModal = false"
            >
              Cancel
            </button>
            <button
              class="flex-1 py-2 bg-primary text-white rounded-lg hover:bg-primary/90 transition-colors"
              @click="handleLogin"
            >
              Login
            </button>
          </div>
        </div>
      </div>
    </teleport>

    <UserProfilePanel
      v-model="showProfilePanel"
      :profile="userProfile"
      :saving="isSavingProfile"
      @save="handleProfileSave"
    />
  </div>
</template>

<script setup>
import { computed, nextTick, ref, watch } from "vue";

import { user_login } from "../api/user";
import UserProfilePanel from "./UserProfilePanel.vue";

const props = defineProps({
  messages: {
    type: Array,
    default: () => [],
  },
  isLogin: {
    type: Boolean,
    default: false,
  },
  userInfo: {
    type: Object,
    default: () => ({ username: "" }),
  },
  userProfile: {
    type: Object,
    default: () => ({
      answer_style: "standard",
      preferred_language: "zh-CN",
      preferred_topics: [],
      prefers_citations: true,
      allow_web_search: false,
      profile_notes: "",
    }),
  },
  isStreaming: {
    type: Boolean,
    default: false,
  },
  isSavingProfile: {
    type: Boolean,
    default: false,
  },
  sessionTitle: {
    type: String,
    default: "Agent Chat",
  },
  outputLevel: {
    type: String,
    default: "standard",
  },
});

const emit = defineEmits([
  "login-success",
  "logout",
  "send-message",
  "save-profile",
  "update:output-level",
]);

const inputMessage = ref("");
const showLoginModal = ref(false);
const showProfilePanel = ref(false);
const showUserMenu = ref(false);
const chatContainer = ref(null);
const detailOpenMap = ref({});
const loginForm = ref({
  username: "EdenXie",
  password: "123456",
});
const outputLevelOptions = [
  { label: "Concise", value: "concise" },
  { label: "Standard", value: "standard" },
  { label: "Detailed", value: "detailed" },
];
const outputLevelLabelMap = {
  concise: "Concise",
  standard: "Standard",
  detailed: "Detailed",
};

const outputLevelLabel = computed(
  () => outputLevelLabelMap[props.outputLevel] || props.outputLevel || "Standard"
);
const profileAnswerStyleLabel = computed(
  () => outputLevelLabelMap[props.userProfile?.answer_style] || props.userProfile?.answer_style || "Standard"
);

const scrollToBottom = async () => {
  await nextTick();
  if (chatContainer.value) {
    chatContainer.value.scrollTop = chatContainer.value.scrollHeight;
  }
};

const isUserMessage = (message) => message.role === "user";

const handleOutputLevelChange = (event) => {
  emit("update:output-level", event.target.value);
};

const emitSend = () => {
  const query = inputMessage.value.trim();
  if (!query || !props.isLogin || props.isStreaming) return;
  emit("send-message", query);
  inputMessage.value = "";
};

const handleLogin = () => {
  user_login(loginForm.value).then((res) => {
    if (res.code === 200) {
      localStorage.setItem("token", res.data.access_token);
      localStorage.setItem("userInfo", JSON.stringify(res.data.user));
      showLoginModal.value = false;
      showUserMenu.value = false;
      emit("login-success", { user: res.data.user });
    }
  });
};

const logout = () => {
  showUserMenu.value = false;
  showProfilePanel.value = false;
  localStorage.removeItem("token");
  localStorage.removeItem("userInfo");
  emit("logout");
};

const openProfilePanel = () => {
  showUserMenu.value = false;
  showProfilePanel.value = true;
};

const handleProfileSave = (payload) => {
  emit("save-profile", payload);
  showProfilePanel.value = false;
};

const toggleDetails = (messageId) => {
  detailOpenMap.value = {
    ...detailOpenMap.value,
    [messageId]: !detailOpenMap.value[messageId],
  };
};

const isDetailOpen = (messageId) => !!detailOpenMap.value[messageId];

watch(
  () => props.messages,
  () => {
    scrollToBottom();
  },
  { deep: true }
);
</script>

<style scoped>
.detail-panel {
  margin-top: 4px;
  padding: 12px;
  border-radius: 12px;
  background: #f8fafc;
  border: 1px solid #e5e7eb;
}

.detail-row {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  font-size: 12px;
  color: #475569;
}

.detail-row + .detail-row {
  margin-top: 8px;
}

.detail-column {
  margin-top: 12px;
}

.detail-label {
  font-size: 12px;
  font-weight: 600;
  color: #64748b;
}

.detail-card {
  padding: 10px 12px;
  border-radius: 10px;
  background: #ffffff;
  border: 1px solid #e5e7eb;
}
</style>
