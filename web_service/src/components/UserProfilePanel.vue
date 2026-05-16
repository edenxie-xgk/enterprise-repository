<template>
  <el-dialog
    :model-value="modelValue"
    title="个人设置"
    width="640px"
    destroy-on-close
    :close-on-click-modal="!saving"
    :close-on-press-escape="!saving"
    :show-close="!saving"
    @close="handleClose"
  >
    <el-form label-position="top" class="profile-form">
      <el-row :gutter="16">
        <el-col :span="12">
          <el-form-item label="默认回答风格">
            <el-select v-model="form.answer_style" class="w-full" :disabled="saving">
              <el-option label="精简" value="concise" />
              <el-option label="标准" value="standard" />
              <el-option label="详细" value="detailed" />
            </el-select>
          </el-form-item>
        </el-col>
        <el-col :span="12">
          <el-form-item label="偏好语言">
            <el-select v-model="form.preferred_language" class="w-full" :disabled="saving">
              <el-option label="中文" value="zh-CN" />
              <el-option label="英文" value="en-US" />
            </el-select>
          </el-form-item>
        </el-col>
      </el-row>

      <el-form-item label="偏好主题">
        <el-select
          v-model="form.preferred_topics"
          multiple
          filterable
          allow-create
          default-first-option
          placeholder="按回车可连续添加多个主题"
          class="w-full"
          :disabled="saving"
        />
        <div class="helper-text">
          这些主题会在查询改写和检索扩展时作为弱提示使用。
        </div>
      </el-form-item>

      <el-row :gutter="16">
        <el-col :span="12">
          <el-form-item label="显示引用">
            <el-switch v-model="form.prefers_citations" :disabled="saving" />
            <div class="helper-text">
              关闭后，界面会尽量隐藏引用信息，但审计数据仍然保留。
            </div>
          </el-form-item>
        </el-col>
        <el-col :span="12">
          <el-form-item label="允许联网搜索">
            <el-switch v-model="form.allow_web_search" :disabled="saving" />
            <div class="helper-text">
              开启后，涉及实时公共信息的问题可能会走联网搜索。
            </div>
          </el-form-item>
        </el-col>
      </el-row>

      <el-form-item label="备注">
        <el-input
          v-model="form.profile_notes"
          type="textarea"
          :rows="4"
          maxlength="1000"
          show-word-limit
          placeholder="可选：填写你的角色背景、沟通偏好或长期关注点。"
          :disabled="saving"
        />
      </el-form-item>
    </el-form>

    <template #footer>
      <div v-if="errorMessage" class="save-error">
        <i class="fas fa-circle-exclamation"></i>
        <span>{{ errorMessage }}</span>
      </div>
      <div class="dialog-footer">
        <el-button :disabled="saving" @click="handleClose">取消</el-button>
        <el-button type="primary" :loading="saving" :disabled="saving" @click="handleSave">保存</el-button>
      </div>
    </template>
  </el-dialog>
</template>

<script setup>
import { reactive, watch } from "vue";

const props = defineProps({
  modelValue: {
    type: Boolean,
    default: false,
  },
  profile: {
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
  saving: {
    type: Boolean,
    default: false,
  },
  errorMessage: {
    type: String,
    default: "",
  },
});

const emit = defineEmits(["update:modelValue", "save"]);

const form = reactive({
  answer_style: "standard",
  preferred_language: "zh-CN",
  preferred_topics: [],
  prefers_citations: true,
  allow_web_search: false,
  profile_notes: "",
});

const normalizeTopics = (topics) => {
  const seen = new Set();
  return (Array.isArray(topics) ? topics : [])
    .map((topic) => String(topic || "").trim())
    .filter(Boolean)
    .filter((topic) => {
      const key = topic.toLocaleLowerCase();
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });
};

const syncForm = (profile) => {
  form.answer_style = profile?.answer_style || "standard";
  form.preferred_language = profile?.preferred_language || "zh-CN";
  form.preferred_topics = normalizeTopics(profile?.preferred_topics);
  form.prefers_citations = profile?.prefers_citations !== false;
  form.allow_web_search = !!profile?.allow_web_search;
  form.profile_notes = profile?.profile_notes || "";
};

watch(
  () => props.profile,
  (value) => {
    syncForm(value);
  },
  { immediate: true, deep: true }
);

watch(
  () => props.modelValue,
  (open) => {
    if (open) {
      syncForm(props.profile);
    }
  }
);

const handleClose = () => {
  if (props.saving) return;
  emit("update:modelValue", false);
};

const handleSave = () => {
  emit("save", {
    answer_style: form.answer_style,
    preferred_language: form.preferred_language,
    preferred_topics: normalizeTopics(form.preferred_topics),
    prefers_citations: form.prefers_citations,
    allow_web_search: form.allow_web_search,
    profile_notes: form.profile_notes.trim(),
  });
};
</script>

<style scoped>
.profile-form :deep(.el-form-item) {
  margin-bottom: 18px;
}

.helper-text {
  margin-top: 6px;
  font-size: 12px;
  line-height: 1.5;
  color: #6b7280;
}

.dialog-footer {
  display: flex;
  justify-content: flex-end;
  gap: 12px;
}

.save-error {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 12px;
  border-radius: 8px;
  background: #fef2f2;
  padding: 8px 10px;
  color: #b91c1c;
  font-size: 13px;
  line-height: 1.4;
  text-align: left;
}

.w-full {
  width: 100%;
}
</style>
