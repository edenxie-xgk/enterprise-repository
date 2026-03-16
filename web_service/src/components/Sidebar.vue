<template>
  <div class="w-80 bg-white border-r border-gray-200 flex flex-col">
    <!-- 侧边栏头部 -->
    <div class="p-4 border-b border-gray-200">
      <h2 class="text-lg font-semibold text-primary">RAG检索助手</h2>
    </div>

    <!-- 手风琴区域 -->
    <div class="flex-1 overflow-y-auto p-2 sidebar-height">
      <!-- 文件/文件夹手风琴 -->
      <div class="mb-2" :class="{ 'accordion-expanded': isFilesExpanded }">
        <div
          class="accordion-header flex items-center justify-between p-3 bg-secondary rounded-lg cursor-pointer hover:bg-gray-200 transition-colors"
          @click="toggleAccordion('files')">
          <div class="flex items-center">
            <i class="fas fa-folder-open mr-2 text-primary"></i>
            <span class="font-medium">上传文件</span>
          </div>
          <i class="fas fa-chevron-down text-xs transition-transform duration-300"
            :class="{ 'rotate-180': isFilesExpanded }"></i>
        </div>
        <div class="accordion-content mt-2 pl-2">
          <!-- 文件列表（重构后） -->
          <div id="fileList" class="space-y-2">
            <!-- 1. 无文件时显示提示 -->
            <div v-if="!fileList || fileList.length <= 0" class="text-gray-500 text-sm italic pl-6">
              暂无上传文件，请点击下方上传按钮
            </div>

            <!-- 2. 有文件时，按部门分组显示为文件夹 -->
            <div v-else class="space-y-2">
              <div v-for="(folder, folderIndex) in fileList" :key="folderIndex" class="border-l-2 border-gray-200 pl-4">
                <!-- 文件夹头部（可展开/收起） -->
                <div class="flex items-center justify-between p-2 rounded hover:bg-gray-100 cursor-pointer"
                  @click="toggleFolder(folder.dept_id)">
                  <div class="flex items-center">
                    <i class="fas mr-2 text-accent"
                      :class="expandedFolders.includes(folder.dept_id) ? 'fa-folder-open' : 'fa-folder'"></i>
                    <span class="text-sm font-medium">{{ folder.dept_name || `部门 ${folder.dept_id}` }}</span>
                    <span class="ml-2 text-xs text-gray-500">({{ folder.children.length }}个文件)</span>
                  </div>
                  <i class="fas fa-chevron-down text-xs transition-transform duration-300"
                    :class="{ 'rotate-180': expandedFolders.includes(folder.dept_id) }"></i>
                </div>

                <!-- 文件夹内的文件列表 -->
                <div v-if="expandedFolders.includes(folder.dept_id)"
                  class="space-y-1 mt-1 pl-4 border-l-2 border-gray-100">
                  <div v-for="(file, fileIndex) in folder.children" :key="fileIndex"
                    class="p-2 rounded hover:bg-gray-100 cursor-pointer flex items-center justify-between">
                    <div class="flex items-center truncate">
                      <i class="fas mr-2 text-accent" :class="getFileIcon(file.type)"></i>
                      <div class="truncate">
                        <div class="text-sm font-medium truncate">{{ file.file_name }}</div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- 聊天历史手风琴 -->
      <div class="mb-2" :class="{ 'accordion-expanded': isHistoryExpanded }">
        <div
          class="accordion-header flex items-center justify-between p-3 bg-secondary rounded-lg cursor-pointer hover:bg-gray-200 transition-colors"
          @click="toggleAccordion('history')">
          <div class="flex items-center">
            <i class="fas fa-history mr-2 text-primary"></i>
            <span class="font-medium">聊天历史</span>
          </div>
          <i class="fas fa-chevron-down text-xs transition-transform duration-300"
            :class="{ 'rotate-180': isHistoryExpanded }"></i>
        </div>
        <div class="accordion-content mt-2 pl-2">
          <!-- 聊天历史列表 -->
          <div class="space-y-2">
            <div v-for="(history, index) in chatHistory" :key="index"
              class="p-2 rounded hover:bg-gray-100 cursor-pointer flex items-center justify-between">
              <div class="flex items-center truncate">
                <i class="fas fa-comment-dots mr-2 text-accent"></i>
                <span class="truncate text-sm">{{ history.title }}</span>
              </div>
              <i class="fas fa-trash-alt text-gray-400 hover:text-red-500 text-xs"></i>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- 上传功能区 -->
    <div class="p-4 border-t border-gray-200 bg-white">
      <div class="space-y-3">
        <!-- 上传按钮组 -->
        <div class="grid grid-cols-2 gap-3">
          <!-- 文件夹上传按钮 -->
          <label
            class="flex flex-col items-center justify-center p-4 border-2 border-dashed border-gray-300 rounded-lg hover:border-primary hover:bg-primary/5 transition-colors cursor-pointer">
            <input type="file" @change="handleFolderUpload" webkitdirectory directory multiple class="hidden">
            <i class="fas fa-folder-plus text-2xl text-primary mb-2"></i>
            <span class="text-sm font-medium">上传文件夹</span>
          </label>

          <!-- 文件上传按钮 -->
          <label
            class="flex flex-col items-center justify-center p-4 border-2 border-dashed border-gray-300 rounded-lg hover:border-primary hover:bg-primary/5 transition-colors cursor-pointer">
            <input type="file" @change="handleFileUpload" multiple class="hidden">
            <i class="fas fa-file-upload text-2xl text-primary mb-2"></i>
            <span class="text-sm font-medium">上传文件</span>
          </label>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { upload_file, get_file } from "../api/file" // 修正拼写错误 upload_file -> upload_file

// 响应式数据
const isFilesExpanded = ref(true)
const isHistoryExpanded = ref(false)
const fileList = ref([]) // 结构：[{ dept_id, dept_name, children: [file1, file2] }]
const chatHistory = ref([
  { title: '关于产品文档的检索' },
  { title: '技术文档问答' }
])
const expandedFolders = ref([]) // 记录展开的文件夹ID

// 手风琴切换
const toggleAccordion = (type) => {
  if (type === 'files') {
    isFilesExpanded.value = !isFilesExpanded.value
  } else if (type === 'history') {
    isHistoryExpanded.value = !isHistoryExpanded.value
  }
}

// 文件夹展开/收起切换
const toggleFolder = (deptId) => {
  const index = expandedFolders.value.indexOf(deptId)
  if (index > -1) {
    expandedFolders.value.splice(index, 1)
  } else {
    expandedFolders.value.push(deptId)
  }
}

// 文件大小格式化
const formatFileSize = (bytes) => {
  if (bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
}

// 获取文件图标
const getFileIcon = (fileType) => {
  if (!fileType) return 'fa-file'
  if (fileType.includes('pdf')) return 'fa-file-pdf'
  if (fileType.includes('word')) return 'fa-file-word'
  if (fileType.includes('excel')) return 'fa-file-excel'
  if (fileType.includes('image')) return 'fa-file-image'
  return 'fa-file'
}

// 处理文件上传
const handleFilesUpload = (files, isFolder) => {
  if (files.length === 0) return

  // 假设只上传第一个文件，实际项目可循环上传
  const formData = new FormData()
  formData.append("file", files[0])
  formData.append("user_id", 1)
  formData.append("dept_id", 2)

  upload_file(formData).then(res => {
    if (res.code === 200) {
      // 上传成功后刷新文件列表
      queryFileList()
      // 可以添加成功提示
      // this.$message.success('文件上传成功')
    } else {
      // this.$message.error(res.msg || '文件上传失败')
    }
  })
}

// 文件夹上传
const handleFolderUpload = (e) => {
  handleFilesUpload(e.target.files, true)
}

// 文件上传
const handleFileUpload = (e) => {
  handleFilesUpload(e.target.files, false)
}



// 查询文件列表
const queryFileList = () => {
  get_file().then(res => {
    if (res.code === 200) {
      const fileListData = []
      res.data.forEach(v => {
        const index = fileListData.findIndex(item => item.dept_id === v.dept_id)
        if (index > -1) {
          fileListData[index].children.push(v)
        } else {
          fileListData.push({
            dept_id: v.dept_id,
            dept_name: v.dept_name,
            children: [v]
          })
        }
      })
      fileList.value = fileListData
    }
  })
}

// 初始化
onMounted(() => {
  isFilesExpanded.value = true
  queryFileList()
})
</script>