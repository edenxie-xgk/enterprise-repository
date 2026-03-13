<template>
    <div class="flex-1 flex flex-col bg-gray-50">
        <!-- 聊天头部 -->
        <div class="p-4 border-b border-gray-200 bg-white flex items-center justify-between">
            <div class="flex items-center">
                <h3 class="font-semibold text-lg">智能检索对话</h3>
                <span class="ml-2 px-2 py-1 bg-green-100 text-green-800 text-xs rounded-full">已连接</span>
            </div>
            <!-- 登录/用户信息区域（替换原有三个图标按钮） -->
            <div class="flex items-center space-x-3">
                <!-- 未登录状态 -->
                <button v-if="!isLogin"
                    class="px-4 py-1.5 bg-primary text-white rounded-lg hover:bg-primary/90 transition-colors flex items-center"
                    @click="showLoginModal = true">
                    <i class="fas fa-user-circle mr-2"></i>
                    <span>登录</span>
                </button>

                <!-- 已登录状态 -->
                <div v-else class="flex items-center space-x-2">
                    <!-- 用户名 -->
                    <span class="text-sm font-medium text-gray-700 truncate max-w-[80px]">
                        {{ userInfo.username || '未知用户' }}
                    </span>
                    <!-- 下拉菜单按钮 -->
                    <div class="relative" @click="showUserMenu = !showUserMenu">
                        <button class="text-gray-500 hover:text-primary transition-colors">
                            <i class="fas fa-caret-down"></i>
                        </button>
                        <!-- 用户下拉菜单 -->
                        <div v-if="showUserMenu"
                            class="absolute right-0 top-full mt-2 w-40 bg-white rounded-lg shadow-lg border border-gray-100 z-10">
                            <div class="py-2">
                                <a href="#" class="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100">
                                    <i class="fas fa-user mr-2"></i>个人中心
                                </a>
                                <a href="#" class="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100">
                                    <i class="fas fa-cog mr-2"></i>账号设置
                                </a>
                                <div class="border-t border-gray-100 my-1"></div>
                                <button class="w-full text-left px-4 py-2 text-sm text-red-500 hover:bg-gray-100"
                                    @click="logout">
                                    <i class="fas fa-sign-out-alt mr-2"></i>退出登录
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- 聊天内容区域 -->
        <div class="flex-1 p-6 overflow-y-auto chat-height space-y-6">
            <!-- 系统欢迎消息 -->
            <div class="flex flex-col items-start">
                <div class="flex items-center mb-1">
                    <div class="w-8 h-8 rounded-full bg-primary flex items-center justify-center text-white">
                        <i class="fas fa-robot"></i>
                    </div>
                    <span class="ml-2 text-sm font-medium">智能助手</span>
                    <span class="ml-2 text-xs text-gray-400">今天 {{ new Date().toLocaleTimeString() }}</span>
                </div>
                <div class="bg-white p-4 rounded-lg rounded-tl-none shadow-sm max-w-3xl">
                    <p class="text-gray-700">你好！我是你的RAG智能检索助手。请上传文件或文件夹，我可以帮你检索相关信息并回答问题。</p>
                </div>
            </div>

            <!-- 动态渲染聊天消息 -->
            <div v-for="(message, index) in messages" :key="index" class="flex flex-col"
                :class="{ 'items-end': message.isUser, 'items-start': !message.isUser }">
                <div class="flex items-center mb-1">
                    <span v-if="message.isUser" class="mr-2 text-xs text-gray-400">{{ message.time }}</span>
                    <span v-if="message.isUser" class="mr-2 text-sm font-medium">我</span>
                    <div v-if="message.isUser"
                        class="w-8 h-8 rounded-full bg-gray-200 flex items-center justify-center text-gray-700">
                        <i class="fas fa-user"></i>
                    </div>
                    <div v-else class="w-8 h-8 rounded-full bg-primary flex items-center justify-center text-white">
                        <i class="fas fa-robot"></i>
                    </div>
                    <!-- <span v-else class="ml-2 text-sm font-medium">智能助手</span>
                    <span v-else class="ml-2 text-xs text-gray-400">{{ message.time }}</span> -->
                </div>
                <div class="p-4 rounded-lg shadow-sm max-w-3xl" :class="{
                    'rounded-tr-none bg-primary/10': message.isUser,
                    'rounded-tl-none bg-white': !message.isUser
                }">
                    <p class="text-gray-700">{{ message.content }}</p>
                </div>
            </div>
        </div>

        <!-- 聊天输入区域 -->
        <div class="p-4 border-t border-gray-200 bg-white">
            <div class="flex flex-col space-y-3">
                <div class="relative">
                    <textarea v-model="inputMessage"
                        class="w-full p-3 pr-10 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary resize-none h-20"
                        placeholder="请输入你的问题..." @keyup.enter="sendMessage"></textarea>
                    <button class="absolute right-3 bottom-3 text-gray-400 hover:text-primary transition-colors">
                        <i class="fas fa-paperclip"></i>
                    </button>
                </div>
                <div class="flex justify-between items-center">
                    <div class="flex space-x-3">
                        <button class="text-gray-500 hover:text-primary transition-colors">
                            <i class="fas fa-microphone"></i>
                        </button>
                        <button class="text-gray-500 hover:text-primary transition-colors">
                            <i class="fas fa-image"></i>
                        </button>
                        <button class="text-gray-500 hover:text-primary transition-colors">
                            <i class="fas fa-history"></i>
                        </button>
                    </div>
                    <button
                        class="px-6 py-2 bg-primary text-white rounded-lg hover:bg-primary/90 transition-colors flex items-center"
                        @click="sendMessage">
                        <span>发送</span>
                        <i class="fas fa-paper-plane ml-2"></i>
                    </button>
                </div>
            </div>
        </div>
        <!-- 极简登录弹窗（仅用户名+密码输入框） -->
        <teleport to="body">
            <div v-if="showLoginModal" class="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
                <div class="bg-white rounded-lg w-full max-w-sm p-6">
                    <!-- 登录标题 -->
                    <h3 class="text-lg font-semibold text-center mb-4">用户登录</h3>

                    <!-- 仅保留用户名+密码输入框 -->
                    <div class="space-y-4">
                        <input type="text" v-model="loginForm.username"
                            class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary/50"
                            placeholder="请输入用户名">
                        <input type="password" v-model="loginForm.password"
                            class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary/50"
                            placeholder="请输入密码">
                    </div>

                    <!-- 登录/关闭按钮 -->
                    <div class="flex space-x-2 mt-4">
                        <button @click="showLoginModal = false"
                            class="flex-1 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition-colors">
                            取消
                        </button>
                        <button @click="handleLogin"
                            class="flex-1 py-2 bg-primary text-white rounded-lg hover:bg-primary/90 transition-colors">
                            登录
                        </button>
                    </div>
                </div>
            </div>
        </teleport>
    </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { user_login } from "../api/user";
import { get_department_role } from "../api/role";

// 响应式数据
const inputMessage = ref('')
const messages = ref([])
const showLoginModal = ref(false)
const loginForm = ref({
    username: '',
    password: ''
})
const userInfo = ref(localStorage.getItem('userInfo') ? JSON.parse(localStorage.getItem('userInfo')) : {
    username: '',
})

// 处理用户菜单点击
const showUserMenu = ref(false)
const isLogin = ref(localStorage.getItem('token') ? true : false)
// 发送消息
const sendMessage = () => {
    if (!inputMessage.value.trim()) return

    // 添加用户消息
    const userMessage = {
        content: inputMessage.value,
        isUser: true,
        time: '刚刚'
    }
    messages.value.push(userMessage)

    // 清空输入框
    inputMessage.value = ''

    // 模拟助手回复（实际项目中替换为API调用）
    setTimeout(() => {
        const assistantMessage = {
            content: '我已收到你的问题，正在基于上传的文件内容为你检索相关信息...',
            isUser: false,
            time: '刚刚'
        }
        messages.value.push(assistantMessage)

        // 滚动到底部
        scrollToBottom()
    }, 1000)

    // 滚动到底部
    scrollToBottom()
}

// 滚动到聊天底部
const scrollToBottom = () => {
    const chatContainer = document.querySelector('.chat-height')
    chatContainer.scrollTop = chatContainer.scrollHeight
}
//提交登录
const handleLogin = () => {
    user_login(loginForm.value).then(res => {
        if (res.code == 200) {
            isLogin.value = true
            showLoginModal.value = false
            showUserMenu.value = false
            userInfo.value = res.data.user
            localStorage.setItem('token', res.data.access_token)
            localStorage.setItem('userInfo', JSON.stringify(res.data.user))

        }
    })
}
const logout = () => {
    isLogin.value = false
    showUserMenu.value = false
    userInfo.value = {
        username: '',
    }
    localStorage.removeItem('token')
    localStorage.removeItem('userInfo')
}
const query_department_role = () => {
    get_department_role().then(res => {
        if (res.code == 200) {
            console.log(res.data)
        }
    })
}

onMounted(() => {
    if (localStorage.getItem('token')) {
        query_department_role()
    }
})


</script>