<template>
    <div class="font-inter bg-gray-50 text-dark antialiased">
      <div class="flex h-screen overflow-hidden">
        <!-- 左侧边栏组件 -->
        <Sidebar />
        
        <!-- 右侧聊天窗口组件 -->
        <ChatWindow />
      </div>
    </div>
  </template>
  
  <script setup>
  import Sidebar from '../../components/Sidebar.vue'
  import ChatWindow from '../../components/ChatWindow.vue'
  </script>
  
 