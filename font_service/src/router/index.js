import { createRouter, createWebHashHistory } from "vue-router";
import Chat from '../view/chat/Chat.vue'

const routes = [
  {
    path: '/',
    redirect: '/chat'
  },
  {
    path: '/chat',
    component: Chat
  }
];

const router = createRouter({
  history: createWebHashHistory(),
  routes,
});

export default router;
