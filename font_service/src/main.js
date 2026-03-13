import { createApp } from 'vue'
import App from './App.vue'
import './assets/css/index.css'
import '@fortawesome/fontawesome-free/css/all.min.css'
import router from './router'
const app = createApp(App)
app.use(router)
app.mount('#app')