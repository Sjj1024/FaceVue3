import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import basicSsl from '@vitejs/plugin-basic-ssl'
import path from 'path'

// https://vite.dev/config/
export default defineConfig({
    plugins: [vue(), basicSsl()],
    server: {
        host: '0.0.0.0',
        proxy: {
            // 前端统一请求 /api/*，由 Vite 代理到后端，避免 CORS
            '/api': {
                target:
                    process.env.VITE_BACKEND_BASE ||
                    'http://192.168.31.198:8000',
                changeOrigin: true,
                secure: false,
                rewrite: (path) => path.replace(/^\/api/, ''),
            },
        },
    },
    resolve: {
        alias: {
            '@': path.resolve(__dirname, 'src'),
        },
    },
})
