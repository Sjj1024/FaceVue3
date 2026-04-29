import { defineConfig, loadEnv } from 'vite'
import vue from '@vitejs/plugin-vue'
import basicSsl from '@vitejs/plugin-basic-ssl'
import path from 'path'

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
    const env = loadEnv(mode, process.cwd(), 'VITE_')
    const teamTarget = env.VITE_TEAM_API_BASE || 'http://192.168.31.212:9080'

    return {
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
                // 队伍接口：开发时走 /team-api/*，由 Vite 转发到 VITE_TEAM_API_BASE，避免浏览器 CORS
                '/team-api': {
                    target: teamTarget,
                    changeOrigin: true,
                    secure: false,
                    rewrite: (p) => p.replace(/^\/team-api/, ''),
                },
            },
        },
        resolve: {
            alias: {
                '@': path.resolve(__dirname, 'src'),
            },
        },
    }
})
