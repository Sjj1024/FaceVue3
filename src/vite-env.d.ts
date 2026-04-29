/// <reference types="vite/client" />

interface ImportMetaEnv {
    /** 队伍服务根地址，如 http://192.168.31.212:9080（不要末尾 /） */
    readonly VITE_TEAM_API_BASE?: string
}

interface ImportMeta {
    readonly env: ImportMetaEnv
}
