<script setup lang="ts">
import { ref } from 'vue'
import CameraCapture from '../components/CameraCapture.vue'

// 开发环境走 Vite 代理，避免 CORS：/api -> VITE_BACKEND_BASE
const API_BASE = '/api'

const name = ref('张三')
// 可选
const personId = ref('')
// 最近一次错误信息
const lastError = ref('')
// 最近一次录入结果
const lastResult = ref<any>(null)
// 最近一次拍照的 Blob
const lastPhotoBlob = ref<Blob | null>(null)
// 最近一次拍照的文件名
const lastPhotoFilename = ref('capture.jpg')

// 最近一次拍照的预览 URL
const uploadPreviewUrl = ref('')
// 最近一次“拍照”得到的预览（dataUrl）。用于拍完后隐藏实时画面，仅展示静态图。
const capturedPreviewUrl = ref('')
// 成功提示（短暂显示）
const successMsg = ref('')
let successTimer: number | null = null

function showSuccess(msg: string) {
    successMsg.value = msg
    if (successTimer != null) window.clearTimeout(successTimer)
    successTimer = window.setTimeout(() => {
        successMsg.value = ''
        successTimer = null
    }, 2600)
    alert(msg)
    // 刷新页面
    window.location.reload()
}

// 选择文件
function onFilePicked(e: Event) {
    lastError.value = ''
    lastResult.value = null
    successMsg.value = ''

    const input = e.target as HTMLInputElement
    const file = input.files?.[0]
    if (!file) return

    lastPhotoBlob.value = file
    lastPhotoFilename.value = file.name || 'upload.jpg'

    const url = URL.createObjectURL(file)
    uploadPreviewUrl.value = url
}

function clearUpload() {
    uploadPreviewUrl.value = ''
    capturedPreviewUrl.value = ''
    lastPhotoBlob.value = null
    lastPhotoFilename.value = 'capture.jpg'
    lastError.value = ''
    lastResult.value = null
    successMsg.value = ''
}

async function enroll() {
    lastError.value = ''
    lastResult.value = null
    successMsg.value = ''

    if (!name.value.trim()) {
        lastError.value = '请先填写名字（例如：张三）。'
        return
    }
    if (!lastPhotoBlob.value) {
        lastError.value = '请先拍照或上传照片。'
        return
    }

    const form = new FormData()
    form.append('name', name.value.trim())
    if (personId.value.trim()) form.append('person_id', personId.value.trim())
    form.append('image', lastPhotoBlob.value, lastPhotoFilename.value)

    try {
        const resp = await fetch(`${API_BASE}/enroll`, {
            method: 'POST',
            body: form,
        })
        const data = await resp.json().catch(() => ({}))
        if (!resp.ok) {
            lastError.value = data?.detail
                ? String(data.detail)
                : `请求失败：HTTP ${resp.status}`
            return
        }
        lastResult.value = data
        showSuccess(`录入成功：${name.value.trim()}`)
    } catch (e) {
        lastError.value = e instanceof Error ? e.message : String(e)
    }
}
</script>

<template>
    <section class="page">
        <header class="header">
            <div>
                <h1>录入（拍照 + 标记是谁）</h1>
                <p class="sub">
                    拍一张照片，填写名字，调用后端计算 embedding 并保存。
                </p>
            </div>
        </header>

        <div class="panel">
            <div class="form">
                <label class="field">
                    <span>名字（必填）</span>
                    <input v-model="name" placeholder="例如：张三" />
                </label>
                <!-- <label class="field">
                    <span>person_id（可选）</span>
                    <input
                        v-model="personId"
                        placeholder="不填则后端自动生成 UUID"
                    />
                </label> -->
                <label class="field">
                    <span>上传照片（可选）</span>
                    <input
                        type="file"
                        accept="image/*"
                        @change="onFilePicked"
                    />
                </label>
                <button class="btn primary" @click="enroll">提交录入</button>
                <button
                    class="btn"
                    :disabled="!lastPhotoBlob"
                    @click="clearUpload"
                >
                    清除已选照片
                </button>
            </div>

            <div v-if="uploadPreviewUrl" class="uploadPreview">
                <img :src="uploadPreviewUrl" alt="upload preview" />
            </div>

            <div v-if="capturedPreviewUrl" class="uploadPreview">
                <img :src="capturedPreviewUrl" alt="captured preview" />
            </div>

            <CameraCapture
                v-if="!capturedPreviewUrl"
                :autoStart="false"
                @captured="
                    ({ blob, dataUrl }) => (
                        (successMsg = ''),
                        (lastPhotoBlob = blob),
                        (lastPhotoFilename = 'capture.jpg'),
                        (uploadPreviewUrl = ''),
                        (capturedPreviewUrl = dataUrl)
                    )
                "
                @error="(m) => (lastError = m)"
            />

            <p v-if="successMsg" class="success">{{ successMsg }}</p>
            <p v-if="lastError" class="error">{{ lastError }}</p>
            <pre v-else-if="lastResult" class="pre">
                {{ JSON.stringify(lastResult, null, 2) }}
            </pre>
            <p v-else class="hint">
                提示：后端默认是 `http://127.0.0.1:8000`。
            </p>
        </div>
    </section>
</template>

<style scoped>
.page {
    width: min(1100px, calc(100% - 32px));
    margin: 0 auto;
    padding: 18px 0 40px;
    display: grid;
    gap: 14px;
}
.header h1 {
    margin: 0 0 6px;
    font-size: 28px;
}
.sub {
    margin: 0;
    color: var(--text);
}
.panel {
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 14px;
    background: color-mix(in oklab, var(--bg) 92%, var(--code-bg));
    box-shadow: var(--shadow);
    display: grid;
    gap: 12px;
}
.form {
    display: flex;
    flex-wrap: wrap;
    gap: 10px 12px;
    align-items: flex-end;
    justify-content: flex-start;
}
.field {
    display: grid;
    gap: 6px;
    font-size: 14px;
}
input {
    height: 34px;
    border-radius: 10px;
    padding: 0 10px;
    border: 1px solid var(--border);
    background: color-mix(in oklab, var(--bg) 92%, var(--code-bg));
    color: var(--text-h);
}
.uploadPreview {
    border-radius: 18px;
    overflow: hidden;
    border: 1px solid var(--border);
    background: color-mix(in oklab, var(--bg) 92%, var(--code-bg));
}
.uploadPreview img {
    width: 100%;
    display: block;
}
.btn {
    height: 34px;
    padding: 0 14px;
    border-radius: 10px;
    border: 1px solid var(--border);
    background: color-mix(in oklab, var(--bg) 92%, var(--code-bg));
    color: var(--text-h);
    cursor: pointer;
}
.btn.primary {
    border-color: color-mix(in oklab, var(--accent) 55%, var(--border));
    background: color-mix(in oklab, var(--accent-bg) 45%, var(--bg));
}
.error {
    margin: 0;
    font-size: 14px;
    color: color-mix(in oklab, #ff3b30 80%, var(--text));
}
.success {
    margin: 0;
    font-size: 14px;
    color: color-mix(in oklab, #34c759 80%, var(--text));
}
.hint {
    margin: 0;
    font-size: 14px;
}
.pre {
    margin: 0;
    padding: 10px 12px;
    border-radius: 12px;
    border: 1px solid var(--border);
    background: var(--code-bg);
    font-size: 12px;
    overflow: auto;
}
</style>
