<script setup lang="ts">
import { onBeforeUnmount, ref } from 'vue'
import CameraCapture from '../components/CameraCapture.vue'

// 开发环境走 Vite 代理，避免 CORS：/api -> VITE_BACKEND_BASE
const API_BASE = '/api'

// 阈值 threshold：越小越像
const threshold = ref(0.8)
// TopK：返回候选数量
const topK = ref(3)
// 最近一次错误信息
const lastError = ref('')
// 最近一次识别结果
const lastResult = ref<any>(null)
// 最近一次拍照的 Blob
const lastPhotoBlob = ref<Blob | null>(null)
// 最近一次拍照的文件名
const lastPhotoFilename = ref('capture.jpg')
// 最近一次拍照的预览 URL
const uploadPreviewUrl = ref('')

// 摄像头实例
const camRef = ref<InstanceType<typeof CameraCapture> | null>(null)
// 是否开启实时识别
const liveOn = ref(false)
// 最近一次实时识别的名字
const liveName = ref<string>('—')
// 最近一次实时识别的距离
const liveDistance = ref<number | null>(null)
// 最近一次实时识别的匹配结果
const liveMatched = ref<boolean | null>(null)
// 最近一次实时识别的信息
const liveInfo = ref<string>('')
// 最近一次实时识别的人脸
const liveFaces = ref<
    {
        bbox_xyxy: [number, number, number, number]
        matched: boolean
        best: {
            name: string | null
            person_id: string
            distance: number
        } | null
    }[]
>([])
const liveImageSize = ref<{ width: number; height: number } | null>(null)
let liveTimer: number | null = null
let liveInFlight = false

// 选择文件
function onFilePicked(e: Event) {
    lastError.value = ''
    lastResult.value = null

    const input = e.target as HTMLInputElement
    const file = input.files?.[0]
    if (!file) return

    lastPhotoBlob.value = file
    lastPhotoFilename.value = file.name || 'upload.jpg'
    uploadPreviewUrl.value = URL.createObjectURL(file)
}

// 清除已选照片
function clearUpload() {
    uploadPreviewUrl.value = ''
    lastPhotoBlob.value = null
    lastPhotoFilename.value = 'capture.jpg'
}

// 识别一次
async function identify() {
    lastError.value = ''
    lastResult.value = null

    if (!lastPhotoBlob.value) {
        lastError.value = '请先拍照或上传照片。'
        return
    }

    const form = new FormData()
    form.append('threshold', String(threshold.value))
    form.append('top_k', String(topK.value))
    form.append('image', lastPhotoBlob.value, lastPhotoFilename.value)

    try {
        const resp = await fetch(`${API_BASE}/identify`, {
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
    } catch (e) {
        lastError.value = e instanceof Error ? e.message : String(e)
    }
}

// 从摄像头识别一次
async function identifyOnceFromCamera() {
    if (!camRef.value) return
    if (!camRef.value.getIsRunning()) {
        liveInfo.value = '请先在下方点击“开始”打开摄像头。'
        return
    }
    if (liveInFlight) return
    liveInFlight = true
    try {
        const { blob } = await camRef.value.captureFrame({
            updatePreview: false,
            quality: 0.75,
        })
        const form = new FormData()
        form.append('threshold', String(threshold.value))
        form.append('top_k', String(topK.value))
        // 最多人脸
        form.append('max_faces', '5')
        form.append('image', blob, 'frame.jpg')

        const resp = await fetch(`${API_BASE}/identify_multi`, {
            method: 'POST',
            body: form,
        })
        const data = await resp.json().catch(() => ({}))
        if (!resp.ok) {
            liveInfo.value = data?.detail
                ? String(data.detail)
                : `识别失败：HTTP ${resp.status}`
            return
        }

        const faces = Array.isArray(data?.faces) ? data.faces : []
        liveImageSize.value =
            typeof data?.image?.width === 'number' &&
            typeof data?.image?.height === 'number' &&
            data.image.width > 0 &&
            data.image.height > 0
                ? { width: data.image.width, height: data.image.height }
                : null
        liveFaces.value = faces
            .map((f: any) => ({
                bbox_xyxy: f?.bbox_xyxy as [number, number, number, number],
                matched: Boolean(f?.matched),
                best: f?.best ?? null,
            }))
            .filter(
                (f: any) =>
                    Array.isArray(f.bbox_xyxy) && f.bbox_xyxy.length === 4
            )

        // 兼容原来的单条显示：取第一张“有名字”的脸（未知不展示）
        const first = liveFaces.value.find((x) => Boolean(x?.best?.name))
        liveMatched.value = first ? first.matched : null
        liveName.value = first?.best?.name ?? '—'
        liveDistance.value = first?.best?.distance ?? null
        liveInfo.value = ''
    } catch (e) {
        liveInfo.value = e instanceof Error ? e.message : String(e)
    } finally {
        liveInFlight = false
    }
}

// 计算左边百分比
function leftPct(x1: number, x2: number) {
    const s = liveImageSize.value
    if (!s) return '0%'
    const mirrored = camRef.value?.getIsMirrored?.() ?? false
    // 画面镜像时：x' = W - x，所以 left 应该用 (W - x2)
    const leftPx = mirrored ? s.width - x2 : x1
    return `${(leftPx / s.width) * 100}%`
}

// 计算顶部百分比
function topPct(y1: number) {
    const s = liveImageSize.value
    if (!s) return '0%'
    return `${(y1 / s.height) * 100}%`
}

// 计算宽度百分比
function widthPct(x1: number, x2: number) {
    const s = liveImageSize.value
    if (!s) return '0%'
    return `${((x2 - x1) / s.width) * 100}%`
}

// 计算高度百分比
function heightPct(y1: number, y2: number) {
    const s = liveImageSize.value
    if (!s) return '0%'
    return `${((y2 - y1) / s.height) * 100}%`
}

// 开启实时识别
function startLive() {
    liveOn.value = true
    liveInfo.value = ''

    const loop = async () => {
        if (!liveOn.value) return
        await identifyOnceFromCamera()
        liveTimer = window.setTimeout(loop, 900)
    }

    void loop()
}

function stopLive() {
    liveOn.value = false
    if (liveTimer != null) {
        clearTimeout(liveTimer)
        liveTimer = null
    }
}

onBeforeUnmount(stopLive)
</script>

<template>
    <section class="page">
        <header class="header">
            <div>
                <h1>识别（拍照 → 返回是谁）</h1>
                <p class="sub">
                    拍照后调用后端 /identify，返回最像的名字与候选列表。
                </p>
            </div>
        </header>

        <div class="panel">
            <div class="form">
                <label class="field">
                    <span>阈值 threshold</span>
                    <input
                        v-model.number="threshold"
                        type="number"
                        step="0.01"
                        min="0.05"
                        max="1.5"
                    />
                </label>
                <label class="field">
                    <span>TopK</span>
                    <input
                        v-model.number="topK"
                        type="number"
                        min="1"
                        max="20"
                    />
                </label>
                <label class="field">
                    <span>上传照片（可选）</span>
                    <input
                        type="file"
                        accept="image/*"
                        @change="onFilePicked"
                    />
                </label>
                <button class="btn primary" @click="identify">开始识别</button>
                <button
                    class="btn"
                    :class="{ primary: liveOn }"
                    @click="liveOn ? stopLive() : startLive()"
                >
                    {{ liveOn ? '停止实时识别' : '开启实时识别' }}
                </button>
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

            <CameraCapture
                ref="camRef"
                @captured="
                    ({ blob }) => (
                        (lastPhotoBlob = blob),
                        (lastPhotoFilename = 'capture.jpg'),
                        (uploadPreviewUrl = '')
                    )
                "
                @error="(m) => (lastError = m)"
            >
                <template #overlay>
                    <div class="liveLayer">
                        <div v-if="liveInfo" class="liveInfo">
                            {{ liveInfo }}
                        </div>

                        <div
                            v-for="(f, idx) in liveFaces.filter(
                                (x) => x.matched && Boolean(x.best?.name)
                            )"
                            :key="idx"
                            class="box"
                            :class="{ ok: f.matched }"
                            :style="{
                                left: leftPct(f.bbox_xyxy[0], f.bbox_xyxy[2]),
                                top: topPct(f.bbox_xyxy[1]),
                                width: widthPct(f.bbox_xyxy[0], f.bbox_xyxy[2]),
                                height: heightPct(
                                    f.bbox_xyxy[1],
                                    f.bbox_xyxy[3]
                                ),
                            }"
                        >
                            <div class="label">
                                {{ f.best?.name }}
                                <span
                                    class="dist"
                                    v-if="typeof f.best?.distance === 'number'"
                                >
                                    · d={{ f.best.distance.toFixed(3) }}
                                </span>
                            </div>
                        </div>
                    </div>
                </template>
            </CameraCapture>

            <p v-if="lastError" class="error">{{ lastError }}</p>

            <div v-else-if="lastResult" class="result">
                <div class="card">
                    <h2>识别结果</h2>
                    <p class="kv">
                        <b>matched</b>: <code>{{ lastResult.matched }}</code>
                    </p>
                    <p class="kv">
                        <b>best</b>:
                        <code>{{ lastResult.best?.name ?? '未知' }}</code>
                        <span class="muted"
                            >（person_id={{
                                lastResult.best?.person_id
                            }}）</span
                        >
                    </p>
                    <p class="kv">
                        <b>distance</b>:
                        <code>{{ lastResult.best?.distance }}</code>
                    </p>
                </div>

                <pre class="pre">{{ JSON.stringify(lastResult, null, 2) }}</pre>
            </div>

            <p v-else class="hint">
                提示：先到“录入”页面给人脸建档，再来识别。
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
.liveInfo {
    position: absolute;
    left: 12px;
    top: 12px;
    max-width: calc(100% - 24px);
    padding: 8px 10px;
    border-radius: 12px;
    border: 1px solid color-mix(in oklab, var(--border) 70%, transparent);
    background: rgba(0, 0, 0, 0.45);
    color: #ffd1d1;
    backdrop-filter: blur(8px);
    font-size: 12px;
}
.liveLayer {
    position: absolute;
    inset: 0;
}
.box {
    position: absolute;
    border: 2px solid rgba(255, 59, 48, 0.95);
    border-radius: 10px;
    box-sizing: border-box;
}
.box.ok {
    border-color: rgba(52, 199, 89, 0.95);
}
.box .label {
    position: absolute;
    left: 0;
    top: -30px;
    max-width: 100%;
    padding: 6px 8px;
    border-radius: 10px;
    background: rgba(0, 0, 0, 0.6);
    color: #fff;
    font-size: 12px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.box .dist {
    opacity: 0.85;
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
.hint {
    margin: 0;
    font-size: 14px;
}
.result {
    display: grid;
    gap: 12px;
}
.card {
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 12px;
    background: color-mix(in oklab, var(--bg) 92%, var(--code-bg));
}
.card h2 {
    margin: 0 0 8px;
    font-size: 18px;
}
.kv {
    margin: 0;
    font-size: 14px;
}
.muted {
    opacity: 0.85;
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
