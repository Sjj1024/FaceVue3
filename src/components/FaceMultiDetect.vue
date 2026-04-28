<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import {
    FaceLandmarker,
    FilesetResolver,
    type NormalizedLandmark,
} from '@mediapipe/tasks-vision'

type CameraOption = { deviceId: string; label: string }
type FacingMode = 'user' | 'environment'

// 视频元素
const videoEl = ref<HTMLVideoElement | null>(null)
// 画布元素
const canvasEl = ref<HTMLCanvasElement | null>(null)

const cameras = ref<CameraOption[]>([])
// 选中设备 ID
const selectedDeviceId = ref<string>('')
// 摄像头朝向
const facingMode = ref<FacingMode>('user')
// 是否运行
const isRunning = ref(false)
// 最近一次错误信息
const lastError = ref<string>('')

// 阈值 threshold：越小越像
const scoreThreshold = ref(0.5)
// 最多人脸
const maxFaces = ref(6)
// 是否显示人脸特征点
const showLandmarks = ref(true)
// 是否在控制台输出人脸特征点
const consoleLogLandmarks = ref(false)
// 人脸特征点预览数量
const landmarksPreviewCount = ref(12)
// 最近一次人脸特征点
const lastLandmarks = ref<NormalizedLandmark[][]>([])
// 最近一次人脸特征点时间
const lastLandmarksAt = ref<number>(0)
// 人脸检测器
let landmarker: FaceLandmarker | null = null
let stream: MediaStream | null = null
let rafId: number | null = null

const canStart = computed(
    () => !isRunning.value && !!videoEl.value && !!canvasEl.value
)

function buildVideoConstraints(): MediaTrackConstraints | boolean {
    // iOS/iPadOS 上很多场景 enumerateDevices 只有 1 个 videoinput，
    // 或 deviceId 不可用；此时靠 facingMode 切换前/后置更稳定。
    if (selectedDeviceId.value) {
        return {
            deviceId: { exact: selectedDeviceId.value },
            width: { ideal: 1280 },
            height: { ideal: 720 },
        }
    }

    return {
        facingMode: { ideal: facingMode.value },
        width: { ideal: 1280 },
        height: { ideal: 720 },
    }
}

async function startStreamWithFallback() {
    const video = buildVideoConstraints()
    try {
        return await navigator.mediaDevices.getUserMedia({
            audio: false,
            video,
        })
    } catch (e) {
        // iOS 对 exact/ideal 兼容有差异，失败则逐级降级
        if (
            e instanceof DOMException &&
            (e.name === 'OverconstrainedError' ||
                e.name === 'NotFoundError' ||
                e.name === 'NotReadableError')
        ) {
            try {
                return await navigator.mediaDevices.getUserMedia({
                    audio: false,
                    video: {
                        facingMode: { ideal: facingMode.value },
                        width: { ideal: 1280 },
                        height: { ideal: 720 },
                    },
                })
            } catch {
                return await navigator.mediaDevices.getUserMedia({
                    audio: false,
                    video: true,
                })
            }
        }
        throw e
    }
}

async function ensureLandmarker() {
    if (landmarker) return landmarker

    const vision = await FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.34/wasm'
    )

    landmarker = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath:
                'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
        },
        runningMode: 'VIDEO',
        numFaces: maxFaces.value,
        minFaceDetectionConfidence: scoreThreshold.value,
    })

    return landmarker
}

async function refreshCameraList() {
    const devices = await navigator.mediaDevices.enumerateDevices()
    const opts = devices
        .filter((d) => d.kind === 'videoinput')
        .map((d, idx) => ({
            deviceId: d.deviceId,
            label: d.label || `摄像头 ${idx + 1}`,
        }))

    cameras.value = opts
    if (!selectedDeviceId.value && opts.length > 0)
        selectedDeviceId.value = opts[0].deviceId
}

function stopStream() {
    if (rafId != null) {
        cancelAnimationFrame(rafId)
        rafId = null
    }

    if (stream) {
        for (const track of stream.getTracks()) track.stop()
        stream = null
    }

    const v = videoEl.value
    if (v) v.srcObject = null
}

function resizeCanvasToVideo() {
    const v = videoEl.value
    const c = canvasEl.value
    if (!v || !c) return
    const w = v.videoWidth || 0
    const h = v.videoHeight || 0
    if (w > 0 && h > 0) {
        c.width = w
        c.height = h
    }
}

function computeFaceBox(landmarks: NormalizedLandmark[], w: number, h: number) {
    let minX = Number.POSITIVE_INFINITY
    let minY = Number.POSITIVE_INFINITY
    let maxX = Number.NEGATIVE_INFINITY
    let maxY = Number.NEGATIVE_INFINITY

    for (const p of landmarks) {
        if (p.x < minX) minX = p.x
        if (p.y < minY) minY = p.y
        if (p.x > maxX) maxX = p.x
        if (p.y > maxY) maxY = p.y
    }

    const x = Math.max(0, Math.min(w, minX * w))
    const y = Math.max(0, Math.min(h, minY * h))
    const bw = Math.max(0, Math.min(w - x, (maxX - minX) * w))
    const bh = Math.max(0, Math.min(h - y, (maxY - minY) * h))

    return { x, y, w: bw, h: bh }
}

function drawFaces(faces: NormalizedLandmark[][]) {
    const c = canvasEl.value
    const v = videoEl.value
    if (!c || !v) return

    const ctx = c.getContext('2d')
    if (!ctx) return

    ctx.clearRect(0, 0, c.width, c.height)

    // 镜像显示：让画面像自拍一样更直观
    ctx.save()
    ctx.translate(c.width, 0)
    ctx.scale(-1, 1)

    const lineW = Math.max(2, Math.round(c.width / 320))
    ctx.lineWidth = lineW
    ctx.font = `${Math.max(
        14,
        Math.round(c.width / 40)
    )}px ui-monospace, SFMono-Regular, Menlo, monospace`

    faces.forEach((landmarks, i) => {
        if (!landmarks?.length) return
        const { x, y, w, h } = computeFaceBox(landmarks, c.width, c.height)
        const label = `face #${i + 1}`

        ctx.strokeStyle = 'rgba(0, 255, 140, 0.95)'
        ctx.fillStyle = 'rgba(0, 255, 140, 0.18)'
        ctx.fillRect(x, y, w, h)
        ctx.strokeRect(x, y, w, h)

        const pad = 6
        const textW = ctx.measureText(label).width
        const textH = parseInt(ctx.font, 10)
        const bx = x
        const by = Math.max(0, y - (textH + pad * 2))

        ctx.fillStyle = 'rgba(0, 0, 0, 0.6)'
        ctx.fillRect(bx, by, textW + pad * 2, textH + pad * 2)
        ctx.fillStyle = 'white'
        ctx.fillText(label, bx + pad, by + textH + pad - 2)
    })

    ctx.restore()
}

async function tick() {
    const v = videoEl.value
    const c = canvasEl.value
    if (!v || !c || !isRunning.value) return

    try {
        const lm = await ensureLandmarker()
        const res = lm.detectForVideo(v, performance.now())
        drawFaces(res.faceLandmarks ?? [])

        const faces = res.faceLandmarks ?? []
        lastLandmarks.value = faces
        lastLandmarksAt.value = Date.now()
        if (consoleLogLandmarks.value && faces.length > 0) {
            // 避免刷屏：只输出一张脸的前 N 个点
            const n = Math.max(
                1,
                Math.min(landmarksPreviewCount.value, faces[0].length)
            )
            // eslint-disable-next-line no-console
            console.log('[faceLandmarks]', {
                at: new Date(lastLandmarksAt.value).toISOString(),
                faces: faces.length,
                firstFacePreview: faces[0].slice(0, n),
            })
        }
    } catch (e) {
        lastError.value = e instanceof Error ? e.message : String(e)
        await stop()
        return
    }

    rafId = requestAnimationFrame(tick)
}

async function start() {
    lastError.value = ''
    if (!canStart.value) return
    if (!window.isSecureContext) {
        lastError.value =
            '当前不是安全上下文：摄像头只允许在 https 或 localhost 下使用。'
        return
    }
    if (!navigator.mediaDevices?.getUserMedia) {
        lastError.value = '当前浏览器不支持 getUserMedia（无法访问摄像头）。'
        return
    }

    stopStream()

    try {
        // 先触发权限弹窗：部分浏览器在未授权前 enumerateDevices 可能返回空
        stream = await startStreamWithFallback()

        const v = videoEl.value!
        v.srcObject = stream
        await v.play()

        await refreshCameraList()
        if (!selectedDeviceId.value) {
            const track = stream.getVideoTracks()[0]
            const settings = track?.getSettings?.()
            if (settings?.deviceId) selectedDeviceId.value = settings.deviceId
        }

        resizeCanvasToVideo()
        isRunning.value = true

        rafId = requestAnimationFrame(tick)
    } catch (e) {
        lastError.value =
            e instanceof DOMException && e.name === 'NotAllowedError'
                ? '摄像头权限被拒绝。请在浏览器地址栏/设置中允许摄像头权限后刷新重试。'
                : e instanceof DOMException && e.name === 'NotFoundError'
                ? '未找到可用摄像头设备。'
                : e instanceof DOMException && e.name === 'NotReadableError'
                ? '摄像头可能被其它应用占用，或系统拒绝了访问。'
                : e instanceof Error
                ? e.message
                : String(e)
        stopStream()
    }
}

async function stop() {
    isRunning.value = false
    stopStream()
    if (canvasEl.value) {
        const ctx = canvasEl.value.getContext('2d')
        ctx?.clearRect(0, 0, canvasEl.value.width, canvasEl.value.height)
    }
    lastLandmarks.value = []
    lastLandmarksAt.value = 0
}

watch([scoreThreshold, maxFaces], async () => {
    // 运行中调整参数：重建 detector 以应用新配置
    if (!isRunning.value) return
    if (landmarker) {
        landmarker.close()
        landmarker = null
    }
})

watch([selectedDeviceId, facingMode], async () => {
    // 运行中切换前/后置或具体设备：自动重启 stream
    if (!isRunning.value) return
    await stop()
    await start()
})

onMounted(async () => {
    lastError.value = ''
    if (!navigator.mediaDevices?.enumerateDevices) {
        lastError.value = '当前浏览器不支持 enumerateDevices。'
        return
    }

    await refreshCameraList()

    // 设备 label 通常需要在用户授权摄像头后才能拿到
    navigator.mediaDevices.addEventListener?.('devicechange', refreshCameraList)
})

onBeforeUnmount(async () => {
    navigator.mediaDevices.removeEventListener?.(
        'devicechange',
        refreshCameraList
    )
    await stop()
    if (landmarker) landmarker.close()
    landmarker = null
})
</script>

<template>
    <section class="page">
        <header class="header">
            <div class="title">
                <h1>Vue3 浏览器端人脸识别（多脸）</h1>
                <p class="sub">
                    打开后会申请摄像头权限，在本地实时检测多张人脸并画框（不上传视频）。
                </p>
            </div>

            <div class="controls">
                <label class="field">
                    <span>前/后置</span>
                    <select v-model="facingMode" :disabled="isRunning">
                        <option value="user">前置</option>
                        <option value="environment">后置</option>
                    </select>
                </label>

                <label class="field">
                    <span>打印特征点</span>
                    <select v-model="showLandmarks">
                        <option :value="true">页面显示</option>
                        <option :value="false">隐藏</option>
                    </select>
                </label>

                <label class="field">
                    <span>控制台输出</span>
                    <select v-model="consoleLogLandmarks">
                        <option :value="false">关闭</option>
                        <option :value="true">开启</option>
                    </select>
                </label>

                <label class="field">
                    <span>预览点数</span>
                    <input
                        v-model.number="landmarksPreviewCount"
                        type="number"
                        min="1"
                        max="100"
                    />
                </label>

                <label class="field">
                    <span>摄像头</span>
                    <select
                        v-model="selectedDeviceId"
                        :disabled="isRunning || cameras.length === 0"
                    >
                        <option value="">自动/使用前后置选择</option>
                        <option
                            v-for="c in cameras"
                            :key="c.deviceId"
                            :value="c.deviceId"
                        >
                            {{ c.label }}
                        </option>
                    </select>
                </label>

                <label class="field">
                    <span>最多人脸</span>
                    <input
                        v-model.number="maxFaces"
                        type="number"
                        min="1"
                        max="50"
                        :disabled="isRunning"
                    />
                </label>

                <label class="field">
                    <span>阈值</span>
                    <input
                        v-model.number="scoreThreshold"
                        type="number"
                        min="0"
                        max="1"
                        step="0.05"
                        :disabled="isRunning"
                    />
                </label>

                <div class="actions">
                    <button
                        class="btn primary"
                        :disabled="!canStart"
                        @click="start"
                    >
                        开始
                    </button>
                    <button class="btn" :disabled="!isRunning" @click="stop">
                        停止
                    </button>
                </div>
            </div>
        </header>

        <div class="stage">
            <div class="videoWrap" :class="{ running: isRunning }">
                <video ref="videoEl" class="video" playsinline muted></video>
                <canvas ref="canvasEl" class="overlay" />
            </div>

            <p v-if="lastError" class="error">{{ lastError }}</p>
            <p v-else class="hint">
                提示：需要在 <code>https</code> 或
                <code>localhost</code> 下才能调用摄像头。
            </p>
        </div>
    </section>

    <section v-if="showLandmarks" class="landmarks">
        <header class="landmarksHeader">
            <h2>人脸特征点（faceLandmarks）</h2>
            <p class="landmarksMeta">
                {{ isRunning ? '运行中' : '未运行' }} ·
                {{ lastLandmarks.length }} 张脸 ·
                <span v-if="lastLandmarksAt">
                    {{ new Date(lastLandmarksAt).toLocaleTimeString() }}
                </span>
                <span v-else>暂无数据</span>
            </p>
        </header>

        <pre class="landmarksPre">{{
            JSON.stringify(
                lastLandmarks.map((face, idx) => ({
                    face: idx + 1,
                    totalPoints: face.length,
                    preview: face.slice(
                        0,
                        Math.max(
                            1,
                            Math.min(landmarksPreviewCount, face.length)
                        )
                    ),
                })),
                null,
                2
            )
        }}</pre>
        <p class="landmarksHint">
            说明：坐标为归一化 \(x,y\in[0,1]\)，z 为相对深度；每张脸通常有 468
            个点。
        </p>
    </section>
</template>

<style scoped>
.page {
    width: min(1100px, calc(100% - 32px));
    margin: 0 auto;
    padding: 28px 0 40px;
    display: flex;
    flex-direction: column;
    gap: 18px;
}

.header {
    display: flex;
    flex-wrap: wrap;
    gap: 18px 22px;
    align-items: flex-end;
    justify-content: space-between;
}

.title h1 {
    margin: 0 0 6px;
    font-size: 34px;
}

.sub {
    margin: 0;
    color: var(--text);
}

.controls {
    display: flex;
    flex-wrap: wrap;
    gap: 10px 12px;
    align-items: flex-end;
    justify-content: flex-end;
}

.field {
    display: grid;
    gap: 6px;
    font-size: 14px;
}

.field > span {
    opacity: 0.85;
}

select,
input {
    height: 34px;
    border-radius: 10px;
    padding: 0 10px;
    border: 1px solid var(--border);
    background: color-mix(in oklab, var(--bg) 92%, var(--code-bg));
    color: var(--text-h);
}

.actions {
    display: flex;
    gap: 10px;
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
.btn:disabled {
    opacity: 0.55;
    cursor: not-allowed;
}
.btn.primary {
    border-color: color-mix(in oklab, var(--accent) 55%, var(--border));
    background: color-mix(in oklab, var(--accent-bg) 45%, var(--bg));
    color: var(--text-h);
}

.stage {
    display: grid;
    gap: 10px;
}

.videoWrap {
    position: relative;
    width: 100%;
    aspect-ratio: 16 / 9;
    border-radius: 18px;
    overflow: hidden;
    border: 1px solid var(--border);
    background: color-mix(in oklab, var(--bg) 72%, black);
    box-shadow: var(--shadow);
}

.video,
.overlay {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    object-fit: cover;
}

.video {
    transform: scaleX(-1);
}

.overlay {
    pointer-events: none;
}

.hint,
.error {
    margin: 0;
    font-size: 14px;
}

.error {
    color: color-mix(in oklab, #ff3b30 80%, var(--text));
}

.landmarks {
    width: min(1100px, calc(100% - 32px));
    margin: 0 auto 40px;
    border: 1px solid var(--border);
    border-radius: 18px;
    overflow: hidden;
    background: color-mix(in oklab, var(--bg) 92%, var(--code-bg));
    box-shadow: var(--shadow);
}

.landmarksHeader {
    padding: 14px 16px 10px;
    border-bottom: 1px solid var(--border);
}

.landmarksHeader h2 {
    margin: 0 0 6px;
    font-size: 18px;
}

.landmarksMeta {
    margin: 0;
    font-size: 13px;
    opacity: 0.9;
}

.landmarksPre {
    margin: 0;
    padding: 12px 16px;
    max-height: 260px;
    overflow: auto;
    font-size: 12px;
    line-height: 1.4;
}

.landmarksHint {
    margin: 0;
    padding: 10px 16px 14px;
    font-size: 12px;
    border-top: 1px solid var(--border);
    opacity: 0.9;
}
</style>
