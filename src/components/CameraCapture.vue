<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'

type FacingMode = 'user' | 'environment'

const props = defineProps<{
    initialFacingMode?: FacingMode
}>()

const emit = defineEmits<{
    (e: 'captured', payload: { blob: Blob; dataUrl: string }): void
    (e: 'error', message: string): void
}>()

const videoEl = ref<HTMLVideoElement | null>(null)
const previewUrl = ref<string>('')
const isRunning = ref(false)

const facingMode = ref<FacingMode>(props.initialFacingMode ?? 'user')
// 实际摄像头朝向（以 track.getSettings() 为准）
const actualFacingMode = ref<FacingMode>('user')
const cameras = ref<{ deviceId: string; label: string }[]>([])
const selectedDeviceId = ref<string>('')

let stream: MediaStream | null = null

const canStart = computed(() => !isRunning.value && !!videoEl.value)
const canCapture = computed(() => isRunning.value && !!videoEl.value)
const isMirrored = computed(() => actualFacingMode.value === 'user')

function stopStream() {
    if (stream) {
        stream.getTracks().forEach((t) => t.stop())
        stream = null
    }
    if (videoEl.value) videoEl.value.srcObject = null
    isRunning.value = false
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

function buildConstraints(): MediaTrackConstraints | boolean {
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

async function start() {
    previewUrl.value = ''

    if (!window.isSecureContext) {
        emit(
            'error',
            '当前不是安全上下文：摄像头只允许在 https 或 localhost 下使用。'
        )
        return
    }
    if (!navigator.mediaDevices?.getUserMedia) {
        emit('error', '当前浏览器不支持 getUserMedia（无法访问摄像头）。')
        return
    }

    stopStream()
    try {
        // 先触发权限弹窗（部分浏览器未授权时 enumerateDevices 不完整）
        stream = await navigator.mediaDevices.getUserMedia({
            audio: false,
            video: buildConstraints(),
        })
        // 读取真实 facingMode（iPad 上用 deviceId 选择时，UI facingMode 可能不等于实际）
        try {
            const track = stream.getVideoTracks()[0]
            const fm = (track?.getSettings?.() as any)?.facingMode as
                | FacingMode
                | undefined
            if (fm === 'user' || fm === 'environment')
                actualFacingMode.value = fm
            else actualFacingMode.value = facingMode.value
        } catch {
            actualFacingMode.value = facingMode.value
        }
        const v = videoEl.value!
        v.srcObject = stream
        await v.play()
        isRunning.value = true
        await refreshCameraList()
    } catch (e) {
        const msg =
            e instanceof DOMException && e.name === 'NotAllowedError'
                ? '摄像头权限被拒绝。请在浏览器设置中允许摄像头权限后重试。'
                : e instanceof DOMException && e.name === 'NotFoundError'
                ? '未找到可用摄像头设备。'
                : e instanceof Error
                ? e.message
                : String(e)
        emit('error', msg)
        stopStream()
    }
}

async function restart() {
    if (!isRunning.value) return
    await start()
}

async function captureFrame(opts?: {
    updatePreview?: boolean
    quality?: number
}) {
    const v = videoEl.value
    if (!v) throw new Error('video 未就绪')
    const w = v.videoWidth
    const h = v.videoHeight
    if (!w || !h) {
        throw new Error('摄像头画面尚未就绪，请稍后再试。')
    }

    const canvas = document.createElement('canvas')
    canvas.width = w
    canvas.height = h
    const ctx = canvas.getContext('2d')
    if (!ctx) {
        throw new Error('无法创建画布上下文。')
    }

    // 和视频显示一致：镜像自拍更直观（只镜像 user）
    if (facingMode.value === 'user') {
        ctx.translate(w, 0)
        ctx.scale(-1, 1)
    }
    ctx.drawImage(v, 0, 0, w, h)

    const quality = opts?.quality ?? 0.9
    const dataUrl = canvas.toDataURL('image/jpeg', quality)
    if (opts?.updatePreview !== false) previewUrl.value = dataUrl

    const blob: Blob = await new Promise((resolve, reject) => {
        canvas.toBlob(
            (b) => {
                if (!b) reject(new Error('生成图片失败。'))
                else resolve(b)
            },
            'image/jpeg',
            quality
        )
    })

    return { blob, dataUrl }
}

async function capture() {
    try {
        const { blob, dataUrl } = await captureFrame({
            updatePreview: true,
            quality: 0.92,
        })
        emit('captured', { blob, dataUrl })
    } catch (e) {
        emit('error', e instanceof Error ? e.message : String(e))
    }
}

function clearPreview() {
    previewUrl.value = ''
}

watch([selectedDeviceId, facingMode], restart)

defineExpose({
    /** 抓取当前视频帧（不一定更新预览） */
    captureFrame,
    /** 获取 video 元素（用于上层判断是否运行中） */
    getVideoEl: () => videoEl.value,
    /** 当前是否在运行 */
    getIsRunning: () => isRunning.value,
    /** 当前画面是否做了左右镜像（前置摄像头通常镜像显示） */
    getIsMirrored: () => isMirrored.value,
    /** 实际摄像头朝向（基于 track settings） */
    getFacingMode: () => actualFacingMode.value,
})

onMounted(async () => {
    try {
        await refreshCameraList()
        navigator.mediaDevices?.addEventListener?.(
            'devicechange',
            refreshCameraList
        )
    } catch {
        // ignore
    }
})

onBeforeUnmount(() => {
    navigator.mediaDevices?.removeEventListener?.(
        'devicechange',
        refreshCameraList
    )
    stopStream()
})
</script>

<template>
    <div class="wrap">
        <div class="controls">
            <label class="field">
                <span>前/后置</span>
                <select v-model="facingMode" :disabled="isRunning">
                    <option value="user">前置</option>
                    <option value="environment">后置</option>
                </select>
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

            <div class="actions">
                <button
                    class="btn primary"
                    :disabled="!canStart"
                    @click="start"
                >
                    开始
                </button>
                <button class="btn" :disabled="!isRunning" @click="stopStream">
                    停止
                </button>
                <button class="btn" :disabled="!canCapture" @click="capture">
                    拍照
                </button>
                <button
                    class="btn"
                    :disabled="!previewUrl"
                    @click="clearPreview"
                >
                    清除预览
                </button>
            </div>
        </div>

        <div class="stage">
            <div class="videoWrap" :class="{ mirrored: isMirrored }">
                <video ref="videoEl" class="video" playsinline muted></video>
                <div class="overlay">
                    <slot name="overlay" />
                </div>
            </div>
            <div v-if="previewUrl" class="preview">
                <img :src="previewUrl" alt="preview" />
            </div>
        </div>
    </div>
</template>

<style scoped>
.wrap {
    display: grid;
    gap: 12px;
}
.controls {
    display: flex;
    flex-wrap: wrap;
    gap: 10px 12px;
    align-items: flex-end;
}
.field {
    display: grid;
    gap: 6px;
    font-size: 14px;
}
.field > span {
    opacity: 0.85;
}
select {
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
    flex-wrap: wrap;
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
.video {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    object-fit: cover;
}
.videoWrap.mirrored .video {
    transform: scaleX(-1);
}
.overlay {
    position: absolute;
    inset: 0;
    pointer-events: none;
}
.preview {
    border-radius: 18px;
    overflow: hidden;
    border: 1px solid var(--border);
    background: color-mix(in oklab, var(--bg) 92%, var(--code-bg));
}
.preview img {
    width: 100%;
    display: block;
}
</style>
