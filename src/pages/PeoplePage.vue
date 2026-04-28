<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'

// 开发环境走 Vite 代理，避免 CORS：/api -> VITE_BACKEND_BASE
const API_BASE = '/api'

type PersonItem = {
    person_id: string
    name: string
    embeddings: number
    last_embedding_at: string | null
    created_at: string
    updated_at: string
}

/** 加载状态 */
const loading = ref(false)
//
/** 错误信息 */
const error = ref('')
/** 人员列表 */
const items = ref<PersonItem[]>([])
/** 删除中的人的 person_id */
const deletingId = ref<string>('')
/** 提示信息 */
const toast = ref<string>('')

/** 总人数 */
const total = computed(() => items.value.length)
/** 总 embedding 数 */
const totalEmbeddings = computed(() =>
    items.value.reduce((sum, p) => sum + (p.embeddings || 0), 0)
)

/** 格式化时间 */
function fmtTime(s: string | null) {
    if (!s) return '—'
    const d = new Date(s)
    if (Number.isNaN(d.getTime())) return s
    return d.toLocaleString()
}

/** 刷新人员列表 */
async function refresh() {
    loading.value = true
    error.value = ''
    toast.value = ''
    try {
        const resp = await fetch(`${API_BASE}/people`)
        const data = await resp.json().catch(() => ({}))
        if (!resp.ok) {
            error.value = data?.detail
                ? String(data.detail)
                : `请求失败：HTTP ${resp.status}`
            items.value = []
            return
        }
        items.value = Array.isArray(data?.items)
            ? (data.items as PersonItem[])
            : []
    } catch (e) {
        error.value = e instanceof Error ? e.message : String(e)
        items.value = []
    } finally {
        loading.value = false
    }
}

// 删除人员
async function deletePerson(p: PersonItem) {
    toast.value = ''
    error.value = ''

    const ok = window.confirm(
        `确定删除“${p.name}”（person_id=${p.person_id}）吗？\n这会同时删除该人的所有 embeddings。`
    )
    if (!ok) return

    deletingId.value = p.person_id
    try {
        const resp = await fetch(
            `${API_BASE}/people/${encodeURIComponent(p.person_id)}`,
            {
                method: 'DELETE',
            }
        )
        const data = await resp.json().catch(() => ({}))
        if (!resp.ok) {
            error.value = data?.detail
                ? String(data.detail)
                : `删除失败：HTTP ${resp.status}`
            return
        }
        toast.value = `已删除：${p.name}（embeddings_deleted=${
            data?.embeddings_deleted ?? 0
        }）`
        await refresh()
    } catch (e) {
        error.value = e instanceof Error ? e.message : String(e)
    } finally {
        deletingId.value = ''
    }
}

// 组件挂载后刷新人员列表
onMounted(refresh)
</script>

<template>
    <section class="page">
        <header class="header">
            <div>
                <h1>已标注人员列表</h1>
                <p class="sub">
                    从后端 <code>/people</code> 拉取当前已录入的名字与 embedding
                    统计。
                </p>
            </div>
            <div class="actions">
                <button class="btn" :disabled="loading" @click="refresh">
                    {{ loading ? '刷新中...' : '刷新' }}
                </button>
            </div>
        </header>

        <p v-if="error" class="error">{{ error }}</p>
        <p v-else-if="toast" class="toast">{{ toast }}</p>

        <div class="summary" v-else>
            <div class="pill">
                人数：<b>{{ total }}</b>
            </div>
            <div class="pill">
                embedding 总数：<b>{{ totalEmbeddings }}</b>
            </div>
            <div class="pill muted">
                后端：<code>{{ API_BASE }}</code>
            </div>
        </div>

        <div class="tableWrap">
            <table class="table">
                <thead>
                    <tr>
                        <th>姓名</th>
                        <th>person_id</th>
                        <th class="num">embeddings</th>
                        <th>最近录入</th>
                        <th>更新时间</th>
                        <th class="ops">操作</th>
                    </tr>
                </thead>
                <tbody>
                    <tr v-if="!loading && items.length === 0">
                        <td colspan="6" class="empty">
                            暂无数据。请先去“录入”页面添加人脸。
                        </td>
                    </tr>
                    <tr v-for="p in items" :key="p.person_id">
                        <td class="name">{{ p.name }}</td>
                        <td>
                            <code>{{ p.person_id }}</code>
                        </td>
                        <td class="num">{{ p.embeddings }}</td>
                        <td>{{ fmtTime(p.last_embedding_at) }}</td>
                        <td>{{ fmtTime(p.updated_at) }}</td>
                        <td class="ops">
                            <button
                                class="btn danger"
                                :disabled="deletingId === p.person_id"
                                @click="deletePerson(p)"
                            >
                                {{
                                    deletingId === p.person_id
                                        ? '删除中…'
                                        : '删除'
                                }}
                            </button>
                        </td>
                    </tr>
                </tbody>
            </table>
        </div>
    </section>
</template>

<style scoped>
.page {
    width: min(1100px, calc(100% - 32px));
    margin: 0 auto;
    padding: 18px 0 40px;
    display: grid;
    gap: 12px;
}
.header {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    gap: 12px;
    flex-wrap: wrap;
}
.header h1 {
    margin: 0 0 6px;
    font-size: 28px;
}
.sub {
    margin: 0;
    color: var(--text);
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
.btn.danger {
    border-color: color-mix(in oklab, #ff3b30 55%, var(--border));
    background: color-mix(in oklab, #ff3b30 18%, var(--bg));
}
.error {
    margin: 0;
    font-size: 14px;
    color: color-mix(in oklab, #ff3b30 80%, var(--text));
}
.toast {
    margin: 0;
    font-size: 14px;
    color: color-mix(in oklab, #34c759 70%, var(--text-h));
}
.summary {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}
.pill {
    border: 1px solid var(--border);
    padding: 6px 10px;
    border-radius: 999px;
    background: color-mix(in oklab, var(--bg) 92%, var(--code-bg));
    font-size: 13px;
    color: var(--text-h);
}
.pill.muted {
    color: var(--text);
}
.tableWrap {
    border: 1px solid var(--border);
    border-radius: 18px;
    overflow: hidden;
    background: color-mix(in oklab, var(--bg) 92%, var(--code-bg));
    box-shadow: var(--shadow);
}
.table {
    width: 100%;
    border-collapse: collapse;
}
th,
td {
    padding: 12px 12px;
    border-bottom: 1px solid var(--border);
    text-align: left;
    vertical-align: top;
    font-size: 14px;
}
th {
    font-size: 13px;
    opacity: 0.9;
}
.ops {
    text-align: right;
    white-space: nowrap;
}
.num {
    text-align: right;
    font-variant-numeric: tabular-nums;
}
.name {
    font-weight: 600;
    color: var(--text-h);
}
.empty {
    text-align: center;
    opacity: 0.85;
}
tbody tr:hover td {
    background: color-mix(in oklab, var(--accent-bg) 18%, transparent);
}
</style>
