<script setup lang="ts">
import { RouterLink, RouterView } from 'vue-router'
</script>

<template>
    <div class="app">
        <header class="top">
            <div class="brand">
                <div class="title">人脸识别 Demo</div>
                <div class="sub">录入 + 识别（embedding）</div>
            </div>
            <nav class="nav">
                <RouterLink class="link" to="/enroll">录入</RouterLink>
                <RouterLink class="link" to="/identify">识别</RouterLink>
                <RouterLink class="link" to="/people">列表</RouterLink>
            </nav>
        </header>

        <main class="main">
            <RouterView />
        </main>
    </div>
</template>

<style scoped>
.app {
    width: 100%;
    display: grid;
    gap: 0;
}
.top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    padding: 14px 16px;
    border-bottom: 1px solid var(--border);
    background: color-mix(in oklab, var(--bg) 92%, var(--code-bg));
    position: sticky;
    top: 0;
    z-index: 10;
}
.brand .title {
    font-size: 16px;
    color: var(--text-h);
    font-weight: 600;
}
.brand .sub {
    font-size: 12px;
    opacity: 0.85;
}
.nav {
    display: flex;
    gap: 10px;
}
.link {
    padding: 6px 10px;
    border-radius: 10px;
    border: 1px solid var(--border);
    text-decoration: none;
    color: var(--text-h);
    background: color-mix(in oklab, var(--bg) 92%, var(--code-bg));
}
.link.router-link-active {
    border-color: color-mix(in oklab, var(--accent) 55%, var(--border));
    background: color-mix(in oklab, var(--accent-bg) 45%, var(--bg));
}
.main {
    padding: 0;
}
</style>
