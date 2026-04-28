import { createRouter, createWebHistory, type RouteRecordRaw } from 'vue-router'

import EnrollPage from './pages/EnrollPage.vue'
import IdentifyPage from './pages/IdentifyPage.vue'
import PeoplePage from './pages/PeoplePage.vue'

const routes: RouteRecordRaw[] = [
    { path: '/', redirect: '/enroll' },
    { path: '/enroll', name: 'enroll', component: EnrollPage },
    { path: '/identify', name: 'identify', component: IdentifyPage },
    { path: '/people', name: 'people', component: PeoplePage },
]

export const router = createRouter({
    history: createWebHistory(),
    routes,
})
