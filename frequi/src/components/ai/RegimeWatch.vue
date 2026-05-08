<template>
  <div class="regime-watch">
    <h3>Regime Watch</h3>
    <div class="regime-grid">
      <div v-for="r in regimes" :key="r.pair" class="regime-row" :class="r.regime">
        <span class="pair">{{ r.pair }}</span>
        <span class="regime-tag">{{ r.regime }}</span>
        <span class="adx">ADX {{ r.adx.toFixed(1) }}</span>
        <span class="ttl">TTL {{ r.ttl_seconds }}s</span>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'

interface RegimeRow {
  pair: string
  regime: string
  adx: number
  ttl_seconds: number
}

const regimes = ref<RegimeRow[]>([])
let timer: ReturnType<typeof setInterval> | null = null

async function refresh() {
  try {
    const r = await fetch('/api/v1/ai/regime')
    if (r.ok) regimes.value = await r.json()
  } catch (e) {
    console.error('[RegimeWatch] fetch failed', e)
  }
}

onMounted(() => {
  refresh()
  timer = setInterval(refresh, 15000)
})
onUnmounted(() => {
  if (timer) clearInterval(timer)
})
</script>

<style scoped>
.regime-watch { padding: 12px; background: #0a0a0a; color: #0f0; font-family: monospace; }
.regime-grid { display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 6px; }
.regime-row { display: contents; }
.regime-row.bull { color: #0f0; }
.regime-row.bear { color: #f55; }
.regime-row.ranging { color: #888; }
.regime-row.volatile { color: #ff0; }
</style>
