<template>
  <div class="organism-health">
    <h3>Organism Health</h3>
    <div class="metric"><span>Cortisol</span><strong :class="hormoneClass(state.cortisol)">{{ state.cortisol.toFixed(2) }}</strong></div>
    <div class="metric"><span>Dopamine</span><strong>{{ state.dopamine.toFixed(2) }}</strong></div>
    <div class="metric"><span>Serotonin</span><strong>{{ state.serotonin.toFixed(2) }}</strong></div>
    <div class="metric"><span>Adrenaline</span><strong>{{ state.adrenaline.toFixed(2) }}</strong></div>
    <div class="metric"><span>Market Stress</span><strong>{{ state.market_stress.toFixed(2) }}</strong></div>
    <div class="metric"><span>Portfolio Health</span><strong>{{ state.portfolio_health.toFixed(2) }}</strong></div>
    <div class="metric"><span>Streak</span><strong>{{ state.streak }}</strong></div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'

const state = ref({
  cortisol: 1, dopamine: 1, serotonin: 1, adrenaline: 1,
  market_stress: 0, portfolio_health: 0.5, streak: 0,
})
let timer: ReturnType<typeof setInterval> | null = null

function hormoneClass(v: number) {
  if (v < 0.6) return 'low'
  if (v > 1.4) return 'high'
  return 'mid'
}

async function refresh() {
  try {
    const r = await fetch('/api/v1/ai/organism')
    if (r.ok) state.value = await r.json()
  } catch (e) {
    console.error('[OrganismHealth] fetch failed', e)
  }
}

onMounted(() => {
  refresh()
  timer = setInterval(refresh, 10000)
})
onUnmounted(() => {
  if (timer) clearInterval(timer)
})
</script>

<style scoped>
.organism-health { padding: 12px; background: #0a0a0a; color: #0f0; font-family: monospace; }
.metric { display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid #0a4; }
.low { color: #ff0; } .high { color: #f55; } .mid { color: #0f0; }
</style>
