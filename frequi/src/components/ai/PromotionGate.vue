<template>
  <div class="promotion-gate">
    <h3>Real-Capital Promotion Gate (D.9)</h3>
    <div :class="['status', gate.passed ? 'pass' : 'block']">
      {{ gate.passed ? 'READY' : `BLOCKED (${gate.blocked_by.length})` }}
    </div>
    <div class="eligibility">Eligibility: {{ (gate.eligibility_pct * 100).toFixed(0) }}%</div>
    <div class="bar"><div class="bar-fill" :style="{ width: (gate.eligibility_pct * 100) + '%' }"></div></div>
    <ul v-if="gate.blocked_by.length">
      <li v-for="b in gate.blocked_by" :key="b">{{ b }}</li>
    </ul>
    <div class="metrics" v-if="gate.metrics">
      <div>Trades: {{ gate.metrics.n_trades }} | WR {{ ((gate.metrics.winrate || 0) * 100).toFixed(1) }}%</div>
      <div>Sharpe {{ (gate.metrics.sharpe || 0).toFixed(2) }} | DD {{ ((gate.metrics.max_dd || 0) * 100).toFixed(1) }}%</div>
      <div>Liquid {{ gate.metrics.n_liquid || 0 }} | Autonomy L{{ gate.metrics.autonomy_level || 0 }}</div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'

interface Gate {
  passed: boolean
  eligibility_pct: number
  blocked_by: string[]
  metrics: Record<string, number>
}

const gate = ref<Gate>({
  passed: false, eligibility_pct: 0, blocked_by: [], metrics: {},
})
let timer: ReturnType<typeof setInterval> | null = null

async function refresh() {
  try {
    const r = await fetch('/api/v1/ai/promotion_gate')
    if (r.ok) gate.value = await r.json()
  } catch (e) {
    console.error('[PromotionGate] fetch failed', e)
  }
}

onMounted(() => {
  refresh()
  timer = setInterval(refresh, 60000)
})
onUnmounted(() => {
  if (timer) clearInterval(timer)
})
</script>

<style scoped>
.promotion-gate { padding: 12px; background: #0a0a0a; color: #0f0; font-family: monospace; }
.status.pass { color: #0f0; font-size: 1.5em; }
.status.block { color: #f55; font-size: 1.5em; }
.bar { width: 100%; height: 8px; background: #222; margin: 6px 0; }
.bar-fill { height: 8px; background: linear-gradient(to right, #f55, #ff0, #0f0); }
ul { color: #f55; }
</style>
