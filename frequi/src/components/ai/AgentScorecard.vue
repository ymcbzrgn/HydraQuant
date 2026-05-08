<template>
  <div class="agent-scorecard">
    <h3>Agent Scorecard (EarnedTrust)</h3>
    <table>
      <thead>
        <tr><th>Agent</th><th>Trust</th><th>Decisions</th><th>Win-rate</th><th>Recovery</th></tr>
      </thead>
      <tbody>
        <tr v-for="a in agents" :key="a.name">
          <td>{{ a.name }}</td>
          <td :class="trustClass(a.trust)">{{ (a.trust * 100).toFixed(0) }}%</td>
          <td>{{ a.n_decisions }}</td>
          <td>{{ (a.winrate * 100).toFixed(1) }}%</td>
          <td>{{ (a.recovery_rate * 100).toFixed(1) }}%</td>
        </tr>
      </tbody>
    </table>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'

interface AgentScore {
  name: string
  trust: number
  n_decisions: number
  winrate: number
  recovery_rate: number
}

const agents = ref<AgentScore[]>([])
let timer: ReturnType<typeof setInterval> | null = null

function trustClass(t: number) {
  if (t < 0.4) return 'low'
  if (t > 0.7) return 'high'
  return 'mid'
}

async function refresh() {
  try {
    const r = await fetch('/api/v1/ai/agents/scorecard')
    if (r.ok) agents.value = await r.json()
  } catch (e) {
    console.error('[AgentScorecard] fetch failed', e)
  }
}

onMounted(() => {
  refresh()
  timer = setInterval(refresh, 30000)
})
onUnmounted(() => {
  if (timer) clearInterval(timer)
})
</script>

<style scoped>
.agent-scorecard { padding: 12px; background: #0a0a0a; color: #0f0; font-family: monospace; }
table { width: 100%; border-collapse: collapse; }
th, td { padding: 4px 8px; border: 1px solid #0a4; text-align: left; }
.low { color: #f55; } .high { color: #0f0; } .mid { color: #ff0; }
</style>
