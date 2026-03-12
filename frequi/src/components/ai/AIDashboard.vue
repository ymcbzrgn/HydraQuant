<template>
  <div class="ai-dashboard p-4">
    <div class="flex items-center justify-between mb-4">
      <h2 class="text-2xl font-bold">AI Trading Dashboard</h2>
      <div class="flex items-center gap-3">
        <span v-if="aiStore.lastFetchTime" class="text-xs text-gray-400">
          Updated: {{ aiStore.lastFetchTime?.toLocaleTimeString() }}
        </span>
        <span
          class="px-2 py-1 rounded text-xs font-bold"
          :class="onlineClass"
        >
          {{ aiStore.isAiOnline ? 'AI ONLINE' : 'AI OFFLINE' }}
        </span>
      </div>
    </div>

    <!-- Portfolio Overview -->
    <div class="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
      <div class="p-4 rounded-lg border bg-blue-50 border-blue-200">
        <div class="text-sm text-blue-600">Portfolio Value</div>
        <div class="text-2xl font-bold">{{ aiStore.portfolioValueFormatted }}</div>
        <div v-if="aiStore.portfolio?.assets" class="text-xs text-blue-500 mt-1">
          {{ Object.keys(aiStore.portfolio.assets).length }} asset(s) |
          {{ aiStore.portfolio?.stake_currency || 'USDT' }}
        </div>
      </div>
      <div class="p-4 rounded-lg border bg-purple-50 border-purple-200">
        <div class="text-sm text-purple-600">Active Model</div>
        <div class="text-xl font-bold">{{ aiStore.status?.active_model || 'awaiting...' }}</div>
        <div class="text-xs text-purple-500 mt-1">
          Autonomy: L{{ aiStore.autonomy?.current_level ?? '?' }} |
          Cost: ${{ aiStore.costSummary?.today_cost?.toFixed(4) || '0' }}
        </div>
      </div>
    </div>

    <!-- Health Status -->
    <div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
      <div class="p-4 rounded-lg border" :class="healthClass">
        <div class="text-sm text-gray-500">System Status</div>
        <div class="text-xl font-bold uppercase">{{ aiStore.health?.status || 'unknown' }}</div>
        <div v-if="aiStore.health?.alerts?.length" class="text-xs text-red-500 mt-1">
          {{ aiStore.health.alerts.length }} alert(s)
        </div>
      </div>

      <div class="p-4 rounded-lg border bg-white dark:bg-gray-800">
        <div class="text-sm text-gray-500">LLM Cost Today</div>
        <div class="text-xl font-bold">${{ aiStore.metrics?.llm_cost_today?.toFixed(4) || '0.0000' }}</div>
      </div>

      <div class="p-4 rounded-lg border bg-white dark:bg-gray-800">
        <div class="text-sm text-gray-500">Cache Hit Rate</div>
        <div class="text-xl font-bold">{{ ((aiStore.metrics?.cache_hit_rate || 0) * 100).toFixed(1) }}%</div>
      </div>

      <div class="p-4 rounded-lg border bg-white dark:bg-gray-800">
        <div class="text-sm text-gray-500">Decisions (24h)</div>
        <div class="text-xl font-bold">{{ aiStore.metrics?.total_decisions || 0 }}</div>
      </div>
    </div>

    <!-- Second Row: Autonomy + Risk + Error Rate -->
    <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
      <div class="p-4 rounded-lg border bg-white dark:bg-gray-800">
        <div class="text-sm text-gray-500">Autonomy Level</div>
        <div class="text-xl font-bold">L{{ aiStore.autonomy?.current_level ?? '?' }}</div>
        <div class="text-xs text-gray-400">Kelly: {{ aiStore.autonomy?.kelly_fraction?.toFixed(3) || '-' }}</div>
      </div>

      <div class="p-4 rounded-lg border bg-white dark:bg-gray-800">
        <div class="text-sm text-gray-500">Risk Budget Used</div>
        <div class="text-xl font-bold">{{ aiStore.risk?.utilization_pct?.toFixed(1) || '0' }}%</div>
        <div class="text-xs text-gray-400">
          ${{ aiStore.risk?.consumed?.toFixed(2) || '0' }} / ${{ aiStore.risk?.daily_budget?.toFixed(2) || '0' }}
        </div>
      </div>

      <div class="p-4 rounded-lg border bg-white dark:bg-gray-800">
        <div class="text-sm text-gray-500">Error Rate (24h)</div>
        <div class="text-xl font-bold">{{ ((aiStore.metrics?.error_rate || 0) * 100).toFixed(1) }}%</div>
        <div class="text-xs text-gray-400">{{ aiStore.metrics?.retrieval_count || 0 }} retrievals</div>
      </div>
    </div>

    <!-- Forgone P&L -->
    <div v-if="aiStore.forgonePnl" class="p-4 rounded-lg border mb-6" :class="forgoneClass">
      <div class="flex justify-between items-center">
        <div>
          <div class="text-sm text-gray-500">Forgone P&L (missed trades)</div>
          <div class="text-xl font-bold">${{ aiStore.forgonePnl?.total_forgone?.toFixed(2) || '0' }}</div>
        </div>
        <div class="text-xs text-gray-400">{{ aiStore.forgonePnl?.recent_signals || 0 }} signals tracked</div>
      </div>
    </div>

    <!-- Recent Signals -->
    <div class="mb-6">
      <h3 class="text-lg font-semibold mb-2">Recent Signals</h3>
      <div class="border rounded-lg overflow-hidden">
        <table class="w-full text-sm">
          <thead class="bg-gray-100 dark:bg-gray-700">
            <tr>
              <th class="p-2 text-left">Pair</th>
              <th class="p-2 text-left">Signal</th>
              <th class="p-2 text-left">Confidence</th>
              <th class="p-2 text-left">Outcome</th>
              <th class="p-2 text-left">Time</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(sig, idx) in recentSignals" :key="idx" class="border-t">
              <td class="p-2 font-mono">{{ sig.pair }}</td>
              <td class="p-2">
                <span :class="signalClass(sig.signal)" class="px-2 py-0.5 rounded text-xs font-bold">
                  {{ sig.signal }}
                </span>
              </td>
              <td class="p-2">{{ (sig.confidence * 100).toFixed(0) }}%</td>
              <td class="p-2">{{ sig.outcome || 'Pending' }}</td>
              <td class="p-2 text-gray-500">{{ formatTime(sig.timestamp) }}</td>
            </tr>
            <tr v-if="!recentSignals.length">
              <td colspan="5" class="p-4 text-center text-gray-400">No signals yet</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- Quick Links -->
    <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
      <router-link to="/ai/analytics" class="p-4 rounded-lg border bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700 text-center">
        Analytics & Signals
      </router-link>
      <router-link to="/ai/risk" class="p-4 rounded-lg border bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700 text-center">
        Risk Dashboard
      </router-link>
      <router-link to="/ai/settings" class="p-4 rounded-lg border bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700 text-center">
        AI Settings
      </router-link>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import { useAiStore } from '@/stores/aiStore';

const aiStore = useAiStore();

const onlineClass = computed(() =>
  aiStore.isAiOnline ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800',
);

const healthClass = computed(() => {
  const status = aiStore.health?.status;
  if (status === 'healthy') return 'bg-green-50 border-green-300';
  if (status === 'degraded') return 'bg-yellow-50 border-yellow-300';
  if (status === 'critical') return 'bg-red-50 border-red-300';
  return 'bg-gray-50 border-gray-300';
});

const forgoneClass = computed(() => {
  const forgone = aiStore.forgonePnl?.total_forgone || 0;
  if (forgone > 100) return 'bg-red-50 border-red-300';
  if (forgone > 20) return 'bg-yellow-50 border-yellow-300';
  return 'bg-white dark:bg-gray-800';
});

const recentSignals = computed(() => (aiStore.signals || []).slice(0, 15));

function signalClass(signal: string) {
  if (signal === 'BULLISH') return 'bg-green-100 text-green-800';
  if (signal === 'BEARISH') return 'bg-red-100 text-red-800';
  return 'bg-gray-100 text-gray-800';
}

function formatTime(ts: string) {
  if (!ts) return '-';
  try {
    return new Date(ts).toLocaleString();
  } catch {
    return ts;
  }
}
</script>
