<template>
  <div class="p-3 sm:p-4 md:p-6 max-w-7xl mx-auto space-y-4 sm:space-y-6">
    <div class="flex items-center justify-between">
      <h1 class="text-2xl font-bold flex items-center gap-2">
        <i class="pi pi-shield text-primary"></i> Risk Oversight
      </h1>
      <Tag severity="danger" value="Live Constraints" />
    </div>

    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">

      <!-- Left: Risk + Autonomy -->
      <div class="space-y-6">
        <RiskPanel />
        <AutonomyLevel />
      </div>

      <!-- Right: Positions + Portfolio -->
      <div class="space-y-6">
        <!-- Active Positions -->
        <div class="p-4 bg-white dark:bg-gray-800 rounded-lg shadow-sm border dark:border-gray-700">
          <h2 class="text-lg font-bold mb-4">Active Positions</h2>

          <div v-if="activePositions > 0" class="text-center py-4">
            <div class="text-4xl font-black text-primary">{{ activePositions }}</div>
            <div class="text-sm text-gray-500 mt-1">open positions (last 24h signals)</div>
          </div>
          <div v-else class="text-center py-6 text-gray-400 text-sm">
            No active positions. AI pipeline is waiting for signals.
          </div>

          <!-- Portfolio breakdown if available -->
          <div v-if="portfolioAssets.length > 0" class="mt-4 border-t dark:border-gray-700 pt-4">
            <h3 class="text-sm font-semibold text-gray-500 mb-3">Portfolio Assets</h3>
            <div class="space-y-2">
              <div v-for="asset in portfolioAssets" :key="asset.currency"
                   class="flex items-center justify-between p-2 border dark:border-gray-700 bg-gray-50 dark:bg-gray-900 rounded text-sm">
                <span class="font-bold font-mono">{{ asset.currency }}</span>
                <span class="text-gray-500">${{ asset.usd.toFixed(2) }}</span>
              </div>
            </div>
          </div>
        </div>

        <!-- Risk Summary -->
        <div class="p-4 bg-white dark:bg-gray-800 rounded-lg shadow-sm border dark:border-gray-700">
          <h2 class="text-lg font-bold mb-4">Risk Summary</h2>
          <div class="grid grid-cols-1 sm:grid-cols-2 gap-4 text-sm">
            <div>
              <div class="text-gray-500">Portfolio Value</div>
              <div class="font-bold font-mono">${{ (aiStore.risk?.portfolio_value || 0).toLocaleString() }}</div>
            </div>
            <div>
              <div class="text-gray-500">Daily VaR Budget</div>
              <div class="font-bold font-mono">${{ (aiStore.risk?.daily_budget || 0).toFixed(2) }}</div>
            </div>
            <div>
              <div class="text-gray-500">Budget Consumed</div>
              <div class="font-bold font-mono">${{ (aiStore.risk?.consumed || 0).toFixed(2) }}</div>
            </div>
            <div>
              <div class="text-gray-500">Error Rate (24h)</div>
              <div class="font-bold font-mono">{{ ((aiStore.metrics?.error_rate || 0) * 100).toFixed(1) }}%</div>
            </div>
          </div>
        </div>
      </div>

    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted } from 'vue';
import { useAiStore } from '@/stores/aiStore';

import RiskPanel from '@/components/ai/RiskPanel.vue';
import AutonomyLevel from '@/components/ai/AutonomyLevel.vue';
import Tag from 'primevue/tag';

const aiStore = useAiStore();

onMounted(async () => {
  await Promise.all([
    aiStore.fetchRisk(),
    aiStore.fetchAutonomy(),
    aiStore.fetchPortfolio(),
    aiStore.fetchMetrics(),
  ]);
});

const activePositions = computed(() => aiStore.risk?.active_positions || 0);

const portfolioAssets = computed(() => {
  const assets = aiStore.portfolio?.assets || {};
  return Object.entries(assets)
    .filter(([, info]) => {
      if (typeof info === 'object' && info !== null && 'usd' in info) {
        return (info as any).usd >= 1.0;
      }
      return false;
    })
    .map(([currency, info]) => ({
      currency,
      usd: (info as any).usd as number,
    }))
    .sort((a, b) => b.usd - a.usd);
});
</script>
