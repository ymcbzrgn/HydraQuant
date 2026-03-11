<template>
  <div class="card p-4">
    <div class="flex justify-between items-center mb-4">
      <h2 class="text-xl font-bold flex items-center gap-2">
        <i class="pi pi-shield"></i> Risk Oversight
      </h2>
      <Button icon="pi pi-refresh" @click="refreshData" :loading="aiStore.loading" class="p-button-sm p-button-text p-button-rounded" />
    </div>

    <Message v-if="aiStore.error" severity="error" :closable="false">{{ aiStore.error }}</Message>

    <div v-if="aiStore.risk" class="flex flex-col gap-6">
      <!-- Budget Overview -->
      <div class="grid grid-cols-2 gap-4">
        <div class="bg-gray-50 dark:bg-gray-800 p-4 rounded-lg border dark:border-gray-700">
          <div class="text-sm text-gray-500 mb-1">Daily VaR Budget</div>
          <div class="text-2xl font-bold">${{ aiStore.risk.daily_budget.toFixed(2) }}</div>
        </div>
        <div class="bg-gray-50 dark:bg-gray-800 p-4 rounded-lg border dark:border-gray-700">
          <div class="text-sm text-gray-500 mb-1">Active Positions</div>
          <div class="text-2xl font-bold text-blue-500">{{ aiStore.risk.active_positions }}</div>
        </div>
      </div>

      <!-- Utilization Bar -->
      <div>
        <div class="flex justify-between text-sm mb-2">
          <span class="font-medium text-gray-600 dark:text-gray-300">Budget Consumed</span>
          <span class="font-bold" :class="getColorText(aiStore.risk.utilization_pct)">
            ${{ aiStore.risk.consumed.toFixed(2) }} ({{ aiStore.risk.utilization_pct.toFixed(1) }}%)
          </span>
        </div>
        <ProgressBar :value="aiStore.risk.utilization_pct" :showValue="false" style="height: 12px;" :class="getColorBg(aiStore.risk.utilization_pct)" />
        <div v-if="aiStore.risk.utilization_pct >= 100" class="mt-2 text-xs text-red-500 font-bold flex items-center gap-1">
          <i class="pi pi-exclamation-triangle"></i> Daily Risk Budget Exceeded! New entries paused.
        </div>
      </div>

      <!-- 7-Day History Chart (Placeholder until primevue/chart.js is installed, or simpler CSS bars) -->
      <div class="mt-2">
        <h3 class="text-sm font-semibold mb-3 text-gray-500">7-Day Utilization Trend</h3>
        <div class="flex items-end gap-2 h-24 border-b border-l p-2 pt-0 dark:border-gray-700">
          <div v-for="(day, i) in riskHistoryMocks" :key="i" class="w-full flex flex-col justify-end items-center group relative cursor-pointer">
            <div class="w-full rounded-t-sm transition-all duration-300" :class="getColorBg(day.pct)" :style="{ height: `${Math.min(day.pct, 100)}%` }"></div>
            <div class="opacity-0 group-hover:opacity-100 absolute -top-8 bg-gray-800 text-white text-xs p-1 rounded z-10 whitespace-nowrap">
              {{ day.date }}: {{ day.pct.toFixed(0) }}%
            </div>
            <div class="text-[10px] text-gray-400 mt-1">{{ day.dayLabel }}</div>
          </div>
        </div>
      </div>

    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue';
import { useAiStore } from '@/stores/aiStore';

import Button from 'primevue/button';
import Message from 'primevue/message';
import ProgressBar from 'primevue/progressbar';

const aiStore = useAiStore();

// Mocking 7-day history since the backend /api/ai/risk endpoint doesn't return full historical arrays yet 
// (It returns single day parameters. We will map this visually as a CSS bar chart).
const riskHistoryMocks = ref([
  { date: 'Mon', dayLabel: 'M', pct: 45 },
  { date: 'Tue', dayLabel: 'T', pct: 60 },
  { date: 'Wed', dayLabel: 'W', pct: 30 },
  { date: 'Thu', dayLabel: 'T', pct: 85 },
  { date: 'Fri', dayLabel: 'F', pct: 15 },
  { date: 'Sat', dayLabel: 'S', pct: 95 },
  { date: 'Sun', dayLabel: 'S', pct: 20 },
]);

const refreshData = async () => {
  await aiStore.fetchRisk();
};

onMounted(async () => {
  await refreshData();
});

const getColorText = (pct: number) => {
  if (pct >= 100) return 'text-red-500';
  if (pct >= 75) return 'text-orange-500';
  if (pct >= 50) return 'text-yellow-500';
  return 'text-green-500';
};

const getColorBg = (pct: number) => {
  if (pct >= 100) return 'bg-red-500';
  if (pct >= 75) return 'bg-orange-500';
  if (pct >= 50) return 'bg-yellow-500';
  return 'bg-green-500';
};
</script>
