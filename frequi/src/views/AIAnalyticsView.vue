<template>
  <div class="p-6 max-w-[1600px] mx-auto space-y-6">
    <div class="flex items-center justify-between">
      <h1 class="text-2xl font-bold flex items-center gap-2">
        <i class="pi pi-chart-bar text-primary"></i> AI Analytics
      </h1>
      <Button icon="pi pi-refresh" @click="refreshAll" :loading="aiStore.loading"
              class="p-button-sm p-button-outlined" label="Refresh" />
    </div>

    <!-- Top Row: Cost + Forgone P&L -->
    <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
      <div class="lg:col-span-1 border dark:border-gray-700 rounded-lg overflow-hidden">
        <ModelStatusCard />
      </div>
      <div class="lg:col-span-2 border dark:border-gray-700 rounded-lg overflow-hidden">
        <ForgonePnLTracker />
      </div>
    </div>

    <!-- Signal History -->
    <div class="border dark:border-gray-700 rounded-lg shadow-sm">
      <AISignalPanel />
    </div>

    <!-- Confidence Calibration -->
    <div class="p-4 bg-white dark:bg-gray-800 rounded-lg shadow-sm border dark:border-gray-700">
      <div class="flex justify-between items-center mb-4">
        <h2 class="text-lg font-bold">Confidence Calibration</h2>
        <Tag v-if="calibrationData.length > 0" severity="info"
             :value="`${calibrationData.length} data points`" />
        <Tag v-else severity="secondary" value="Awaiting data" />
      </div>

      <div v-if="calibrationData.length > 0"
           class="h-[160px] w-full flex items-end gap-1 px-4 py-4 border-b border-l dark:border-gray-600 text-xs text-gray-500">
        <div v-for="(val, index) in calibrationData" :key="index"
             class="flex-1 bg-primary/30 hover:bg-primary transition-colors cursor-pointer rounded-t"
             :style="{ height: `${Math.max(val.predicted_confidence * 100, 2)}%` }"
             :title="`Confidence: ${(val.predicted_confidence * 100).toFixed(0)}% | Outcome: ${val.actual_outcome ?? 'Pending'}`">
        </div>
      </div>
      <div v-else class="text-center text-gray-400 text-sm py-8">
        Calibration data will appear after the AI makes decisions and outcomes are recorded.
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted } from 'vue';
import { useAiStore } from '@/stores/aiStore';

import ModelStatusCard from '@/components/ai/ModelStatusCard.vue';
import ForgonePnLTracker from '@/components/ai/ForgonePnLTracker.vue';
import AISignalPanel from '@/components/ai/AISignalPanel.vue';
import Tag from 'primevue/tag';
import Button from 'primevue/button';

const aiStore = useAiStore();

onMounted(async () => {
  await Promise.all([
    aiStore.fetchConfidenceHistory(),
    aiStore.fetchCostSummary(),
    aiStore.fetchForgonePnl(),
  ]);
});

const refreshAll = async () => {
  await Promise.all([
    aiStore.fetchConfidenceHistory(),
    aiStore.fetchCostSummary(),
    aiStore.fetchForgonePnl(),
    aiStore.fetchSignals(50),
  ]);
};

const calibrationData = computed(() => aiStore.confidenceHistory || []);
</script>
