<template>
  <div class="p-6 max-w-[1600px] mx-auto space-y-6">
    <div class="flex items-center justify-between">
      <h1 class="text-3xl font-bold flex items-center gap-3">
        <i class="pi pi-chart-bar text-primary"></i> AI Analytics Engine
      </h1>
      <div class="text-sm text-gray-500">Live Telemetry Diagnostics</div>
    </div>

    <!-- Top Row: Diagnostics -->
    <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
      <div class="lg:col-span-1 border dark:border-gray-700 rounded-lg overflow-hidden">
        <ModelStatusCard />
      </div>
      <div class="lg:col-span-2 border dark:border-gray-700 rounded-lg overflow-hidden">
        <ForgonePnLTracker />
      </div>
    </div>

    <!-- Middle Row: Signal Core -->
    <div class="border dark:border-gray-700 rounded-lg shadow-sm">
      <AISignalPanel />
    </div>

    <!-- Bottom Row: Calibration Display (Mocked until historical endpoints resolve arrays) -->
    <div class="card p-4 bg-white dark:bg-gray-800 rounded-lg shadow-sm border dark:border-gray-700">
      <div class="flex justify-between items-center mb-4">
        <h2 class="text-xl font-bold">Confidence Calibration Trend</h2>
        <Tag severity="success" value="Highly Calibrated" />
      </div>
      
      <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
        <div class="p-3 bg-gray-50 dark:bg-gray-900 rounded border dark:border-gray-700">
          <div class="text-xs text-gray-500 mb-1">Brier Score</div>
          <div class="text-xl font-bold text-green-500">0.142</div>
        </div>
        <div class="p-3 bg-gray-50 dark:bg-gray-900 rounded border dark:border-gray-700">
          <div class="text-xs text-gray-500 mb-1">Average Prediction Conviction</div>
          <div class="text-xl font-bold">68.5%</div>
        </div>
        <div class="p-3 bg-gray-50 dark:bg-gray-900 rounded border dark:border-gray-700">
          <div class="text-xs text-gray-500 mb-1">30-Day Signal Accuracy</div>
          <div class="text-xl font-bold text-blue-500">71.2%</div>
        </div>
      </div>

      <div class="h-[200px] w-full flex items-end gap-1 px-4 py-8 border-b border-l text-xs text-gray-500 dark:border-gray-600 relative">
        <div class="absolute -left-8 top-0">1.0</div>
        <div class="absolute -left-8 bottom-0">0.0</div>
        <!-- Simulated Calibration line representation -->
        <div v-for="(val, index) in mockCalibration" :key="index" 
             class="flex-1 bg-primary/20 hover:bg-primary transition-colors cursor-pointer rounded-t"
             :style="{ height: `${val * 100}%` }"
             :title="`Score: ${val.toFixed(2)}`">
        </div>
      </div>
      <div class="text-center text-xs text-gray-400 mt-2">Days (Last 30)</div>

    </div>
    
  </div>
</template>

<script setup lang="ts">
import ModelStatusCard from '@/components/ai/ModelStatusCard.vue';
import ForgonePnLTracker from '@/components/ai/ForgonePnLTracker.vue';
import AISignalPanel from '@/components/ai/AISignalPanel.vue';

import Tag from 'primevue/tag';

// Generating a pseudo-random stable curve to simulate the 30 day history calibration
const mockCalibration = Array.from({ length: 30 }, (_, i) => {
  return 0.6 + (Math.sin(i * 0.5) * 0.15) + (Math.random() * 0.1);
});
</script>
