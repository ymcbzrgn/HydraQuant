<template>
  <div class="p-4 bg-white dark:bg-gray-800 rounded-lg shadow-sm border dark:border-gray-700 flex flex-col items-center justify-center">
    <h3 class="text-sm text-gray-600 dark:text-gray-400 font-semibold mb-2">MADAM Confidence</h3>
    <Knob v-model="displayValue" readonly :size="100" :valueColor="currentColor" valueTemplate="{value}%" />
    <div class="mt-2 text-xs text-center font-medium" :style="{ color: currentColor }">
      {{ confidenceLabel }}
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import Knob from 'primevue/knob';

const props = defineProps<{
  confidence: number // 0.0 to 1.0
}>();

const displayValue = computed(() => {
  return Math.round(props.confidence * 100);
});

const currentColor = computed(() => {
  const val = props.confidence;
  if (val > 0.85) return '#22c55e'; // green-500
  if (val > 0.70) return '#4ade80'; // green-400
  if (val > 0.50) return '#eab308'; // yellow-500
  if (val > 0.35) return '#f97316'; // orange-500
  return '#ef4444'; // red-500
});

const confidenceLabel = computed(() => {
  const val = props.confidence;
  if (val > 0.85) return 'Very High Conviction';
  if (val > 0.70) return 'High Conviction';
  if (val > 0.50) return 'Moderate Conviction';
  if (val > 0.35) return 'Low Conviction';
  return 'Uncertain / No Trade';
});
</script>
