<template>
  <div class="card p-4">
    <div class="flex justify-between items-center mb-4">
      <h2 class="text-xl font-bold flex items-center gap-2">
        <i class="pi pi-briefcase"></i> Blocked Signals Impact
      </h2>
      <Button icon="pi pi-refresh" @click="refreshData" :loading="aiStore.loading" class="p-button-sm p-button-text p-button-rounded" />
    </div>

    <Message v-if="aiStore.error" severity="error" :closable="false">{{ aiStore.error }}</Message>

    <div v-if="aiStore.forgonePnl" class="flex flex-col gap-6">
      <!-- Total Forgone Overview -->
      <div class="flex items-center gap-4 bg-gray-50 dark:bg-gray-800 p-4 rounded-lg border dark:border-gray-700">
        <div class="text-4xl font-black" :class="getForgoneColor(aiStore.forgonePnl.total_forgone)">
          {{ aiStore.forgonePnl.total_forgone > 0 ? '+' : '' }}{{ aiStore.forgonePnl.total_forgone.toFixed(2) }}%
        </div>
        <div>
          <div class="font-bold text-lg">Total Theoretical P&L</div>
          <div class="text-sm text-gray-600 dark:text-gray-400">From {{ aiStore.forgonePnl.recent_signals }} rejected signals</div>
        </div>
      </div>

      <!-- Value Destruction Warning -->
      <Message v-if="aiStore.forgonePnl.total_forgone > 0" severity="warn" :closable="false" icon="pi pi-exclamation-triangle">
        <strong>Guardrail Warning:</strong> The AI model's rejected trades would have generated positive returns. Consider elevating the Autonomy Level or relaxing Risk Budgets.
      </Message>

      <!-- Weekly Summary -->
      <div>
        <h3 class="text-sm font-semibold mb-3 text-gray-600 dark:text-gray-400">Weekly Breakdown</h3>
        <div class="grid grid-cols-2 lg:grid-cols-4 gap-2">
           <div v-for="(val, week) in aiStore.forgonePnl.weekly_summary" :key="week" 
                class="p-2 border rounded text-center dark:border-gray-700 bg-white dark:bg-gray-800 shadow-sm"
                :class="{'border-green-500/30': val > 0, 'border-red-500/30': val < 0}">
             <div class="text-[10px] text-gray-400 font-mono">{{ week }}</div>
             <div class="font-bold text-sm" :class="getForgoneColor(val)">
               {{ val > 0 ? '+' : '' }}{{ val.toFixed(2) }}%
             </div>
           </div>
        </div>
      </div>

    </div>
    <div v-else class="text-gray-400 italic text-sm py-4">
      No forgone signals tracked yet.
    </div>
  </div>
</template>

<script setup lang="ts">
import { onMounted } from 'vue';
import { useAiStore } from '@/stores/aiStore';

import Button from 'primevue/button';
import Message from 'primevue/message';

const aiStore = useAiStore();

const refreshData = async () => {
  await aiStore.fetchForgonePnl();
};

onMounted(async () => {
  await refreshData();
});

const getForgoneColor = (val: number) => {
  // Positive means the guardrails DESTROYED value.
  if (val > 2) return 'text-red-500';
  if (val > 0) return 'text-orange-500';
  if (val < -2) return 'text-green-500'; // Negative means guardrails SAVED us from losses.
  return 'text-green-400';
};
</script>
