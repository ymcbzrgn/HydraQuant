<template>
  <div class="card p-4">
    <div class="flex justify-between items-center mb-4">
      <h2 class="text-xl font-bold flex items-center gap-2">
        <i class="pi pi-server"></i> LLM Status & Cost
      </h2>
      <Button icon="pi pi-refresh" @click="refreshData" :loading="aiStore.loading" class="p-button-sm p-button-text p-button-rounded" />
    </div>

    <Message v-if="aiStore.error" severity="error" :closable="false">{{ aiStore.error }}</Message>

    <div v-if="aiStore.costSummary" class="flex flex-col gap-6">
      
      <!-- General Summary -->
      <div class="grid grid-cols-2 gap-4">
        <div class="bg-gray-50 dark:bg-gray-800 p-4 rounded-lg border dark:border-gray-700">
          <div class="text-sm text-gray-600 dark:text-gray-400 mb-1">Today's Cost</div>
          <div class="text-2xl font-bold" :class="{'text-red-500': aiStore.costSummary.today_cost > 1.0, 'text-green-500': aiStore.costSummary.today_cost <= 1.0}">
            ${{ aiStore.costSummary.today_cost.toFixed(4) }}
          </div>
        </div>
        <div class="bg-gray-50 dark:bg-gray-800 p-4 rounded-lg border dark:border-gray-700">
          <div class="text-sm text-gray-600 dark:text-gray-400 mb-1">Budget Remaining</div>
          <div class="text-2xl font-bold text-blue-500">
            ${{ aiStore.costSummary.budget_remaining.toFixed(2) }}
          </div>
        </div>
      </div>

      <!-- Models List -->
      <div>
        <h3 class="text-sm font-semibold mb-3 text-gray-600 dark:text-gray-400">Active Providers</h3>
        <div class="space-y-3">
          <div v-for="(details, modelName) in aiStore.costSummary.models" :key="modelName" 
               class="flex items-center justify-between p-3 border dark:border-gray-700 rounded bg-white dark:bg-gray-800 shadow-sm">
            <div class="flex items-center gap-3">
              <i class="pi pi-code text-xl text-primary"></i>
              <div>
                <div class="font-bold text-sm">{{ modelName }}</div>
                <div class="text-xs text-gray-600 dark:text-gray-400">{{ details.calls || 0 }} Requests</div>
              </div>
            </div>
            <div class="text-right">
              <div class="font-bold text-sm text-gray-700 dark:text-gray-300">${{ (details.cost || 0).toFixed(4) }}</div>
              <div class="text-xs text-gray-500 dark:text-gray-400">{{ (details.tokens || 0).toLocaleString() }} Tokens</div>
            </div>
          </div>
        </div>
      </div>

      <!-- Semantic Cache Hit Rate -->
      <div v-if="aiStore.status && aiStore.status.cache_hit_rate !== undefined" class="pt-2 border-t dark:border-gray-700">
        <div class="flex justify-between items-center mb-1">
          <span class="text-sm font-semibold text-gray-600 dark:text-gray-400">Cache Hit Rate</span>
          <span class="text-sm font-bold text-green-500">{{ (aiStore.status.cache_hit_rate * 100).toFixed(1) }}%</span>
        </div>
        <ProgressBar :value="aiStore.status.cache_hit_rate * 100" :showValue="false" style="height: 6px;" class="bg-green-500" />
      </div>

    </div>
    <div v-else class="text-gray-400 italic text-sm py-4">
      No cost data tracked yet.
    </div>
  </div>
</template>

<script setup lang="ts">
import { onMounted } from 'vue';
import { useAiStore } from '@/stores/aiStore';

import Button from 'primevue/button';
import Message from 'primevue/message';
import ProgressBar from 'primevue/progressbar';

const aiStore = useAiStore();

const refreshData = async () => {
  await Promise.all([
    aiStore.fetchCostSummary(),
    aiStore.fetchStatus()
  ]);
};

onMounted(async () => {
  await refreshData();
});
</script>
