<template>
  <div class="card p-4">
    <div class="flex justify-between items-center mb-4">
      <h2 class="text-xl font-bold">Autonomy Level</h2>
      <Button icon="pi pi-refresh" @click="refreshData" :loading="aiStore.loading" class="p-button-sm p-button-text p-button-rounded" />
    </div>

    <Message v-if="aiStore.error" severity="error" :closable="false">{{ aiStore.error }}</Message>

    <div v-if="aiStore.autonomy" class="flex flex-col gap-6">
      <!-- Current Level Display -->
      <div class="flex items-center gap-4 bg-gray-50 dark:bg-gray-800 p-4 rounded-lg border dark:border-gray-700">
        <div class="text-4xl font-black text-primary">L{{ aiStore.autonomy.current_level }}</div>
        <div>
          <div class="font-bold text-lg">{{ getLevelName(aiStore.autonomy.current_level) }}</div>
          <div class="text-sm text-gray-500">Max Kelly Fraction: {{ (aiStore.autonomy.kelly_fraction * 100).toFixed(1) }}%</div>
        </div>
      </div>

      <!-- Level Tracker -->
      <div>
        <h3 class="text-sm font-semibold mb-3 text-gray-500">Capability Stages</h3>
        <Steps :model="levelSteps" :activeIndex="aiStore.autonomy.current_level" class="text-xs" />
      </div>

      <!-- Criteria Requirements -->
      <div v-if="aiStore.autonomy.criteria" class="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg">
        <h3 class="text-sm font-semibold mb-2 flex items-center gap-2">
          <i class="pi pi-info-circle"></i> Promotion Criteria
        </h3>
        <div class="grid grid-cols-2 gap-4 text-sm mt-2">
          <div>
            <div class="text-gray-500">Min Win Rate</div>
            <div class="font-medium font-mono text-green-600 dark:text-green-400">
              {{ (aiStore.autonomy.criteria.min_win_rate * 100).toFixed(0) }}%
            </div>
          </div>
          <div>
            <div class="text-gray-500">Min Trades</div>
            <div class="font-medium font-mono text-blue-600 dark:text-blue-400">
              {{ aiStore.autonomy.criteria.min_trades }}
            </div>
          </div>
          <div>
            <div class="text-gray-500">Max Drawdown</div>
            <div class="font-medium font-mono text-red-600 dark:text-red-400">
              {{ (aiStore.autonomy.criteria.max_drawdown * 100).toFixed(1) }}%
            </div>
          </div>
          <div>
            <div class="text-gray-500">Min Sharpe</div>
            <div class="font-medium font-mono text-purple-600 dark:text-purple-400">
              {{ aiStore.autonomy.criteria.min_sharpe.toFixed(2) }}
            </div>
          </div>
        </div>
      </div>

      <!-- History Timeline -->
      <div v-if="aiStore.autonomy.history && aiStore.autonomy.history.length > 0">
        <h3 class="text-sm font-semibold mb-3 text-gray-500">Recent Transitions</h3>
        <Timeline :value="recentHistory" layout="vertical" class="text-sm">
          <template #opposite="slotProps">
            <small class="text-gray-500">{{ formatDate(slotProps.item.timestamp) }}</small>
          </template>
          <template #marker="slotProps">
            <span class="flex w-6 h-6 items-center justify-center text-white rounded-full z-10 shadow-sm"
                  :class="slotProps.item.new_level > slotProps.item.old_level ? 'bg-green-500' : 'bg-red-500'">
              <i class="pi text-xs" :class="slotProps.item.new_level > slotProps.item.old_level ? 'pi-arrow-up' : 'pi-arrow-down'"></i>
            </span>
          </template>
          <template #content="slotProps">
            <div class="font-bold">L{{ slotProps.item.old_level }} <i class="pi pi-arrow-right text-xs mx-1"></i> L{{ slotProps.item.new_level }}</div>
            <div class="text-gray-500 text-xs mt-1">{{ slotProps.item.reason }}</div>
          </template>
        </Timeline>
      </div>
      <div v-else class="text-sm text-gray-400 italic text-center py-4">
        No level transitions recorded yet.
      </div>

    </div>
    
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue';
import { useAiStore } from '@/stores/aiStore';

import Button from 'primevue/button';
import Message from 'primevue/message';
import Steps from 'primevue/steps';
import Timeline from 'primevue/timeline';

const aiStore = useAiStore();

const refreshData = async () => {
  await aiStore.fetchAutonomy();
};

onMounted(async () => {
  await refreshData();
});

const getLevelName = (level: number) => {
  const names = [
    'Backtest Only',
    'Paper Trading',
    'Micro Live',
    'Standard Live',
    'High Conviction',
    'Full Autonomy'
  ];
  return names[level] || `Level ${level}`;
};

const levelSteps = ref([
  { label: 'L0' },
  { label: 'L1' },
  { label: 'L2' },
  { label: 'L3' },
  { label: 'L4' },
  { label: 'L5' }
]);

const recentHistory = computed(() => {
  if (!aiStore.autonomy || !aiStore.autonomy.history) return [];
  // Show only last 5
  return [...aiStore.autonomy.history].reverse().slice(0, 5);
});

const formatDate = (isoStr: string) => {
  if (!isoStr) return '';
  const d = new Date(isoStr);
  return `${d.toLocaleDateString()} ${d.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}`;
};

</script>
