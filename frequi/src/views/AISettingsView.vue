<template>
  <div class="p-3 sm:p-4 md:p-6 max-w-7xl mx-auto space-y-4 sm:space-y-6">
    <div class="flex items-center justify-between">
      <h1 class="text-2xl font-bold flex items-center gap-2">
        <i class="pi pi-cog text-primary"></i> AI Settings
      </h1>
      <Tag severity="info" value="Read-Only" />
    </div>

    <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">

      <!-- Left Column: Autonomy + Config -->
      <div class="lg:col-span-2 space-y-6">
        <AutonomyLevel />

        <!-- Configuration Parameters -->
        <div class="p-4 bg-white dark:bg-gray-800 rounded-lg shadow-sm border dark:border-gray-700">
          <h2 class="text-lg font-bold mb-4">Configuration</h2>
          <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div class="flex flex-col gap-2">
              <label class="text-sm font-semibold text-gray-600 dark:text-gray-300">Daily VaR Budget</label>
              <InputNumber v-model="configBudget" inputId="currency-us" mode="currency"
                           currency="USD" locale="en-US" disabled />
              <small class="text-gray-500">{{ configVarPct }}% of portfolio (auto-calculated)</small>
            </div>

            <div class="flex flex-col gap-2">
              <label class="text-sm font-semibold text-gray-600 dark:text-gray-300">Semantic Cache TTL</label>
              <div class="p-inputgroup">
                <InputNumber v-model="configTTL" disabled />
                <span class="p-inputgroup-addon">sec</span>
              </div>
              <small class="text-gray-500">Cache duration for duplicate queries</small>
            </div>

            <div class="flex flex-col gap-2">
              <label class="text-sm font-semibold text-gray-600 dark:text-gray-300">Confidence Exponent</label>
              <InputNumber v-model="configExponent" :minFractionDigits="1" :maxFractionDigits="1" disabled />
              <small class="text-gray-500">Kelly fraction curve (base 2.0)</small>
            </div>

            <div class="flex flex-col gap-2">
              <label class="text-sm font-semibold text-gray-600 dark:text-gray-300">RAG Chunk Overlap</label>
              <InputNumber v-model="configOverlap" disabled />
              <small class="text-gray-500">Token overlap between document chunks</small>
            </div>
          </div>
        </div>
      </div>

      <!-- Right Column: Models -->
      <div class="space-y-6">
        <div class="p-4 bg-white dark:bg-gray-800 rounded-lg shadow-sm border dark:border-gray-700">
          <h2 class="text-lg font-bold mb-4">Model Providers</h2>
          <div class="flex flex-col gap-4">
            <div class="flex items-center justify-between border-b pb-2 dark:border-gray-700">
              <div class="flex flex-col">
                <span class="font-bold text-sm">Primary</span>
                <span class="text-xs text-gray-500">Google Gemini</span>
              </div>
              <Tag severity="success" :value="aiStore.status?.active_model || 'gemini-2.5-flash'" />
            </div>
            <div class="flex items-center justify-between border-b pb-2 dark:border-gray-700">
              <div class="flex flex-col">
                <span class="font-bold text-sm">Fallback</span>
                <span class="text-xs text-gray-500">Groq Engine</span>
              </div>
              <Tag severity="warn" value="llama-3-70b" />
            </div>
            <div class="flex items-center justify-between">
              <div class="flex flex-col">
                <span class="font-bold text-sm">Deep Eval</span>
                <span class="text-xs text-gray-500">OpenRouter</span>
              </div>
              <Tag severity="info" value="claude-3-haiku" />
            </div>
          </div>
        </div>

        <div class="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
          <div class="flex items-start gap-3">
            <i class="pi pi-info-circle text-blue-500 mt-1"></i>
            <div class="text-sm text-blue-700 dark:text-blue-300">
              Settings are read from <code>ai_config.py</code> on the server.
              Edit that file directly to change configuration.
            </div>
          </div>
        </div>
      </div>

    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue';
import { useAiStore } from '@/stores/aiStore';
import AutonomyLevel from '@/components/ai/AutonomyLevel.vue';

import Tag from 'primevue/tag';
import InputNumber from 'primevue/inputnumber';

const aiStore = useAiStore();

const configBudget = ref(100.0);
const configVarPct = ref('1.0');
const configTTL = ref(300);
const configExponent = ref(2.0);
const configOverlap = ref(100);

onMounted(async () => {
  await aiStore.fetchSettings();
  if (aiStore.settings) {
    configBudget.value = aiStore.settings.daily_budget;
    configVarPct.value = (aiStore.settings.daily_var_pct * 100).toFixed(1);
    configTTL.value = aiStore.settings.semantic_cache_ttl;
    configExponent.value = aiStore.settings.confidence_exponent;
    configOverlap.value = aiStore.settings.rag_chunk_overlap;
  }
});
</script>
