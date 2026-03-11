<template>
  <div class="p-6 max-w-7xl mx-auto space-y-6">
    <div class="flex items-center justify-between">
      <h1 class="text-3xl font-bold flex items-center gap-3">
        <i class="pi pi-cog text-primary"></i> AI Engine Settings
      </h1>
      <Tag severity="info" value="Read-Only Mode"></Tag>
    </div>

    <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
      
      <!-- Left Column: Autonomy & Core -->
      <div class="lg:col-span-2 space-y-6">
        <AutonomyLevel />

        <!-- Mock Settings Form (Read-Only Representation of ai_config.py) -->
        <div class="card p-4 bg-white dark:bg-gray-800 rounded-lg shadow-sm border dark:border-gray-700">
          <h2 class="text-xl font-bold mb-4">Configuration Parameters</h2>
          <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div class="flex flex-col gap-2">
              <label class="text-sm font-semibold text-gray-600 dark:text-gray-300">Daily Risk Budget Limit</label>
              <InputNumber v-model="mockSettings.budget" inputId="currency-us" mode="currency" currency="USD" locale="en-US" disabled />
              <small class="text-gray-500">Maximum USD loss allowed per day before AI pausing.</small>
            </div>
            
            <div class="flex flex-col gap-2">
              <label class="text-sm font-semibold text-gray-600 dark:text-gray-300">Semantic Cache TTL</label>
              <div class="p-inputgroup">
                <InputNumber v-model="mockSettings.ttl" disabled />
                <span class="p-inputgroup-addon">Seconds</span>
              </div>
              <small class="text-gray-500">Duration duplicate queries rely on SQLite bindings.</small>
            </div>

            <div class="flex flex-col gap-2">
              <label class="text-sm font-semibold text-gray-600 dark:text-gray-300">MADAM Confidence Exponent</label>
              <InputNumber v-model="mockSettings.exponent" :minFractionDigits="1" :maxFractionDigits="1" disabled />
              <small class="text-gray-500">Kelly fraction curve accelerator value (base 2.0).</small>
            </div>

            <div class="flex flex-col gap-2">
              <label class="text-sm font-semibold text-gray-600 dark:text-gray-300">RAG Chunk Overlap</label>
              <InputNumber v-model="mockSettings.overlap" disabled />
              <small class="text-gray-500">Contextual overlap between scraped document embeddings.</small>
            </div>
          </div>
        </div>
      </div>

      <!-- Right Column: Models & Preferences -->
      <div class="space-y-6">
        <div class="card p-4 bg-white dark:bg-gray-800 rounded-lg shadow-sm border dark:border-gray-700">
          <h2 class="text-xl font-bold mb-4">Model Preferences</h2>
          <div class="flex flex-col gap-4">
            
            <div class="flex items-center justify-between border-b pb-2 dark:border-gray-700">
              <div class="flex flex-col">
                <span class="font-bold text-sm">Primary LLM</span>
                <span class="text-xs text-gray-500">Google Gemini</span>
              </div>
              <Tag severity="success" value="gemini-2.5-flash"></Tag>
            </div>

            <div class="flex items-center justify-between border-b pb-2 dark:border-gray-700">
              <div class="flex flex-col">
                <span class="font-bold text-sm">Fallback LLM</span>
                <span class="text-xs text-gray-500">Groq Engine</span>
              </div>
              <Tag severity="warning" value="llama-3-70b"></Tag>
            </div>

            <div class="flex items-center justify-between">
              <div class="flex flex-col">
                <span class="font-bold text-sm">Deep Evaluation</span>
                <span class="text-xs text-gray-500">OpenRouter</span>
              </div>
              <Tag severity="info" value="claude-3-haiku"></Tag>
            </div>

          </div>
        </div>

        <div class="card p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
          <div class="flex items-start gap-3">
            <i class="pi pi-info-circle text-blue-500 mt-1"></i>
            <div class="text-sm text-blue-700 dark:text-blue-300">
              Settings editing is currently locked to the backend <code>ai_config.py</code> file. 
              The frontend is currently mapping environment telemetry natively via Read-Only access bridging the 8890 endpoint.
            </div>
          </div>
        </div>

      </div>

    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue';
import AutonomyLevel from '@/components/ai/AutonomyLevel.vue';

import Tag from 'primevue/tag';
import InputNumber from 'primevue/inputnumber';

// Mock values mirroring core user_data/scripts/ai_config.py structure bindings
const mockSettings = ref({
  budget: 50.00,
  ttl: 600,
  exponent: 2.0,
  overlap: 100
});
</script>
