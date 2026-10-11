<script setup lang="ts">
import { ref, useId } from "vue";
import { ChevronDown } from "lucide-vue-next";

defineProps<{ title: string }>();

const expanded = ref(true);
const contentId = `adjustment-panel-${useId()}`;
</script>

<template>
  <section class="rounded-xl border border-slate-200 bg-white p-3 text-slate-900 shadow-sm dark:border-zinc-800 dark:bg-zinc-900 dark:text-white">
    <header class="flex items-center gap-2" :class="expanded ? 'mb-2' : ''">
      <button
        type="button"
        class="flex min-w-0 flex-1 items-center gap-2 text-left"
        :aria-expanded="expanded"
        :aria-controls="contentId"
        @click="expanded = !expanded"
      >
        <ChevronDown class="h-3.5 w-3.5 shrink-0 text-slate-600 transition dark:text-zinc-200" :class="expanded ? '' : '-rotate-90'" />
        <span class="truncate text-sm font-semibold">{{ title }}</span>
      </button>
      <div v-if="$slots.actions" class="flex shrink-0 items-center" @click.stop>
        <slot name="actions" />
      </div>
    </header>
    <div :id="contentId" v-show="expanded" class="min-w-0">
      <slot />
    </div>
  </section>
</template>
