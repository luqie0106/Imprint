<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'

const props = defineProps<{
  src: string
  alt: string
  viewportWidth: number
  viewportHeight: number
  stageWidth: number
  stageHeight: number
  panX: number
  panY: number
  pixelRatio: number
}>()

const emit = defineEmits<{
  loaded: [dimensions: { width: number; height: number }]
  error: [event: Event | string]
}>()

const canvas = ref<HTMLCanvasElement | null>(null)
const sourceImage = ref<HTMLImageElement | null>(null)
const sourceWidth = computed(() => sourceImage.value?.naturalWidth ?? 0)
const sourceHeight = computed(() => sourceImage.value?.naturalHeight ?? 0)

function nonNegative(value: number): number {
  return Number.isFinite(value) ? Math.max(0, value) : 0
}

function effectivePixelRatio(): number {
  return Number.isFinite(props.pixelRatio) && props.pixelRatio > 0 ? props.pixelRatio : 1
}

const backingWidth = computed(() => Math.round(nonNegative(props.viewportWidth) * effectivePixelRatio()))
const backingHeight = computed(() => Math.round(nonNegative(props.viewportHeight) * effectivePixelRatio()))

let mounted = false
let loadGeneration = 0
let pendingImage: HTMLImageElement | null = null
let drawFrame: number | null = null

function cancelPendingImage(): void {
  if (!pendingImage) return
  pendingImage.onload = null
  pendingImage.onerror = null
  pendingImage.src = ''
  pendingImage = null
}

function scheduleDraw(): void {
  if (!mounted || drawFrame !== null) return
  drawFrame = window.requestAnimationFrame(() => {
    drawFrame = null
    drawVisibleCrop()
  })
}

function drawVisibleCrop(): void {
  const element = canvas.value
  if (!element) return

  const targetWidth = backingWidth.value
  const targetHeight = backingHeight.value
  if (element.width !== targetWidth) element.width = targetWidth
  if (element.height !== targetHeight) element.height = targetHeight

  const context = element.getContext('2d')
  if (!context) return

  const viewportWidth = nonNegative(props.viewportWidth)
  const viewportHeight = nonNegative(props.viewportHeight)
  if (viewportWidth === 0 || viewportHeight === 0 || targetWidth === 0 || targetHeight === 0) return

  const scaleX = targetWidth / viewportWidth
  const scaleY = targetHeight / viewportHeight
  context.setTransform(scaleX, 0, 0, scaleY, 0, 0)
  context.clearRect(0, 0, viewportWidth, viewportHeight)

  const image = sourceImage.value
  const stageWidth = props.stageWidth
  const stageHeight = props.stageHeight
  if (
    !image
    || image.naturalWidth === 0
    || image.naturalHeight === 0
    || !Number.isFinite(stageWidth)
    || !Number.isFinite(stageHeight)
    || !Number.isFinite(props.panX)
    || !Number.isFinite(props.panY)
    || stageWidth <= 0
    || stageHeight <= 0
  ) return

  const imageLeft = (viewportWidth - stageWidth) / 2 + props.panX
  const imageTop = (viewportHeight - stageHeight) / 2 + props.panY
  const visibleLeft = Math.max(0, imageLeft)
  const visibleTop = Math.max(0, imageTop)
  const visibleRight = Math.min(viewportWidth, imageLeft + stageWidth)
  const visibleBottom = Math.min(viewportHeight, imageTop + stageHeight)
  if (visibleRight <= visibleLeft || visibleBottom <= visibleTop) return

  const sourceX = ((visibleLeft - imageLeft) / stageWidth) * image.naturalWidth
  const sourceY = ((visibleTop - imageTop) / stageHeight) * image.naturalHeight
  const sourceCropWidth = ((visibleRight - visibleLeft) / stageWidth) * image.naturalWidth
  const sourceCropHeight = ((visibleBottom - visibleTop) / stageHeight) * image.naturalHeight

  context.imageSmoothingEnabled = true
  context.imageSmoothingQuality = 'high'
  context.drawImage(
    image,
    sourceX,
    sourceY,
    sourceCropWidth,
    sourceCropHeight,
    visibleLeft,
    visibleTop,
    visibleRight - visibleLeft,
    visibleBottom - visibleTop,
  )
}

function loadSource(src: string): void {
  const generation = ++loadGeneration
  cancelPendingImage()

  if (!src) {
    sourceImage.value = null
    scheduleDraw()
    return
  }

  const image = new Image()
  pendingImage = image
  image.onload = () => {
    if (generation !== loadGeneration) return
    pendingImage = null
    sourceImage.value = image
    emit('loaded', { width: image.naturalWidth, height: image.naturalHeight })
    scheduleDraw()
  }
  image.onerror = (event) => {
    if (generation !== loadGeneration) return
    pendingImage = null
    emit('error', event)
  }
  image.src = src
}

watch(
  () => [
    props.viewportWidth,
    props.viewportHeight,
    props.stageWidth,
    props.stageHeight,
    props.panX,
    props.panY,
    props.pixelRatio,
  ],
  scheduleDraw,
)

watch(() => props.src, (src) => {
  if (mounted) loadSource(src)
})

onMounted(() => {
  mounted = true
  loadSource(props.src)
  scheduleDraw()
})

onBeforeUnmount(() => {
  mounted = false
  loadGeneration += 1
  cancelPendingImage()
  if (drawFrame !== null) {
    window.cancelAnimationFrame(drawFrame)
    drawFrame = null
  }
})
</script>

<template>
  <canvas
    ref="canvas"
    class="preview-canvas"
    role="img"
    :aria-label="alt"
    :data-source-width="sourceWidth"
    :data-source-height="sourceHeight"
    :data-render-width="backingWidth"
    :data-render-height="backingHeight"
  />
</template>

<style scoped>
.preview-canvas {
  position: absolute;
  inset: 0;
  display: block;
  width: 100%;
  height: 100%;
  pointer-events: none;
}
</style>
