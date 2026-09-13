import { useCallback, useEffect, useRef, useState } from 'react'
import type { Point } from './api'
import { clusterColor } from './api'

interface Props {
  points: Point[]
  selected: Set<number>
  activeToken: string | null
  onPick: (index: number, additive: boolean) => void
  onBoxSelect: (indices: number[], additive: boolean) => void
}

interface Rect {
  x0: number
  y0: number
  x1: number
  y1: number
}

const PADDING = 28
const RADIUS = 4

/**
 * Canvas scatter plot.
 *
 * Drawn by hand rather than with a charting library: the plot needs a
 * hit-testable point set, drag-box selection and single-click playback, which
 * is less code here than bending a chart library into shape.
 */
export default function Scatter({ points, selected, activeToken, onPick, onBoxSelect }: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const [rect, setRect] = useState<Rect | null>(null)
  const [viewport, setViewport] = useState({ width: 800, height: 600 })
  const dragRef = useRef<{ x: number; y: number; additive: boolean } | null>(null)
  const boundsRef = useRef({ minX: 0, maxX: 1, minY: 0, maxY: 1 })

  // --- coordinate mapping -------------------------------------------------

  const toScreen = useCallback(
    (x: number, y: number) => {
      const { minX, maxX, minY, maxY } = boundsRef.current
      const innerW = viewport.width - PADDING * 2
      const innerH = viewport.height - PADDING * 2
      const spanX = maxX - minX || 1
      const spanY = maxY - minY || 1
      return {
        sx: PADDING + ((x - minX) / spanX) * innerW,
        sy: viewport.height - PADDING - ((y - minY) / spanY) * innerH,
      }
    },
    [viewport],
  )

  const toData = useCallback(
    (sx: number, sy: number) => {
      const { minX, maxX, minY, maxY } = boundsRef.current
      const innerW = viewport.width - PADDING * 2
      const innerH = viewport.height - PADDING * 2
      return {
        x: minX + ((sx - PADDING) / (innerW || 1)) * (maxX - minX || 1),
        y: minY + ((viewport.height - PADDING - sy) / (innerH || 1)) * (maxY - minY || 1),
      }
    },
    [viewport],
  )

  // --- sizing -------------------------------------------------------------

  useEffect(() => {
    const element = canvasRef.current
    if (!element) return
    const parent = element.parentElement
    if (!parent) return
    const observer = new ResizeObserver(() => {
      setViewport({ width: parent.clientWidth, height: parent.clientHeight })
    })
    observer.observe(parent)
    setViewport({ width: parent.clientWidth, height: parent.clientHeight })
    return () => observer.disconnect()
  }, [])

  // --- painting -----------------------------------------------------------

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const dpr = window.devicePixelRatio || 1
    canvas.width = viewport.width * dpr
    canvas.height = viewport.height * dpr
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, viewport.width, viewport.height)

    const styles = getComputedStyle(canvas)
    const gridColor = styles.getPropertyValue('--grid').trim() || 'rgba(128,128,128,.18)'
    const axisColor = styles.getPropertyValue('--muted').trim() || 'rgba(128,128,128,.9)'

    if (points.length === 0) {
      ctx.fillStyle = axisColor
      ctx.font = '13px system-ui, sans-serif'
      ctx.textAlign = 'center'
      ctx.fillText('No points yet — scan a directory and run.', viewport.width / 2, viewport.height / 2)
      return
    }

    let minX = Infinity
    let maxX = -Infinity
    let minY = Infinity
    let maxY = -Infinity
    for (const point of points) {
      if (point.x < minX) minX = point.x
      if (point.x > maxX) maxX = point.x
      if (point.y < minY) minY = point.y
      if (point.y > maxY) maxY = point.y
    }
    const padX = (maxX - minX || 1) * 0.05
    const padY = (maxY - minY || 1) * 0.05
    boundsRef.current = {
      minX: minX - padX,
      maxX: maxX + padX,
      minY: minY - padY,
      maxY: maxY + padY,
    }

    ctx.strokeStyle = gridColor
    ctx.lineWidth = 0.5
    for (let i = 0; i <= 8; i += 1) {
      const gx = PADDING + ((viewport.width - PADDING * 2) / 8) * i
      const gy = PADDING + ((viewport.height - PADDING * 2) / 8) * i
      ctx.beginPath()
      ctx.moveTo(gx, PADDING)
      ctx.lineTo(gx, viewport.height - PADDING)
      ctx.stroke()
      ctx.beginPath()
      ctx.moveTo(PADDING, gy)
      ctx.lineTo(viewport.width - PADDING, gy)
      ctx.stroke()
    }

    for (const point of points) {
      const { sx, sy } = toScreen(point.x, point.y)
      const isSelected = selected.has(point.i)
      const isActive = activeToken !== null && point.token === activeToken
      ctx.beginPath()
      ctx.arc(sx, sy, isSelected || isActive ? RADIUS + 2.5 : RADIUS, 0, Math.PI * 2)
      ctx.fillStyle = clusterColor(point.cluster)
      ctx.globalAlpha = selected.size > 0 && !isSelected ? 0.35 : 0.9
      ctx.fill()
      ctx.globalAlpha = 1
      if (isSelected || isActive) {
        ctx.strokeStyle = axisColor
        ctx.lineWidth = 1.5
        ctx.stroke()
      }
    }

    if (rect) {
      ctx.strokeStyle = axisColor
      ctx.lineWidth = 1
      ctx.setLineDash([4, 3])
      ctx.strokeRect(
        Math.min(rect.x0, rect.x1),
        Math.min(rect.y0, rect.y1),
        Math.abs(rect.x1 - rect.x0),
        Math.abs(rect.y1 - rect.y0),
      )
      ctx.setLineDash([])
    }
  }, [points, selected, activeToken, rect, viewport, toScreen])

  // --- interaction --------------------------------------------------------

  const localPoint = (event: React.MouseEvent<HTMLCanvasElement>) => {
    const bounds = event.currentTarget.getBoundingClientRect()
    return { x: event.clientX - bounds.left, y: event.clientY - bounds.top }
  }

  const nearest = (sx: number, sy: number): number | null => {
    let best: number | null = null
    let bestDistance = 12 * 12
    for (const point of points) {
      const screen = toScreen(point.x, point.y)
      const distance = (screen.sx - sx) ** 2 + (screen.sy - sy) ** 2
      if (distance < bestDistance) {
        bestDistance = distance
        best = point.i
      }
    }
    return best
  }

  const handleMouseDown = (event: React.MouseEvent<HTMLCanvasElement>) => {
    const { x, y } = localPoint(event)
    dragRef.current = { x, y, additive: event.shiftKey || event.ctrlKey || event.metaKey }
    setRect({ x0: x, y0: y, x1: x, y1: y })
  }

  const handleMouseMove = (event: React.MouseEvent<HTMLCanvasElement>) => {
    const drag = dragRef.current
    if (!drag) return
    const { x, y } = localPoint(event)
    if (Math.abs(x - drag.x) + Math.abs(y - drag.y) > 3) {
      setRect({ x0: drag.x, y0: drag.y, x1: x, y1: y })
    }
  }

  const handleMouseUp = (event: React.MouseEvent<HTMLCanvasElement>) => {
    const drag = dragRef.current
    const current = rect
    dragRef.current = null
    setRect(null)
    if (!drag) return

    const { x, y } = localPoint(event)
    const dragged = Math.abs(x - drag.x) + Math.abs(y - drag.y) > 3
    if (!dragged) {
      const hit = nearest(x, y)
      if (hit !== null) onPick(hit, drag.additive)
      return
    }

    if (!current) return
    const data0 = toData(Math.min(current.x0, current.x1), Math.max(current.y0, current.y1))
    const data1 = toData(Math.max(current.x0, current.x1), Math.min(current.y0, current.y1))
    const inside = points
      .filter(
        (point) =>
          point.x >= data0.x && point.x <= data1.x && point.y >= data0.y && point.y <= data1.y,
      )
      .map((point) => point.i)
    onBoxSelect(inside, drag.additive)
  }

  return (
    <canvas
      ref={canvasRef}
      className="scatter"
      style={{ width: '100%', height: '100%' }}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={() => {
        dragRef.current = null
        setRect(null)
      }}
    />
  )
}
