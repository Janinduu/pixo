import { cn } from "../../lib/utils"

function Progress({ value = 0, className, indicatorClassName }) {
  return (
    <div className={cn("h-1.5 w-full overflow-hidden rounded-full bg-zinc-100", className)}>
      <div
        className={cn("h-full rounded-full transition-all duration-500 ease-out bg-zinc-900", indicatorClassName)}
        style={{ width: `${Math.min(100, Math.max(0, value))}%` }}
      />
    </div>
  )
}

export { Progress }
