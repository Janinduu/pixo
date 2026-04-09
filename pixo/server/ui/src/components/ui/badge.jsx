import { cva } from "class-variance-authority"
import { cn } from "../../lib/utils"

const badgeVariants = cva(
  "inline-flex items-center rounded-md px-2 py-0.5 text-[11px] font-medium transition-colors",
  {
    variants: {
      variant: {
        default: "bg-zinc-100 text-zinc-600 border border-zinc-200",
        success: "bg-emerald-50 text-emerald-600 border border-emerald-200",
        warning: "bg-amber-50 text-amber-600 border border-amber-200",
        error: "bg-red-50 text-red-600 border border-red-200",
        outline: "border border-zinc-200 text-zinc-500",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
)

function Badge({ className, variant, ...props }) {
  return (
    <span className={cn(badgeVariants({ variant }), className)} {...props} />
  )
}

export { Badge, badgeVariants }
