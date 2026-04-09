import { cva } from "class-variance-authority"
import { cn } from "../../lib/utils"

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-lg text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-zinc-300 disabled:pointer-events-none disabled:opacity-50 cursor-pointer",
  {
    variants: {
      variant: {
        default: "bg-zinc-900 text-white hover:bg-zinc-800",
        secondary: "bg-zinc-100 text-zinc-700 border border-zinc-200 hover:bg-zinc-200 hover:text-zinc-900",
        ghost: "text-zinc-500 hover:text-zinc-900 hover:bg-zinc-100",
        outline: "border border-zinc-200 text-zinc-600 hover:bg-zinc-50 hover:text-zinc-900",
        destructive: "bg-red-50 text-red-600 border border-red-200 hover:bg-red-100",
      },
      size: {
        default: "h-9 px-4 py-2",
        sm: "h-8 px-3 text-xs",
        lg: "h-10 px-6",
        icon: "h-9 w-9",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
)

function Button({ className, variant, size, ...props }) {
  return (
    <button className={cn(buttonVariants({ variant, size, className }))} {...props} />
  )
}

export { Button, buttonVariants }
