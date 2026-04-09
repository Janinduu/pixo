import { cn } from "../../lib/utils"

function Card({ className, ...props }) {
  return (
    <div className={cn("rounded-xl border border-zinc-200 bg-white", className)} {...props} />
  )
}

function CardHeader({ className, ...props }) {
  return (
    <div className={cn("flex flex-col space-y-1.5 p-5 pb-3", className)} {...props} />
  )
}

function CardTitle({ className, ...props }) {
  return (
    <h3 className={cn("text-sm font-semibold text-zinc-900", className)} {...props} />
  )
}

function CardDescription({ className, ...props }) {
  return (
    <p className={cn("text-xs text-zinc-500", className)} {...props} />
  )
}

function CardContent({ className, ...props }) {
  return (
    <div className={cn("p-5 pt-0", className)} {...props} />
  )
}

function CardFooter({ className, ...props }) {
  return (
    <div className={cn("flex items-center p-5 pt-0", className)} {...props} />
  )
}

export { Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter }
