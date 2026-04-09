import { cn } from "../../lib/utils"

function Input({ className, ...props }) {
  return (
    <input
      className={cn(
        "flex h-9 w-full rounded-lg border border-zinc-200 bg-white px-3 py-2 text-sm text-zinc-900 placeholder:text-zinc-400 focus:outline-none focus:ring-2 focus:ring-zinc-200 focus:border-zinc-300 transition-colors disabled:cursor-not-allowed disabled:opacity-50",
        className
      )}
      {...props}
    />
  )
}

function Select({ className, children, ...props }) {
  return (
    <div className="relative">
      <select
        className={cn(
          "flex h-9 w-full rounded-lg border border-zinc-200 bg-white px-3 py-2 pr-8 text-sm text-zinc-900 focus:outline-none focus:ring-2 focus:ring-zinc-200 focus:border-zinc-300 transition-colors appearance-none cursor-pointer",
          className
        )}
        {...props}
      >
        {children}
      </select>
      <svg className="absolute right-2.5 top-1/2 -translate-y-1/2 pointer-events-none text-zinc-400" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="m6 9 6 6 6-6"/>
      </svg>
    </div>
  )
}

export { Input, Select }
