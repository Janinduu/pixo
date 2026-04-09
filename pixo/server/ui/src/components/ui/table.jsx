import { cn } from "../../lib/utils"

function Table({ className, ...props }) {
  return (
    <div className="relative w-full overflow-auto">
      <table className={cn("w-full caption-bottom text-sm", className)} {...props} />
    </div>
  )
}

function TableHeader({ className, ...props }) {
  return <thead className={cn("[&_tr]:border-b [&_tr]:border-zinc-100", className)} {...props} />
}

function TableBody({ className, ...props }) {
  return <tbody className={cn("[&_tr:last-child]:border-0", className)} {...props} />
}

function TableRow({ className, ...props }) {
  return (
    <tr className={cn("border-b border-zinc-100 transition-colors hover:bg-zinc-50", className)} {...props} />
  )
}

function TableHead({ className, ...props }) {
  return (
    <th className={cn("h-10 px-4 text-left align-middle text-[11px] font-medium uppercase tracking-wider text-zinc-400", className)} {...props} />
  )
}

function TableCell({ className, ...props }) {
  return (
    <td className={cn("px-4 py-3 align-middle text-[13px]", className)} {...props} />
  )
}

export { Table, TableHeader, TableBody, TableRow, TableHead, TableCell }
