"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const NAV_ITEMS = [
  { href: "/", label: "Videos" },
  { href: "/documents", label: "Documents" },
];

export function Navbar() {
  const pathname = usePathname();

  return (
    <nav className="flex items-center gap-0 border-b border-hud-border bg-hud-panel">
      <Link
        href="/"
        className="px-5 py-2.5 text-base font-bold tracking-[0.3em] text-foreground lowercase"
      >
        sanjaya
      </Link>
      <div className="h-5 w-px bg-hud-border" />
      {NAV_ITEMS.map((item) => {
        const isActive =
          item.href === "/"
            ? pathname === "/"
            : pathname.startsWith(item.href);
        return (
          <Link
            key={item.href}
            href={item.href}
            className={`px-4 py-2.5 text-xs font-bold uppercase tracking-[0.15em] transition-colors ${
              isActive
                ? "text-foreground border-b-2 border-hud-green"
                : "text-hud-dim hover:text-foreground"
            }`}
          >
            {item.label}
          </Link>
        );
      })}
    </nav>
  );
}
