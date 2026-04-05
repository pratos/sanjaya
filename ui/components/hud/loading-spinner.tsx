"use client";

/** HUD-style loading spinner — rotating line segments around a center. */
export function LoadingSpinner({ size = 16 }: { size?: number }) {
  return (
    <div
      className="inline-flex items-center justify-center"
      style={{ width: size, height: size }}
    >
      <div
        className="animate-spin"
        style={{ width: size, height: size }}
      >
        <svg viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
          <line x1="8" y1="1" x2="8" y2="4" stroke="currentColor" strokeWidth="1.5" opacity="1" />
          <line x1="12.9" y1="3.1" x2="10.8" y2="5.2" stroke="currentColor" strokeWidth="1.5" opacity="0.75" />
          <line x1="15" y1="8" x2="12" y2="8" stroke="currentColor" strokeWidth="1.5" opacity="0.5" />
          <line x1="12.9" y1="12.9" x2="10.8" y2="10.8" stroke="currentColor" strokeWidth="1.5" opacity="0.375" />
          <line x1="8" y1="15" x2="8" y2="12" stroke="currentColor" strokeWidth="1.5" opacity="0.25" />
          <line x1="3.1" y1="12.9" x2="5.2" y2="10.8" stroke="currentColor" strokeWidth="1.5" opacity="0.2" />
          <line x1="1" y1="8" x2="4" y2="8" stroke="currentColor" strokeWidth="1.5" opacity="0.15" />
          <line x1="3.1" y1="3.1" x2="5.2" y2="5.2" stroke="currentColor" strokeWidth="1.5" opacity="0.1" />
        </svg>
      </div>
    </div>
  );
}
