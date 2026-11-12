/** Reusable empty state with icon, title, and optional action. */
import type { ReactNode } from "react";

interface Props {
  icon?: string;
  title: string;
  text?: string;
  action?: ReactNode;
}

export default function EmptyState({ title, text, action }: Props) {
  return (
    <div className="empty" style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 12 }}>
      <span className="empty-mark" aria-hidden="true">!</span>
      <h3 style={{ color: "var(--text)", fontWeight: 600 }}>{title}</h3>
      {text && <p className="muted" style={{ margin: 0, maxWidth: 400 }}>{text}</p>}
      {action && <div style={{ marginTop: 8 }}>{action}</div>}
    </div>
  );
}
