/** Toast notification component.

Auto-dismisses after a configurable duration with an animated progress bar.

Usage:
  const [toast, setToast] = useState<ToastMsg | null>(null);

  <Toast msg={toast} onDone={() => setToast(null)} />

  setToast({ text: "Saved!", type: "success" });  // shows for 3s
  setToast({ text: "Failed", type: "error" });     // shows for 4s
*/
import { useEffect, useRef } from "react";

export interface ToastMsg {
  text: string;
  type?: "info" | "success" | "error";
  duration?: number; // ms, default 3000
}

interface Props {
  msg: ToastMsg | null;
  onDone: () => void;
}

export default function Toast({ msg, onDone }: Props) {
  // Use a ref for onDone so the effect doesn't depend on a potentially
  // unstable inline arrow function from the parent.
  const onDoneRef = useRef(onDone);
  onDoneRef.current = onDone;

  useEffect(() => {
    if (!msg) return;
    const t = setTimeout(() => onDoneRef.current(), msg.duration ?? 3000);
    return () => clearTimeout(t);
  }, [msg]);

  if (!msg) return null;

  const typeClass = msg.type === "success" ? " success" : msg.type === "error" ? " err" : "";
  return (
    <div className={"toast" + typeClass}>
      {msg.type === "success" && "✓ "}
      {msg.type === "error" && "✕ "}
      {msg.text}
      <span className="toast-bar" />
    </div>
  );
}
