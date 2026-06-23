/** A small labelled on/off toggle for the AI Boost panel (and reuse elsewhere).

 * Keep it dependency-free and accessible: a real checkbox visually styled as a
 * switch, so keyboard + screen-reader behaviour come for free.
 */
export default function Toggle({
  checked,
  onChange,
  label,
  hint,
  disabled,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  label: string;
  hint?: string;
  disabled?: boolean;
}) {
  return (
    <label
      className={"toggle-row" + (disabled ? " disabled" : "")}
      title={hint}
    >
      <input
        type="checkbox"
        checked={checked}
        disabled={disabled}
        onChange={(e) => onChange(e.target.checked)}
      />
      <span className={"toggle-switch" + (checked ? " on" : "")}>
        <span className="toggle-knob" />
      </span>
      <span className="toggle-text">
        <span className="toggle-label">{label}</span>
        {hint && <span className="toggle-hint muted tiny">{hint}</span>}
      </span>
    </label>
  );
}
