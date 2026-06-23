import { describe, it, expect } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { LanguageProvider, useT } from "./i18n";

function Probe({ keys }: { keys: string[] }) {
  const { t } = useT();
  return <>{keys.map((k) => t(k)).join("|")}</>;
}

describe("i18n", () => {
  it("resolves German by default and falls back to the key when missing", () => {
    const html = renderToStaticMarkup(
      <LanguageProvider>
        <Probe keys={["swipe.good", "does.not.exist"]} />
      </LanguageProvider>,
    );
    expect(html).toContain("Gut");
    expect(html).toContain("does.not.exist");
  });

  it("useT outside a provider still returns usable German strings", () => {
    const html = renderToStaticMarkup(<Probe keys={["swipe.good"]} />);
    expect(html).toContain("Gut");
  });
});
