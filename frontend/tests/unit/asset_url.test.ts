/**
 * Asset URL helper — single point of contact for converting backend
 * disk paths into renderer-safe URLs. The fix lives here, so the
 * regression test does too: any future drift back to raw ``file://``
 * URLs (which Electron 31 blocks cross-origin) will trip these
 * assertions before it lands in production.
 */
import { describe, expect, it } from "vitest";

import { ASSET_HOST, ASSET_SCHEME, assetUrl } from "../../src/app/assetUrl";

describe("assetUrl", () => {
  it("returns null for empty / missing input", () => {
    expect(assetUrl(null)).toBeNull();
    expect(assetUrl(undefined)).toBeNull();
    expect(assetUrl("")).toBeNull();
  });

  it("uses the lucidium-asset scheme + fixed host — never raw file://", () => {
    const url = assetUrl("C:/Users/Iris/portrait.png");
    expect(url).not.toBeNull();
    expect(url!.startsWith(`${ASSET_SCHEME}://${ASSET_HOST}/`)).toBe(true);
    expect(url).not.toMatch(/^file:/);
    // The URL must parse with the standard URL constructor (otherwise
    // fetch() rejects it as malformed).
    expect(() => new URL(url!)).not.toThrow();
  });

  it("normalises backslashes to forward slashes", () => {
    const url = assetUrl("C:\\Users\\Iris\\portrait.png");
    expect(url).toContain("C%3A/Users/Iris/portrait.png");
    expect(url).not.toContain("\\");
  });

  it("percent-encodes spaces and unsafe characters", () => {
    const url = assetUrl("C:/some dir/space file.png");
    expect(url).toContain("some%20dir/space%20file.png");
  });

  it("preserves the file extension verbatim", () => {
    const url = assetUrl("C:/x/portrait.png");
    expect(url!.endsWith(".png")).toBe(true);
  });

  it("works for POSIX-shaped paths", () => {
    const url = assetUrl("/home/iris/portrait.png");
    expect(url).toBe(`${ASSET_SCHEME}://${ASSET_HOST}/home/iris/portrait.png`);
  });
});
