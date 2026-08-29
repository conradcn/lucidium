import { describe, expect, it, beforeAll, afterAll } from "vitest";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import {
  confineAssetPath,
  forbiddenResponse,
  resolveAssetRequest,
} from "../../electron/assetPath";

/** Build the URL the renderer's assetUrl() would produce for a disk path. */
function assetUrl(diskPath: string): string {
  const segments = diskPath
    .replace(/\\/g, "/")
    .split("/")
    .filter((s) => s.length > 0)
    .map((s) => encodeURIComponent(s));
  return `lucidium-asset://local/${segments.join("/")}`;
}

let root: string;
let savesRoot: string;
let portrait: string;
let roots: string[];

beforeAll(() => {
  root = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), "lucidium-asset-")));
  savesRoot = path.join(root, "saves");
  const imagesDir = path.join(savesRoot, "game-1", "images");
  fs.mkdirSync(imagesDir, { recursive: true });
  portrait = path.join(imagesDir, "portrait ada.png");
  fs.writeFileSync(portrait, "not really a png");
  fs.writeFileSync(path.join(root, "settings.json"), '{"llm":{"api_key":"secret"}}');
  roots = [savesRoot];
});

afterAll(() => {
  fs.rmSync(root, { recursive: true, force: true });
});

describe("lucidium-asset:// confinement", () => {
  it("serves a legitimate portrait inside the saves root", () => {
    const resolved = resolveAssetRequest(assetUrl(portrait), roots);
    expect(resolved).not.toBeNull();
    expect(fs.readFileSync(resolved as string, "utf-8")).toBe("not really a png");
  });

  it("refuses an absolute path outside the root", () => {
    expect(resolveAssetRequest("lucidium-asset://local/C%3A/Windows/win.ini", roots)).toBeNull();
    expect(resolveAssetRequest(assetUrl("/etc/passwd"), roots)).toBeNull();
  });

  it("refuses %2e%2e traversal that survives URL normalisation", () => {
    // Chromium collapses literal ``..`` segments but leaves ``%2e%2e``
    // alone, so this only becomes a traversal after we decode — the
    // check has to run on the resolved path.
    fs.writeFileSync(path.join(root, "outside.png"), "x");
    const base = assetUrl(path.join(savesRoot, "game-1", "images"));
    for (const target of ["settings.json", "outside.png"]) {
      const url = `${base}/%2e%2e/%2e%2e/%2e%2e/${target}`;
      expect(url).toContain("%2e%2e");
      expect(resolveAssetRequest(url, roots)).toBeNull();
    }
  });

  it("refuses a non-media extension even inside the root", () => {
    const inside = path.join(savesRoot, "game-1", "game.json");
    fs.writeFileSync(inside, "{}");
    expect(resolveAssetRequest(assetUrl(inside), roots)).toBeNull();
  });

  it("refuses a sibling directory that merely shares the root's prefix", () => {
    expect(confineAssetPath(`${savesRoot}-evil${path.sep}x.png`, roots)).toBeNull();
  });

  it("refuses relative and malformed URLs", () => {
    expect(confineAssetPath("images/portrait.png", roots)).toBeNull();
    expect(resolveAssetRequest("lucidium-asset://local/%zz.png", roots)).toBeNull();
    expect(resolveAssetRequest("lucidium-asset://local/", roots)).toBeNull();
  });

  it("accepts audio as well as image extensions", () => {
    const track = path.join(savesRoot, "game-1", "images", "music-abc.wav");
    fs.writeFileSync(track, "riff");
    expect(resolveAssetRequest(assetUrl(track), roots)).not.toBeNull();
  });

  it("answers refusals with a 403", async () => {
    const response = forbiddenResponse();
    expect(response.status).toBe(403);
    expect(await response.text()).toBe("Forbidden");
  });
});
