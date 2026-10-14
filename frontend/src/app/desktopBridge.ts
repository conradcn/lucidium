/**
 * Typed access to the desktop-only helpers the Electron preload puts on
 * ``window.lucidium``. Each returns ``null`` / ``false`` when the
 * renderer runs outside Electron (Vite dev server, Vitest, Playwright
 * renderer-isolation runs), so callers can hide the button instead of
 * crashing.
 */

interface DesktopBridge {
  pickDirectory?: (defaultPath?: string) => Promise<string | null>;
  openFolder?: (dir: string) => Promise<boolean>;
}

function bridge(): DesktopBridge | undefined {
  // Structural cast, justified: ``window.lucidium`` is injected by the
  // Electron preload at runtime and has no ambient declaration here.
  return (window as unknown as { lucidium?: DesktopBridge }).lucidium;
}

export function canPickDirectory(): boolean {
  return typeof bridge()?.pickDirectory === "function";
}

export function canOpenFolder(): boolean {
  return typeof bridge()?.openFolder === "function";
}

/** Native folder picker; ``null`` when cancelled or unavailable. */
export async function pickDirectory(defaultPath?: string): Promise<string | null> {
  const fn = bridge()?.pickDirectory;
  return fn ? fn(defaultPath) : null;
}

/** Reveal a folder in the OS file manager. */
export async function openFolder(dir: string): Promise<boolean> {
  const fn = bridge()?.openFolder;
  return fn ? fn(dir) : false;
}
