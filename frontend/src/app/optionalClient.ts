import { ensureClient } from "./client";
import type { WsClient } from "../ws/client";

/**
 * ``ensureClient()`` typed as a partial.
 *
 * Several unit tests mock the ``app/client`` module with a stub that
 * has no ``on`` (or is a bare ``vi.fn()``), so the runtime value can
 * violate the declared ``WsClient`` return type. Narrowing to a partial
 * lets the caller feature-detect ``on`` without crashing the component
 * tree — and, unlike a hand-written signature, keeps ``on``'s generated
 * per-message payload typing intact.
 */
export function optionalClient(): Partial<WsClient> | undefined {
  return ensureClient() as Partial<WsClient> | undefined;
}
