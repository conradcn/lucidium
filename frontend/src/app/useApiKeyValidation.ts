import { useEffect, useRef, useState } from "react";

import { send } from "./client";
import { optionalClient } from "./optionalClient";

export type ApiKeyValidationStatus =
  | "idle"
  | "pending"
  | "valid"
  | "unauthorized"
  | "unreachable"
  | "invalid_input";

export interface ApiKeyValidationState {
  status: ApiKeyValidationStatus;
  message: string;
}

/**
 * Drive an ``s2c/settings/api_key_validation`` round-trip from the
 * Settings / FirstTimeSetup API-key input. ``validate(base_url,
 * api_key)`` fires ``c2s/settings/validate_api_key``; the latest reply
 * lands in ``state``. ``reset()`` clears the badge — used when the
 * player edits the field again so a stale "valid" doesn't sit on top
 * of unconfirmed input.
 *
 * Why a per-instance subscription rather than a global listener:
 * Settings and FirstTimeSetup can both be mounted across a session,
 * and a single global handler couldn't tell which screen's input
 * the reply belongs to. The backend doesn't echo a correlation id
 * (validation is fire-and-forget from the player's perspective), so
 * the hook just takes the most recent reply — the input's onBlur
 * is the only producer in practice and races aren't realistic.
 */
export function useApiKeyValidation(): {
  state: ApiKeyValidationState;
  validate: (base_url: string, api_key: string) => void;
  reset: () => void;
} {
  const [state, setState] = useState<ApiKeyValidationState>({
    status: "idle",
    message: "",
  });
  // The hook is mounted by multiple screens — guard the listener so
  // each instance only reacts to replies that arrive AFTER it last
  // requested validation. Without this, an unrelated screen's reply
  // could swap a fresh "pending" into a stale "valid" badge.
  const awaitingRef = useRef(false);

  useEffect(() => {
    const client = optionalClient();
    if (!client || typeof client.on !== "function") return;
    const off = client.on("s2c/settings/api_key_validation", (p) => {
      if (!awaitingRef.current) return;
      awaitingRef.current = false;
      const status: ApiKeyValidationStatus =
        p.status === "valid" ||
        p.status === "unauthorized" ||
        p.status === "unreachable" ||
        p.status === "invalid_input"
          ? p.status
          : "unreachable";
      setState({ status, message: p.message || "" });
    });
    return off;
  }, []);

  const validate = (base_url: string, api_key: string): void => {
    awaitingRef.current = true;
    setState({ status: "pending", message: "Checking…" });
    send("c2s/settings/validate_api_key", { base_url, api_key });
  };

  const reset = (): void => {
    awaitingRef.current = false;
    setState({ status: "idle", message: "" });
  };

  return { state, validate, reset };
}
