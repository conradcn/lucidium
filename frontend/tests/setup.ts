import "@testing-library/react";

const denyNetwork = (): never => {
  throw new Error("Network calls are forbidden in offline tests (Constitution IV).");
};

if (typeof globalThis.fetch === "function") {
  globalThis.fetch = denyNetwork as unknown as typeof fetch;
}
