import type { JSX } from "react";
import { useLucidiumStore } from "../../../state/store";

interface DialogNode {
  id: string;
  parent_id: string | null;
  chosen_option_id: string | null;
  text: string | null;
  state: string;
  options: { id: string; text: string }[];
  generation_metadata?: {
    model?: string | null;
    latency_ms?: number;
    tokens_in?: number;
    tokens_out?: number;
  };
}

const STATE_ORDER = [
  "committed",
  "speculative",
  "pending_text",
  "ready",
  "invalidated",
];

export function DialogTreeTab(): JSX.Element {
  const game = useLucidiumStore((s) => s.game) as
    | {
        current_node_id?: string | null;
        dialog_tree?: {
          nodes: Record<string, DialogNode>;
          committed_path: string[];
        };
      }
    | null;

  if (!game?.dialog_tree) return <p>(no game loaded)</p>;
  const committed = new Set(game.dialog_tree.committed_path);
  const nodes = Object.values(game.dialog_tree.nodes);

  // Counts by state for the freeze-diagnosis box.
  const stateCounts = new Map<string, number>();
  for (const n of nodes) {
    stateCounts.set(n.state, (stateCounts.get(n.state) ?? 0) + 1);
  }

  // Current node + its children — a frozen run usually shows here as
  // a current node whose options have NO ``ready``/``committed``
  // children, the LLM queue idle, and a ``pending_text`` child stuck.
  const currentNode = game.current_node_id
    ? game.dialog_tree.nodes[game.current_node_id]
    : null;
  const childNodes = currentNode
    ? nodes.filter((n) => n.parent_id === currentNode.id)
    : [];

  return (
    <div>
      <details
        open
        style={{
          marginBottom: "0.85rem",
          padding: "0.5rem 0.75rem",
          background: "rgba(255,255,255,0.03)",
          border: "1px solid var(--border-soft)",
          borderRadius: "var(--r-md)",
          fontSize: "0.85rem",
        }}
      >
        <summary
          style={{
            cursor: "pointer",
            color: "var(--accent)",
            fontWeight: 600,
          }}
        >
          Debug
        </summary>
        <div style={{ marginTop: "0.5rem", lineHeight: 1.6 }}>
          <div>
            <strong>nodes:</strong> {nodes.length} total
            {STATE_ORDER.filter((s) => stateCounts.has(s))
              .map((s) => `, ${s} ${stateCounts.get(s)}`)
              .join("")}
            {Array.from(stateCounts.keys())
              .filter((s) => !STATE_ORDER.includes(s))
              .map((s) => `, ${s} ${stateCounts.get(s)}`)
              .join("")}
          </div>
          {currentNode ? (
            <div style={{ marginTop: "0.4rem" }}>
              <strong>current:</strong> {currentNode.id} · {currentNode.state}
              {currentNode.options.length > 0
                ? ` · ${currentNode.options.length} option${currentNode.options.length === 1 ? "" : "s"}`
                : " · continue"}
            </div>
          ) : null}
          {currentNode && childNodes.length > 0 ? (
            <div style={{ marginTop: "0.25rem" }}>
              <strong>children:</strong>
              <ul
                style={{
                  margin: "0.25rem 0 0",
                  paddingLeft: "1.25rem",
                  fontFamily:
                    "ui-monospace, SFMono-Regular, Menlo, monospace",
                  fontSize: "0.8rem",
                }}
              >
                {childNodes.map((child) => {
                  const optionLabel =
                    currentNode.options.find(
                      (o) => o.id === child.chosen_option_id,
                    )?.text ??
                    (child.chosen_option_id === null ? "(continue)" : child.chosen_option_id);
                  return (
                    <li key={child.id}>
                      {child.id} · {child.state} · {optionLabel}
                    </li>
                  );
                })}
              </ul>
            </div>
          ) : null}
          <div
            style={{
              marginTop: "0.5rem",
              color: "var(--muted)",
              fontSize: "0.8rem",
            }}
          >
            If a beat is stuck — current node has options but no{" "}
            <code>committed</code>/<code>ready</code> child and the LLM
            queue is idle — use <strong>Undo</strong> in the top bar to
            roll back to the last choice and try again, or send a free-
            text input to force a fresh generation under a new premise.
          </div>
        </div>
      </details>
      <ul style={{ listStyle: "none", padding: 0 }}>
        {nodes.map((node) => {
          const isCommitted = committed.has(node.id);
          const isCurrent = game.current_node_id === node.id;
          return (
            <li
              key={node.id}
              style={{
                padding: "0.4rem",
                borderLeft: isCurrent
                  ? "3px solid var(--accent)"
                  : isCommitted
                    ? "3px solid var(--fg)"
                    : "3px solid var(--border)",
                opacity: node.state === "invalidated" ? 0.4 : 1,
                marginBottom: "0.4rem",
              }}
            >
              <div style={{ color: "var(--muted)", fontSize: "0.75rem" }}>
                {node.id} · {node.state}
                {node.parent_id ? ` · parent ${node.parent_id}` : ""}
                {node.generation_metadata?.latency_ms
                  ? ` · ${node.generation_metadata.latency_ms}ms`
                  : ""}
              </div>
              <div>{(node.text ?? "(pending)").slice(0, 200)}</div>
              {node.options.length > 0 ? (
                <div style={{ color: "var(--muted)", fontSize: "0.85rem" }}>
                  {node.options.map((o) => o.text).join(" / ")}
                </div>
              ) : null}
            </li>
          );
        })}
      </ul>
    </div>
  );
}
