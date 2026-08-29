import type { JSX } from "react";
import { assetUrl } from "../../app/assetUrl";
import { useLucidiumStore } from "../../state/store";

interface DialogNode {
  id: string;
  parent_id: string | null;
  location_id: string | null;
}

interface Environment {
  id: string;
  image_path: string | null;
}

export function Background(): JSX.Element {
  const game = useLucidiumStore((s) => s.game) as
    | {
        current_node_id?: string | null;
        dialog_tree?: { nodes: Record<string, DialogNode>; committed_path: string[] };
        environments?: Record<string, Environment>;
      }
    | null;

  const imagePath = resolveBackgroundPath(game);
  const url = assetUrl(imagePath);
  return (
    <div
      className="background"
      style={url ? { backgroundImage: `url("${url}")` } : undefined}
      aria-label="background"
      data-image-path={imagePath ?? undefined}
    />
  );
}

function resolveBackgroundPath(game: {
  current_node_id?: string | null;
  dialog_tree?: { nodes: Record<string, DialogNode>; committed_path: string[] };
  environments?: Record<string, Environment>;
} | null): string | null {
  if (!game?.dialog_tree || !game.environments) return null;
  const path = game.dialog_tree.committed_path ?? [];
  for (let i = path.length - 1; i >= 0; i--) {
    const id = path[i] as string;
    const node = game.dialog_tree.nodes[id];
    if (!node?.location_id) continue;
    const env = game.environments[node.location_id];
    if (env?.image_path) return env.image_path;
  }
  return null;
}
