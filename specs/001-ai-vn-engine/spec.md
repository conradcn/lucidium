# Feature Specification: AI-Driven Visual Novel Engine

**Feature Branch**: `001-ai-vn-engine`
**Created**: 2026-05-01
**Status**: Draft
**Input**: User description: "AI-driven visual novel engine: backend maintains world state, characters, and a dialog tree; AI scheduler keeps text and images ready ahead of the player; UI provides start screen, new-game interview, and a main play screen with side-panel editing of all story state."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Begin and play a brand-new AI-generated story (Priority: P1)

A player launches the application, taps "New Game," answers a short interview about setting, genre, visual style, their character, and their character's name, confirms the choices on a review screen, and is dropped into the main play view. They see a generated background, a generated character on stage, dialog text appearing with a typewriter effect, and either a set of options or a free-text input. They click an option (or type), the next dialog node is presented with updated visuals, and the story progresses. The player feels that text and images are ready when they are needed and the world stays internally consistent across turns.

**Why this priority**: This is the core value proposition. Without the ability to start a fresh AI-generated story and play it forward with characters and backgrounds appearing on cue, the product does not exist. All other stories layer onto this foundation.

**Independent Test**: From a freshly installed application with valid LLM and image-backend connections configured, a tester can press "New Game," walk through the interview using only the suggested choices, and play through at least 10 consecutive dialog nodes — encountering at least one option-driven branch and at least one free-text moment — without the screen ever sitting on a blank background or a missing character image at the moment of presentation.

**Acceptance Scenarios**:

1. **Given** a player on the Start Screen with valid backend settings and no existing saves, **When** they press "New Game," **Then** they are taken into the interview with a pregenerated character visible against a white room background.
2. **Given** the player has selected a Visual Style and is now answering the Character Description question, **When** the system has had time to work in the background, **Then** the white-room background has been replaced with a setting-appropriate background and the placeholder character has been replaced with one matching the chosen style, with no action required from the player.
3. **Given** the player has confirmed all interview answers (including any added side characters), **When** they continue, **Then** the main UI loads with the player's character on stage, a setting-appropriate background, and the first dialog node ready to read.
4. **Given** the player is reading the current dialog node and option buttons are shown, **When** they click an option, **Then** the next node's text begins displaying immediately (no perceptible wait) and any newly-entering character is on screen with at most a brief, signaled fade-in if their image is still rendering.
5. **Given** the player is at a node, **When** they type a custom action and submit, **Then** any speculatively-generated future nodes are discarded, a new continuation is generated from their input, and the resulting node respects the established world state and on-stage characters.
6. **Given** a character whose outfit, pose, or expression changes mid-scene, **When** the player advances to a node that records that change, **Then** the character's portrait updates to reflect the change without breaking visual identity (the same person, just different state).

---

### User Story 2 - Resume a previous story from the Start Screen (Priority: P2)

A returning player launches the application and sees a "Continue" button on the Start Screen because at least one save exists. They press it and are returned to the exact dialog node, world state, and character set they left from. Alternatively, they press "Load Game," browse a list of named saves, and pick one to load, rename, or delete.

**Why this priority**: Visual novels are long-form. A player who cannot pick up where they left off will not return. This is required for the product to be usable beyond a single sitting, but it depends on Story 1 producing playable sessions in the first place.

**Independent Test**: A tester can play a session from Story 1 for at least 5 nodes, exit the application, relaunch it, press "Continue," and verify they are placed at the exact node they left from with all on-stage characters, background, and world state intact. The same tester can then open "Load Game," rename the save, delete a different save, and confirm both operations persist across a relaunch.

**Acceptance Scenarios**:

1. **Given** the application has just been launched and at least one save exists, **When** the Start Screen renders, **Then** the "Continue" button is visible and enabled.
2. **Given** no saves exist, **When** the Start Screen renders, **Then** the "Continue" button is hidden.
3. **Given** the player presses "Continue," **When** the save loads, **Then** the main UI restores the same dialog node, on-stage characters with their current outfit/pose/expression, the active background, and all world-state values that were present when the save was last written.
4. **Given** the player opens "Load Game," **When** they select a save and choose "Rename" or "Delete," **Then** the change is reflected immediately and persists after closing and relaunching the application.

---

### User Story 3 - Inspect and edit live story state from the Story side panel (Priority: P2)

While playing, a player opens the "Story" side panel and uses tabs for History, World Info, Environments, Characters, Dialog Tree, and Options to inspect what the AI is tracking. Every field is editable. The player corrects a wrong fact about a character, edits a chat-history line that the AI got wrong, and renames the active environment. From that point forward, generated content respects their edits.

**Why this priority**: AI generations drift. Without the ability to correct course mid-story, a single hallucination can ruin a long playthrough. This makes the engine a *creative tool* rather than a black box, and is what differentiates the product from a static AI chat. It is P2 because the game is still playable without it for a short session.

**Independent Test**: During an active session, a tester can open the Story panel, change a character's name in the Characters tab, change a line of dialog in the History tab, and edit a world-info field. After advancing two more nodes, the tester verifies that the AI's new dialog and any character-descriptor refresh reflect those edits (e.g., the renamed character is referred to by the new name, the corrected fact is consistent with the edited value).

**Acceptance Scenarios**:

1. **Given** the player is in the main UI, **When** they open the Story side panel, **Then** they see tabs for History, World Info, Environments, Characters, Dialog Tree, and Options, with the active environment visually highlighted in Environments.
2. **Given** the player edits a character's Description in the Characters tab and closes the panel, **When** the next node is generated, **Then** the new dialog and any future character-state refresh draw on the edited Description.
3. **Given** the player edits a line in the History tab, **When** they close the panel and continue, **Then** the edited history is what the system uses as conversation context for the next generation.
4. **Given** the player edits any field on any tab, **When** they navigate away from that tab and return, **Then** their edit is still present (changes persist within the session and into saves).

---

### User Story 4 - Configure backend connections and presentation (Priority: P3)

A player opens the Options/Settings screen, points the engine at their preferred LLM endpoint (defaulting to OpenRouter with a default model) and image endpoint (defaulting to a local ComfyUI instance), and adjusts the typewriter speed. The engine validates that the connections work and uses them for all subsequent generation.

**Why this priority**: The default configuration covers the common case (OpenRouter + local ComfyUI), so most players never touch this. But power users with their own LLMs, custom image backends, or accessibility needs (typewriter pacing) require it. P3 because the product is fully playable for the default audience without ever opening this screen.

**Independent Test**: A tester can open Settings, change the LLM base URL to an alternative OpenAI-compatible endpoint, change the image backend port, and confirm that the next generated dialog and the next generated image used the new endpoints (e.g., by hitting an endpoint that returns a recognizable signature). They can also drag the typewriter-speed slider to fastest and slowest and confirm the visible difference on the next node.

**Acceptance Scenarios**:

1. **Given** the player opens Settings on first run, **When** the screen renders, **Then** the LLM connection defaults to an OpenRouter-style endpoint with the default model preselected and the image connection defaults to a local ComfyUI on port 8000.
2. **Given** the player changes the LLM endpoint and saves, **When** the next text generation happens, **Then** the request goes to the new endpoint.
3. **Given** the player adjusts typewriter speed, **When** the next dialog node is presented, **Then** characters appear at the new speed.

---

### User Story 5 - Add custom side characters during the new-game interview (Priority: P3)

On the interview confirmation screen, a player adds one or more side characters by typing a one-line description (for example, "a gruff retired bounty hunter who runs the local bar"). The engine fleshes each one into a full character record (description, appearance attributes, seed) and they become available to enter scenes when the story calls for them.

**Why this priority**: This is a meaningful authoring affordance — the player gets a head start on populating the world they want — but the engine can introduce characters mid-story on its own. P3 because Story 1 already supports a viable session without it.

**Independent Test**: During the interview confirmation step, a tester adds two one-line side-character descriptions and proceeds. Within the first 20 dialog nodes, those two characters appear with full visual portraits consistent with the descriptions, distinct from each other, and distinct from the player character.

**Acceptance Scenarios**:

1. **Given** the player is on the confirmation screen, **When** they add a one-line side-character description and continue, **Then** a full character record (with attributes and seed) is created before the main UI loads.
2. **Given** a side character was created at interview time, **When** the story brings them on stage, **Then** their portrait is consistent on every reappearance (same identity, even when pose, outfit, or expression change).

---

### Edge Cases

- **Image still rendering at presentation time**: When the player advances to a node whose background or character image is not yet ready, the system MUST present a graceful fallback (the most recent generated background, a placeholder for the character with the name tag visible) and swap in the real asset as soon as it is ready, without blocking the dialog text from being read.
- **LLM text not yet ready at advance time**: If the player advances to a node whose text generation has not completed, the system MUST signal a brief wait (do not block indefinitely without feedback) and resume the typewriter as soon as text arrives. This should be rare under normal play because the scheduler runs ahead.
- **Free-text input invalidates speculation**: When the player submits free text, every speculatively-generated downstream node MUST be discarded; image work that is purely about a character or environment that is still valid (still on-stage, unchanged) MAY be retained; in-progress generations should follow the obsolescence rule (drop if obsoleted, keep if still applicable).
- **Character descriptor missing fields**: When a character record is created without all schema fields filled, the LLM scheduler MUST queue a low-priority repair task to fill the missing fields from current information, and this repair MUST happen before the character is shown on stage.
- **Backend unreachable**: If the LLM or image endpoint becomes unreachable during play, the player MUST be shown a clear, non-fatal message and given the option to open Settings to fix the connection; an autosave/checkpoint of current state MUST be preserved so no progress is lost.
- **Conflicting edits in the side panel**: If the player edits a field that is currently being used by an in-flight generation, the in-flight result MUST be discarded once it returns and a fresh generation with the edited value MUST take its place.
- **Plot-thread starvation**: When the active plot threads list grows or shrinks, the world-state refresh MUST decide which threads to drop and may reintroduce dropped threads during a lull. The player MUST be able to see (and edit) the current active and dropped lists in the World Info tab.
- **Save during active generation**: If the player saves or exits while text/image work is in flight, the save MUST capture the committed state only; speculative work in progress is discarded.
- **Exit cleans up the server**: Pressing Exit on the Start Screen MUST shut down both the UI window and any backend processes the application started, releasing the configured ports.

## Requirements *(mandatory)*

### Functional Requirements

#### Backend & Communication

- **FR-001**: System MUST maintain a persistent world state per game that includes Game Name (LLM-generated, editable), Overall Plot Direction, Player Intent forecast, Active Plot Threads, and Dropped Plot Threads.
- **FR-002**: System MUST maintain a persistent character state per character that includes Name, Description, Gender, Age (stored as an exact integer), Ethnicity, Skin, Hair Color, Hairstyle, Eye Color, Build, Bust, Outfit, Pose, Expression, Facts (summarizer-maintained), Images, and a Seed that is generated once at character creation and never changes for the life of the character.
- **FR-003**: System MUST maintain a dialog tree where each node carries Text, Speaker, characters who entered/left at this node, character descriptors for any newly-introduced character, Location (nullable when unchanged), Location Prompt (nullable when unchanged or reused), Options (empty implies a single "Continue"), and per-character changes (with priority on pose, expression, and outfit).
- **FR-004**: System MUST stream world-state and dialog-tree updates to the UI over a websocket connection so that changes made on the backend (text arriving, images completing, world refresh) are reflected in the UI without polling.
- **FR-005**: System MUST transform a character's exact integer Age into a coarse age band (e.g., "twenty," "thirty") via deterministic rules at the moment an image prompt is constructed, while preserving the exact value in storage.
- **FR-005a**: When the configured ComfyUI portrait checkpoint is detected as a Pony-based Stable Diffusion model (model filename contains the substring "pony", case-insensitive), all generated portrait and background prompts MUST be prefixed with the conventional pony scoring tags (`score_9, score_8_up, score_7_up, score_6_up`) and corresponding negative scoring tags appended to negative prompts. Detection MUST occur per-workflow at instantiation time, not per-call.

#### LLM Scheduler

- **FR-006**: The LLM scheduler MUST prioritize work in this order: (1) generate text for upcoming dialog nodes, (2) repair malformed character descriptors for characters who will appear in upcoming nodes, (3) refresh the world state (plot direction, player-intent forecast, plot-thread maintenance, pruning of redundant character/world facts).
- **FR-007**: The LLM scheduler MUST run multiple LLM calls in parallel when work is independent.
- **FR-008**: World-state refresh MUST weight text the player typed (free-text input and side-panel edits) more strongly than text the LLM generated when forecasting Player Intent.
- **FR-009**: The system MUST provide a "stay focused" or "reintroduce thread" steering signal from the world-state refresh into the prompt used for the next text generation.
- **FR-010**: When constructing a text-generation prompt, the system MUST include the conversation history clamped to a configurable character budget, the full attribute set of every on-stage character, the base description (only) of every off-stage character, and the summarizer's current assessment.
- **FR-010a**: The text-generation response MUST be a list of beats. Each beat becomes its own dialog node so that metadata (speaker, character pose/expression/outfit, location) can change between beats. Each beat's text MUST be a single line — no embedded newlines, no carriage returns. The renderer presents one beat at a time; the player advances through them via Continue (or, on the last beat, via the option buttons). Subsequent walks within a chain MUST NOT trigger another LLM call — the chain is pre-generated and walked from the dialog tree.
- **FR-010b**: When the "mature content" setting is enabled, an explicit allowance directive MUST be appended to the system prompt of every text-generation, world-init, and world-refresh call so the LLM does not soften adult or violent content beyond what the narrative calls for.

#### Image Scheduler

- **FR-011**: The image scheduler MUST prioritize work in this order: (1) backgrounds for upcoming nodes, working outward from the current node, (2) portraits for characters who lack one, working outward from the current node, (3) speculative regenerations for character prompt changes starting at the current node.
- **FR-012**: The image scheduler MUST run multiple image generations in parallel when work is independent.
- **FR-013**: When a queued image task becomes obsolete (the underlying background or character has changed in-game since the task was queued), the scheduler MUST discard it before it starts; if it is already in flight, the scheduler MUST allow it to finish and apply the result only if it is still more current than the image it would replace.
- **FR-014**: The image scheduler MUST sit idle only when no qualifying background, missing-portrait, or speculative-regeneration work remains.
- **FR-015**: Character portraits MUST use the character's stored Seed for visual identity consistency across regenerations; only attributes that changed should drive visual change.

#### Free-Text Invalidation

- **FR-016**: When the player submits a free-text input, the system MUST discard all speculatively-generated future dialog nodes and any pending text generations whose context they depended on.
- **FR-017**: The system MUST retain image work that remains valid after a free-text invalidation (i.e., for characters and environments whose state is unchanged), and MUST cancel or discard image work tied to invalidated nodes per the obsolescence rule (FR-013).

#### Start Screen & Save Management

- **FR-018**: The Start Screen MUST show buttons for Continue, New Game, Load Game, Options, and Exit, where Continue is hidden when no saves exist.
- **FR-019**: Continue MUST resume the most recently played save.
- **FR-020**: Load Game MUST list all saves and provide create-from-current, rename, and delete operations, with all changes persisted across application launches.
- **FR-021**: Exit MUST gracefully shut down both the UI and any backend process the application started.

#### New Game Interview

- **FR-022**: The New Game interview MUST visually start with a pregenerated "dream guide" character in a white room. Both images are bundled with the engine (produced once via ComfyUI and shipped as static assets) so the interview is visually furnished from the first frame, before any LLM or image-generation call has run.
- **FR-023**: The interview MUST ask, in order: Setting, Genre, Visual Style, Character Description, Name, then a Confirmation review.
- **FR-024**: For Setting, the engine MUST present 5 random options drawn from a hard-coded list of 30 "intelligent defaults" bundled with the engine, plus a free-text option. No LLM round trip is required for this step.
- **FR-025**: For Genre, the engine MUST present a hard-coded list of "intelligent defaults" bundled with the engine, in full (no random sub-sampling), plus a free-text option. No LLM round trip is required for this step.
- **FR-026**: For Visual Style, the system MUST generate intelligent default options with the same surface-5-of-30 randomization as Setting, plus a free-text option.
- **FR-027**: As soon as the player has answered the Setting step (i.e., when the Genre step is presented to the user), the engine MUST start the Step-4 (Character Description) LLM call in the background. The step-4 prompt depends only on the chosen setting; the prefetched options are awaited when the user reaches Step 4.
- **FR-027a**: As soon as the player has answered the Visual Style step, the engine MUST regenerate the dream-guide character image and the white-room background in the chosen visual style via ComfyUI. The new images replace the bundled placeholders for the remaining interview steps. These regenerations MUST run in the background and not block the player's interaction.
- **FR-028**: For Character Description, the system MUST present LLM-generated default options and a free-text input.
- **FR-029**: For Name, the system MUST present LLM-generated default options and a free-text input.
- **FR-030**: The Confirmation step MUST display all answers as editable fields and MUST provide a section where the player can add additional side characters, each via a single one-line description.
- **FR-031**: When the player confirms, the system MUST generate the full opening background and world information (including Game Name, Overall Plot Direction, initial Active Plot Threads) and then transition to the main UI.
- **FR-031a**: After confirmation and before the main UI is shown, the engine MUST display a loading state that persists until ALL of the following are ready: (a) the opening dialog node's text, (b) every on-stage character's portrait image, and (c) the opening environment's background image. Only when all three are present does the renderer transition to the playable Main UI.

#### Settings

- **FR-032**: Settings MUST expose configuration for the LLM backend (OpenAI-compatible endpoint, with defaults pointing at OpenRouter using a default model), the image backend (defaulting to a local ComfyUI on port 8000), and a typewriter-effect speed control.
- **FR-032a**: Settings MUST include a "mature content" toggle. When enabled, the engine appends a description of mature-content allowance to the system prompt of every text-generation, world-init, and world-refresh LLM call (per FR-010b). Default: disabled.
- **FR-033**: Settings changes MUST take effect for the next generation without requiring a restart.

#### Main UI

- **FR-034**: The main UI MUST present, simultaneously: a Background, the on-stage Characters (each with a name tag and a full-body portrait), and an Interaction Panel showing dialog text, choice buttons (when present), and a free-text input box.
- **FR-034a**: The player character MUST NOT be rendered as an on-stage character in the Main UI. The player is the lens through which the story is experienced, not a stage actor. The player character still exists in the character roster and appears in the Story side panel's Characters tab and in dialog speaker references.
- **FR-035**: The displayed Background MUST be the background attached to the current dialog node; if that background is not yet ready, the most recent prior dialog node's generated background MUST be shown instead.
- **FR-036**: The main UI MUST expose unobtrusive top-of-screen affordances for "Story" and "Menu."
- **FR-037**: The Story side panel MUST contain tabs for History, World Info, Environments, Characters, Dialog Tree, and Options, where every field on every tab is editable.
- **FR-038**: The Environments tab MUST visually highlight the currently active environment among all generated environments.
- **FR-039**: Edits made in any Story tab MUST persist into the current save and MUST be the values used for subsequent generation.
- **FR-040**: Dialog text MUST appear with a typewriter effect at the speed configured in Settings.

### Key Entities *(include if feature involves data)*

- **Game (Save)**: A single playthrough. Carries the world state, the entire dialog tree, the character roster, the generated environments, the configured settings snapshot, the current node pointer, and metadata (name, last-played timestamp).
- **World State**: Per-game container of Game Name, Overall Plot Direction, Player Intent forecast, Active Plot Threads, Dropped Plot Threads.
- **Character**: A persistent identity in the game. Holds Name, Description, Gender, Age (integer), Ethnicity, Skin, Hair Color, Hairstyle, Eye Color, Build, Bust, Outfit, Pose, Expression, Facts (summarizer-maintained), Images, and an immutable Seed.
- **Dialog Node**: A single beat in the story. Holds Text, Speaker, entering/leaving characters, descriptors for newly-introduced characters, Location (nullable), Location Prompt (nullable), Options (empty for "Continue"), and per-character changes.
- **Environment**: A generated background tied to a Location. Reusable across nodes that share the same Location Prompt.
- **Generation Task (LLM or Image)**: A scheduled unit of work. Knows its priority class, the assumptions it depends on (which node, which character version), and whether it is queued, in-flight, or complete. The scheduler uses these to detect obsolescence.
- **Settings**: Per-installation configuration for LLM and image backends and presentation preferences (typewriter speed). Snapshotted into each save at creation.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A new player can complete the New Game interview and reach the first playable node in under three minutes on default backends.
- **SC-002**: During a typical play session, in 95% of advances the dialog text is ready at the moment the player advances (no visible wait beyond the typewriter pacing).
- **SC-003**: During a typical play session, in 90% of advances the background and on-stage character portraits are ready at the moment the player advances; in the remaining 10%, the fallback (prior background, placeholder portrait) is shown and the real asset is swapped in within five seconds.
- **SC-004**: A character that has appeared at least twice is visually recognizable as the same person across appearances in 95% of cases (subjective evaluation by a tester reviewing portrait pairs).
- **SC-005**: After a player edits a character attribute or a history line in the Story panel, the next generated dialog node reflects the edit in 100% of cases.
- **SC-006**: A player who quits and relaunches the application can resume their most recent session via Continue with the exact same on-screen state — node, characters, background, world values — in 100% of cases.
- **SC-007**: When a player submits free text, the next dialog node generated from that input arrives within five seconds in the median case and within fifteen seconds in the 95th percentile.
- **SC-008**: When a player advances rapidly through a stretch of pre-generated nodes, the image scheduler keeps up such that no more than one consecutive node falls back to a placeholder portrait or a stale background.
- **SC-009**: When a node introduces a new character, that character is on screen with a portrait (real or placeholder with name tag) at the moment the dialog text begins, in 100% of cases.
- **SC-010**: A returning power user can change the LLM endpoint, image endpoint, or model in Settings and have the next generation use the new value without restarting the application.

## Assumptions

- The application is a single-player desktop application. Multiplayer, cloud sync, and hosted/multi-tenant deployment are out of scope.
- The application bundles a backend process that the UI starts and stops; "Exit" terminates both. There is no separate long-lived server.
- The default LLM provider is OpenRouter using a default model (e.g., a Qwen-class model). Any OpenAI-compatible HTTP endpoint is acceptable as a substitute via Settings.
- The default image provider is the embedded backend: diffusers plus an SDXL-family checkpoint running inside the engine process, with no external server. A local ComfyUI instance is an optional substitute selected via Settings (`image.backend = "comfyui"`).
- LLM and image work require network reachability for the chosen backends; offline play is not expected to work unless both backends are local.
- Saves are stored locally on the player's machine. Cloud save and cross-device sync are out of scope.
- Saves include their settings snapshot; loading an old save uses the snapshot's endpoints unless the player overrides them.
- The intended audience is adult players: the character schema includes attributes (e.g., Bust) that imply mature/adult themes. Mature content is opt-in per save and OFF by default. The engine does NOT rely on the configured LLM and image backends for safety — it ships its own layered safeguards for the under-18 + sexual content case: a prompt-side age floor that clamps every age sent to the image pipeline to 18, a storage-side age correction with an LLM retcon pass, and an always-on output-side ML content filter that cannot be disabled from the Settings UI. See `SAFETY.md` for the full policy and the code paths implementing it.
- Speculative work is best-effort. The scheduler may discard speculative results aggressively rather than waiting for them; the player's experience is the priority.
- A reasonable cap on simultaneously on-stage characters (e.g., four) is acceptable for v1; the data model does not impose one, but the UI layout assumes a small number.
- The conversation-history clamp on prompt construction is a configurable character count, not a token count.
- The application persists state continuously enough that an unexpected exit does not lose more than the current speculative work; committed nodes and edits survive.
- The Options/Settings screen shown on the Start Screen and inside the main UI's side panel is the same screen — settings are global per installation, not per save.
