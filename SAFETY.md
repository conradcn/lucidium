# Safety Policy

Lucidium is a single-player AI-driven visual novel engine. This
document describes what the engine does and does not allow, the
technical safeguards in place, and how to report concerns.

## Scope of generated content

Lucidium can produce mature content — explicit sexuality, graphic
violence, profanity, morally complex situations — when the player
opts in via Settings → Mature Content. The opt-in is OFF by default
and applies only within a single save.

Mature mode does **not** unlock, and the engine actively prevents:

- **Sexual content involving minors.** Any character described or
  visualised as under 18 in a sexual context is blocked. This is a
  hard policy with no opt-out.
- **Real, identifiable people.** The engine does not provide tools
  for likeness-based generation of real public or private individuals.
- **Content that violates the underlying model licenses** that the
  player loads (Stable Diffusion XL, Pony Diffusion, etc.). The
  engine inherits those licenses' prohibitions.

If you find a way to produce content in any of these categories
through the shipped engine, that is a bug. See "Reporting" below.

## Technical safeguards

The engine layers several defenses for the under-18 + sexual content
case. None is sufficient on its own; they exist to make the failure
mode require explicit user effort to reach.

1. **Prompt-level age floor.** Every age sent to the image pipeline
   is clamped to 18. Stored character ages are unchanged (so
   narration round-trips faithfully); only the prompt-side helper
   raises sub-18 ages. The words `child` and `teenage`
   are not in the prompt vocabulary at all — the floor word is
   `eighteen`, and every band above it is a decade word
   (`twenty`, `thirty`, … `hundred`). This prevents the most direct prompt-driven
   route to a visually-young rendering.
   See `backend/src/lucidium/domain/character.py::age_band`.

2. **Storage-side age correction.** When the engine detects that a
   stored under-18 character has been described with a nudity
   outfit, it bumps the stored age to 18, surfaces a notice modal
   explaining the correction, and runs an LLM retcon pass to
   rewrite past narration that referenced the prior age. This
   makes the storage and the rendered output agree.
   See `backend/src/lucidium/orchestration/assets.py::check_minor_nudity_and_correct`.

3. **Output-side ML content filter.** Every generated portrait is
   classified by an open-source nudity detector. When nudity is
   detected, a face-age estimator runs on detected faces; if any
   face age estimate is below the threshold, the image is
   rejected, the engine retries with a clothed prompt, and (on
   second failure) falls back to an empty portrait slot. The
   filter is on by default and cannot be disabled through the
   Settings UI.
   See `backend/src/lucidium/orchestration/content_filter.py`.

4. **Mature mode is per-save, never global.** A new save defaults
   to mature mode OFF. The Settings checkbox shows an inline
   warning that links to this document.

5. **No bundled image-generation weights.** The Lucidium installer
   does not ship Stable Diffusion XL, Pony Diffusion, or any other
   image-generation checkpoint. Players obtain weights themselves
   from the upstream sources (HuggingFace, Civitai, etc.) and accept
   those licenses directly. The engine surfaces download
   instructions when the models directory is empty.

The U2NET background-removal ONNX model IS bundled; it is small,
permissively licensed, and used purely for compositing transparent
character cut-outs over backgrounds. It does not generate content.

The content filter's own models are bundled too — NudeNet's detector
(shipped inside its wheel) and the detection + age-estimation heads of
insightface's `buffalo_l` pack. They ship in the installer rather than
downloading on first render, so the filter in §3 is active offline and
on first launch. Neither generates content. If you build the app
yourself, the packaging scripts install `backend[safety]` and warm
these models; a build made without them logs an error and shows the
player a notice on the first render, because a silently-inert filter
would make §3 untrue.

## What the engine cannot guarantee

- The engine has no insight into the training data of the model
  weights the player loads. A checkpoint trained on harmful data
  remains capable of producing harmful output regardless of the
  prompt; the output filter exists precisely because we cannot
  rely on the model alone.
- Free-text fields (character description, outfit, narration) are
  user-driven. A determined user can bypass structured-field guards
  by writing harmful text directly. The output-side filter is the
  backstop for this case.
- ML classifiers have false positives and false negatives. The
  filter is calibrated conservatively (more false positives than
  false negatives) but is not 100 % accurate.
- Lucidium does not perform image search to detect likeness of
  real people; if you suspect a generated image resembles a
  real person, do not save or share it.

## Player responsibilities

By using Lucidium you agree:

- You are 18 or older.
- You will not deliberately attempt to bypass the safeguards above.
- You will not share generated content depicting any person in a
  way they have not consented to.
- You accept the licenses of the model weights you choose to load
  (Stability AI, Pony Diffusion, etc.) and will not violate them.

## Distributor responsibilities

If you fork, redistribute, or self-host Lucidium:

- Keep the safeguards intact. Removing the prompt-level floor, the
  storage-side correction, or the output-side filter is out of
  scope for forks intended for end-user distribution.
- Do not bundle image-generation weights with your distribution.
- Keep the SAFETY.md (this file) and the mature-content opt-in
  notice present and unmodified or replace with equally clear
  language.

## Reporting

If you find a safeguard bypass, a false-negative on the output
filter, or content the engine should refuse but does not:

- Open a GitHub issue tagged `safety` at the project's repository.
- For sensitive reports, email the maintainer directly at
  conradcn@gmail.com.

If you encounter generated content that depicts the sexual abuse
of a child, regardless of source:

- In the United States, report it to the National Center for
  Missing & Exploited Children's CyberTipline at
  https://report.cybertip.org/.
- In the United Kingdom, report it to the Internet Watch
  Foundation at https://www.iwf.org.uk/.
- Elsewhere, locate your country's INHOPE-affiliated hotline at
  https://inhope.org/.

These reports are confidential. The maintainers will cooperate
with lawful requests from the relevant authorities.

## Legal disclaimer

Lucidium is open-source software. The maintainers provide it
"as-is" without warranty of any kind. Use of this engine is at
your own risk and subject to your local law. The safeguards in
this document are best-effort technical measures, not a guarantee
of legal compliance in your jurisdiction.

If you are unsure whether your use of Lucidium is lawful where
you live, do not use it.
