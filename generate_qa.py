#!/usr/bin/env python3
"""Generate synthetic DIY home repair Q&A pairs using the Claude API."""

import argparse
import csv
import json
import sys
import uuid

import anthropic

# Substantial system prompt — must exceed Sonnet 4.6's 2048-token cache minimum.
SYSTEM_PROMPT = """You are an expert DIY home repair educator with 30+ years of hands-on experience
helping homeowners tackle repairs, maintenance, and improvements safely and effectively. Your role is
to generate high-quality, accurate question-and-answer pairs for a comprehensive DIY home repair
knowledge base used by beginners and experienced DIYers alike.

════════════════════════════════════════
QUALITY STANDARDS
════════════════════════════════════════

Questions:
- Must reflect realistic situations homeowners actually encounter
- Should be specific enough to be searchable and useful
- Avoid overly academic or textbook-style phrasing
- Frame from the homeowner's perspective ("How do I…", "Why is my…", "What causes…")

Answers:
- Must be accurate, safe, and actionable
- Lead with the most important information or safety warning if one exists
- Provide step-by-step guidance where appropriate
- Mention when to call a licensed professional (especially for structural, electrical, or gas work)
- Include tool and material recommendations where helpful
- Keep answers between 100–250 words — detailed but not overwhelming

Tags:
- 3 to 6 per pair
- Use lowercase, hyphenated strings (e.g., "water-damage", "circuit-breaker")
- Cover the specific action, the component involved, and the skill level implied

════════════════════════════════════════
TOPIC REFERENCE GUIDE
════════════════════════════════════════

PLUMBING
Common questions and subtopics:
• Fixing dripping or running faucets (worn washers, cartridges, O-rings)
• Unclogging sinks, tubs, showers, and toilets (plunger, snake, enzyme cleaners)
• Replacing toilet internals: fill valve, flapper, flush valve, wax ring
• Water heater maintenance: anode rod, sediment flushing, temperature adjustment
• Fixing slow drains and understanding P-trap function
• Shutting off water supply: main shutoff, fixture shutoff valves
• Pipe sweating and condensation vs. active leaks
• Winterizing outdoor spigots and preventing frozen pipes
• Installing or replacing supply lines, shut-off valves, and angle stops
• Low water pressure causes and basic diagnostics
• Sump pump testing and maintenance
• Water softener salt and maintenance basics

Safety notes for plumbing:
- Always turn off the water supply before disassembling any fixture
- Label hot and cold when reassembling
- Use thread seal tape (Teflon) on threaded fittings, not on compression fittings
- Know the location of your home's main water shutoff

ELECTRICAL
Common questions and subtopics:
• Replacing standard outlets (duplex receptacles) and light switches
• GFCI outlet installation and testing (bathrooms, kitchens, garages, outdoors)
• AFCI breakers and when they're required
• Resetting tripped circuit breakers and identifying overloaded circuits
• Installing or replacing ceiling light fixtures and chandeliers
• Ceiling fan installation and wiring (with/without existing switch leg)
• Understanding wire colors: black (hot), white (neutral), green/bare (ground)
• Outlet tester usage and reading wiring faults
• Replacing a dimmer switch and compatibility with LED bulbs
• USB outlet and smart outlet installation
• Reading and understanding your electrical panel directory
• Extension cord safety and proper gauge selection
• Outdoor outlet covers and weatherproof box installation

Safety notes for electrical:
- ALWAYS turn off the breaker and verify with a non-contact voltage tester before touching wires
- Never work on live circuits
- Emphasize when to call a licensed electrician: panel work, new circuits, aluminum wiring, knob-and-tube
- GFCI protection is required within 6 feet of water sources

CARPENTRY
Common questions and subtopics:
• Fixing squeaky floors: locating joists, screw-from-above vs. screw-from-below methods
• Stopping squeaky doors: hinge adjustment, strike plate realignment, planing
• Repairing or replacing interior door hinges, knobs, and deadbolts
• Installing or adjusting door threshold and weather stripping
• Patching or replacing damaged baseboards and door/window casing
• Sticking drawers and cabinet door adjustments
• Deck maintenance: board replacement, screw popping, post and joist inspection
• Wood rot identification and treatment vs. replacement
• Building basic floating shelves with proper wall anchoring
• Cutting and installing crown molding basics
• Stair tread repair and tightening loose balusters
• Wood filler vs. wood epoxy for rot and damage repair

PAINTING
Common questions and subtopics:
• Proper surface prep: cleaning, sanding, filling holes, priming
• Choosing the right sheen: flat/matte, eggshell, satin, semi-gloss, gloss
• Interior vs. exterior paint differences and when each is required
• Cutting in along ceilings and trim without tape vs. with painter's tape
• Rolling technique: W or M pattern, proper nap thickness for surface texture
• Brush selection: natural bristle (oil-based) vs. synthetic (latex/water-based)
• Painting over dark colors or stains: proper primer selection
• Cleaning and storing brushes and rollers between coats and after finishing
• Color matching: using existing paint chips, spectrophotometer services
• Fixing common paint problems: drips, brush marks, lap marks, roller texture
• Painting trim and doors: order of operations, preventing sticking
• Spray painting basics: thinning, distance, overlap patterns
• Lead paint precautions for pre-1978 homes

DRYWALL
Common questions and subtopics:
• Patching nail holes and small dings with spackling compound
• Patching medium holes (1–4 inches) with California patch or mesh patch kits
• Patching large holes (4+ inches) with backer board or clip method
• Feathering joint compound for smooth, invisible repairs
• Taping seams: paper tape vs. fiberglass mesh tape (when to use each)
• Applying multiple thin coats of joint compound (mud) vs. one thick coat
• Sanding drywall: grits, tools, minimizing dust
• Priming repaired drywall before painting (critical step many skip)
• Matching existing wall texture: orange peel, knockdown, skip trowel, popcorn
• Water damage: drying thoroughly before patching, treating for mold
• Hanging new drywall: measuring, scoring and snapping, fastener placement
• Corner bead installation for inside and outside corners

════════════════════════════════════════
DIFFICULTY LEVEL DEFINITIONS
════════════════════════════════════════

easy:
- Requires only basic tools (screwdriver, pliers, hammer)
- No specialized knowledge or skills required
- Can be completed in under 1 hour
- Failure risk is low; mistakes are easily corrected
- Examples: replacing an outlet cover, patching a nail hole, unclogging a drain with a plunger

medium:
- Requires some tools beyond the basics (drill, utility knife, level)
- Requires understanding of basic home systems
- Typically 1–4 hours to complete
- Some risk of error; mistakes may require rework but are recoverable
- Examples: replacing a toilet flapper + fill valve, installing a ceiling fan, patching a 3-inch drywall hole

hard:
- Requires specialized tools or significant physical effort
- Requires solid understanding of building systems or prior DIY experience
- 4+ hours or multi-day projects
- Errors can cause damage, injury, or require professional correction
- Often involves safety-critical systems (electrical panels, structural elements, gas lines)
- Examples: replacing a wax ring and resetting a toilet, running a new electrical circuit, installing a door pre-hung in rough opening

════════════════════════════════════════
REQUIRED OUTPUT FORMAT
════════════════════════════════════════

Return ONLY a valid JSON array. No markdown fences, no preamble, no explanation.
Each element must contain exactly these fields:

{
  "id": "<uuid-v4-string>",
  "topic": "<one of: plumbing | electrical | carpentry | painting | drywall>",
  "difficulty": "<one of: easy | medium | hard>",
  "question": "<the question text>",
  "answer": "<the detailed answer, 100-250 words>",
  "tags": ["<tag1>", "<tag2>", "<tag3>"]
}

Do not include any text before or after the JSON array."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic DIY home repair Q&A pairs using Claude API",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--count", type=int, default=20,
        help="Total number of Q&A pairs to generate",
    )
    parser.add_argument(
        "--topics",
        default="plumbing,electrical,carpentry,painting,drywall",
        help="Comma-separated list of topics to include",
    )
    parser.add_argument(
        "--difficulty", choices=["easy", "medium", "hard", "mixed"], default="mixed",
        help="Difficulty level for generated pairs",
    )
    parser.add_argument(
        "--output", default="qa_data.json",
        help="Output file path",
    )
    parser.add_argument(
        "--format", choices=["json", "csv"], default="json", dest="output_format",
        help="Output file format",
    )
    return parser.parse_args()


def build_user_prompt(topics: list[str], difficulty: str, batch_size: int, offset: int) -> str:
    topic_str = ", ".join(topics)

    if difficulty == "mixed":
        diff_instruction = (
            "Distribute difficulty levels across easy, medium, and hard. "
            "Aim for a roughly even split across the batch."
        )
    else:
        diff_instruction = f"All pairs must be {difficulty} difficulty."

    return (
        f"Generate exactly {batch_size} DIY home repair Q&A pairs.\n\n"
        f"Topics to draw from: {topic_str}\n"
        f"Difficulty: {diff_instruction}\n\n"
        f"Spread the pairs across the available topics. "
        f"These are items {offset + 1}–{offset + batch_size} in a larger dataset, "
        f"so avoid repeating questions from common examples.\n\n"
        f"Return only the JSON array with exactly {batch_size} objects."
    )


def generate_batch(
    client: anthropic.Anthropic,
    topics: list[str],
    difficulty: str,
    batch_size: int,
    offset: int,
) -> list[dict]:
    user_prompt = build_user_prompt(topics, difficulty, batch_size, offset)

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_prompt}],
    )

    raw = next((b.text for b in response.content if b.type == "text"), "").strip()

    # Strip accidental markdown fences if the model wraps the output
    if raw.startswith("```"):
        lines = raw.splitlines()
        start = 1
        end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
        raw = "\n".join(lines[start:end]).strip()

    pairs = json.loads(raw)

    # Guarantee each pair has a valid UUID-style id
    for pair in pairs:
        if not isinstance(pair.get("id"), str) or len(pair["id"]) < 8:
            pair["id"] = str(uuid.uuid4())

    return pairs, response.usage


def save_json(pairs: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(pairs, fh, indent=2, ensure_ascii=False)


def save_csv(pairs: list[dict], path: str) -> None:
    fieldnames = ["id", "topic", "difficulty", "question", "answer", "tags"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for pair in pairs:
            row = {**pair, "tags": "|".join(pair.get("tags", []))}
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    topics = [t.strip() for t in args.topics.split(",") if t.strip()]

    if not topics:
        print("Error: --topics must include at least one topic.", file=sys.stderr)
        sys.exit(1)

    client = anthropic.Anthropic()

    batch_size = 5
    total_batches = (args.count + batch_size - 1) // batch_size

    print(f"Generating {args.count} Q&A pairs in {total_batches} batch(es)")
    print(f"  Topics    : {', '.join(topics)}")
    print(f"  Difficulty: {args.difficulty}")
    print(f"  Output    : {args.output} ({args.output_format})")
    print()

    all_pairs: list[dict] = []
    total_cached = 0
    total_uncached = 0

    for batch_num in range(total_batches):
        remaining = args.count - len(all_pairs)
        this_batch = min(batch_size, remaining)

        print(
            f"  Batch {batch_num + 1}/{total_batches} ({this_batch} pairs)...",
            end=" ",
            flush=True,
        )

        try:
            pairs, usage = generate_batch(client, topics, args.difficulty, this_batch, len(all_pairs))
        except json.JSONDecodeError as exc:
            print(f"FAILED — JSON parse error: {exc}", file=sys.stderr)
            sys.exit(1)
        except anthropic.APIError as exc:
            print(f"FAILED — API error: {exc}", file=sys.stderr)
            sys.exit(1)

        all_pairs.extend(pairs)
        cached = getattr(usage, "cache_read_input_tokens", 0) or 0
        created = getattr(usage, "cache_creation_input_tokens", 0) or 0
        total_cached += cached
        total_uncached += getattr(usage, "input_tokens", 0) or 0

        cache_note = ""
        if created:
            cache_note = " [cache written]"
        elif cached:
            cache_note = f" [cache hit: {cached} tokens]"

        print(f"done  (total: {len(all_pairs)}){cache_note}")

    print()

    if args.output_format == "json":
        save_json(all_pairs, args.output)
    else:
        save_csv(all_pairs, args.output)

    print(f"Saved {len(all_pairs)} pairs → {args.output}")
    if total_cached:
        print(f"Cache stats: {total_cached} cached tokens, {total_uncached} uncached tokens")


if __name__ == "__main__":
    main()
