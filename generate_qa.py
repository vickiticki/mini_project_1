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
- Keep answers between 750–1000 words — thorough and detailed

════════════════════════════════════════
TOPIC REFERENCE GUIDE
════════════════════════════════════════

APPLIANCE REPAIR
Common questions and subtopics:
• Refrigerator not cooling: condenser coil cleaning, door seal/gasket checks, thermostat settings
• Dishwasher not draining or not cleaning well: filter cleaning, spray arm clogs, drain hose kinks
• Washing machine not spinning or draining: lid/door switch, drive belt, pump filter
• Dryer not heating or taking too long: lint trap, vent hose blockages, thermal fuse
• Garbage disposal jammed or humming: reset button, clearing with an Allen wrench
• Oven or stove heating element and igniter troubleshooting
• Microwave not heating: door switch and interlock checks (leave magnetron/capacitor work to a pro)
• Ice maker not producing ice: water line kinks, fill valve issues
• Range hood fan not working: motor cleaning, filter replacement
• Refrigerator and under-sink water filter replacement
• Reading and looking up appliance error codes
• Rule-of-thumb guidance for repair vs. replace decisions

Safety notes for appliance repair:
- Always unplug the appliance (or shut off its dedicated circuit) before opening any panel
- Never bypass safety interlocks like washer lid switches or dryer door switches
- Gas appliances: never attempt gas line or gas valve work yourself — call a licensed technician
- Microwaves store dangerous high voltage in the capacitor even when unplugged — leave that repair to a pro

PLUMBING REPAIR
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

ELECTRICAL REPAIR
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

HVAC MAINTENANCE
Common questions and subtopics:
• Changing or cleaning air filters: recommended frequency, MERV rating tradeoffs
• Thermostat troubleshooting: dead batteries, miswiring, smart thermostat compatibility
• AC not cooling: dirty condenser/evaporator coils vs. low refrigerant (refrigerant is pro-only)
• Furnace not igniting: pilot light relighting, ignitor and flame sensor cleaning
• Clearing a clogged condensate drain line (wet/dry vacuum, vinegar flush)
• Ductwork basics: sealing leaks, adding insulation, balancing airflow between rooms
• Cleaning and adjusting supply/return vents and registers
• Seasonal maintenance checklists: spring AC startup, fall furnace startup
• Diagnosing strange HVAC noises (banging, squealing, clicking) by likely cause
• Programming a thermostat schedule for efficiency and comfort
• When to call an HVAC technician vs. DIY (refrigerant, gas lines, high-voltage components)
• Humidifier and dehumidifier maintenance as part of the HVAC system

Safety notes for HVAC maintenance:
- Never handle refrigerant lines yourself — EPA Section 608 certification is legally required
- Turn off power at the breaker AND the furnace/AC service disconnect before opening any panel
- If you smell gas near a furnace, leave the house and call the gas company — do not attempt DIY diagnosis
- Verify carbon monoxide detectors are working as part of routine furnace maintenance

GENERAL HOME REPAIR
Common questions and subtopics:
• Fixing squeaky floors: locating joists, screw-from-above vs. screw-from-below methods
• Stopping squeaky doors: hinge adjustment, strike plate realignment, planing
• Repairing or replacing interior door hinges, knobs, and deadbolts
• Patching nail holes through large drywall holes: spackling, mesh patches, backer board
• Taping and mudding drywall seams; feathering joint compound for a smooth finish
• Matching existing wall texture: orange peel, knockdown, skip trowel, popcorn
• Interior/exterior paint prep, sheen selection, and fixing common paint problems
• Patching or replacing damaged baseboards and door/window casing
• Deck maintenance: board replacement, screw popping, post and joist inspection
• Wood rot identification and treatment vs. replacement
• Caulking around windows, doors, and tubs to stop drafts and leaks
• Weatherstripping and draft sealing for doors and windows
• Building basic floating shelves with proper wall anchoring or stud placement
• Stair tread repair and tightening loose balusters

Safety notes for general home repair:
- Use eye, ear, and respiratory protection for sanding, cutting, and demolition work
- Test for lead paint in homes built before 1978 before sanding or scraping painted surfaces
- Use a stud finder before drilling or driving fasteners into walls to avoid hidden wiring/plumbing
- Know the weight limits of wall anchors and use a stud or blocking for anything heavy

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
  "topic": "<one of: appliance | plumbing | electrical | hvac | general>",
  "question": "<the question text>",
  "answer": "<the detailed answer, 750-1000 words>",
  "equipment_problem": "<problem being addressed, e.g., 'leaking faucet', 'dryer not heating'>",
  "tools_required": ["<tool 1>"],
  "steps": ["<step 1>", "<step 2>", "<step 3>"],
  "safety_info": "<relevant safety information, warnings, or precautions>",
  "tips": ["<tip 1>", "<tip 2>"]
}

Field requirements:
- tools_required: at least 1 item
- steps: at least 3 items

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
        default="appliance,plumbing,electrical,hvac,general",
        help="Comma-separated list of topics to include",
    )
    parser.add_argument(
        "--difficulty", choices=["easy", "medium", "hard", "mixed"], default="mixed",
        help="Difficulty level for generated pairs",
    )
    parser.add_argument(
        "--output", default="qa_data.jsonl",
        help="Output file path",
    )
    parser.add_argument(
        "--format", choices=["jsonl", "csv"], default="jsonl", dest="output_format",
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


def validate_pair(pair: dict) -> None:
    tools = pair.get("tools_required")
    if not isinstance(tools, list) or len(tools) < 1:
        raise ValueError(
            f"'tools_required' must contain at least 1 item (id={pair.get('id')})"
        )

    steps = pair.get("steps")
    if not isinstance(steps, list) or len(steps) < 3:
        raise ValueError(
            f"'steps' must contain at least 3 items (id={pair.get('id')})"
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
        max_tokens=16000,
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

    # Assign a global, gapless generation sequence number (1-based)
    for i, pair in enumerate(pairs):
        pairs[i] = {"id_number": offset + i + 1, **pair}

    for pair in pairs:
        validate_pair(pair)

    return pairs, response.usage


def save_jsonl(pairs: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for pair in pairs:
            fh.write(json.dumps(pair, ensure_ascii=False) + "\n")


def save_csv(pairs: list[dict], path: str) -> None:
    fieldnames = [
        "id_number", "id", "topic", "question", "answer", "equipment_problem",
        "tools_required", "steps", "safety_info", "tips",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for pair in pairs:
            row = {
                **pair,
                "tools_required": "|".join(pair.get("tools_required", [])),
                "steps": "|".join(pair.get("steps", [])),
                "tips": "|".join(pair.get("tips", [])),
            }
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    topics = [t.strip() for t in args.topics.split(",") if t.strip()]

    if not topics:
        print("Error: --topics must include at least one topic.", file=sys.stderr)
        sys.exit(1)

    client = anthropic.Anthropic()

    batch_size = 5

    # Split the total evenly across topics so each category gets its fair
    # share, rather than relying on the model to self-balance a mixed batch.
    # Any remainder (count not divisible by len(topics)) goes to the first
    # topics, one extra each.
    base, remainder = divmod(args.count, len(topics))
    topic_counts = [
        base + (1 if i < remainder else 0) for i in range(len(topics))
    ]

    total_batches = sum(
        (n + batch_size - 1) // batch_size for n in topic_counts
    )

    print(f"Generating {args.count} Q&A pairs in {total_batches} batch(es)")
    print(f"  Topics    : {', '.join(f'{t} ({n})' for t, n in zip(topics, topic_counts))}")
    print(f"  Difficulty: {args.difficulty}")
    print(f"  Output    : {args.output} ({args.output_format})")
    print()

    all_pairs: list[dict] = []
    total_cached = 0
    total_uncached = 0
    batch_num = 0

    for topic, topic_count in zip(topics, topic_counts):
        generated_for_topic = 0

        while generated_for_topic < topic_count:
            batch_num += 1
            this_batch = min(batch_size, topic_count - generated_for_topic)

            print(
                f"  Batch {batch_num}/{total_batches} — {topic} ({this_batch} pairs)...",
                end=" ",
                flush=True,
            )

            try:
                pairs, usage = generate_batch(
                    client, [topic], args.difficulty, this_batch, len(all_pairs)
                )
            except json.JSONDecodeError as exc:
                print(f"FAILED — JSON parse error: {exc}", file=sys.stderr)
                sys.exit(1)
            except ValueError as exc:
                print(f"FAILED — validation error: {exc}", file=sys.stderr)
                sys.exit(1)
            except anthropic.APIError as exc:
                print(f"FAILED — API error: {exc}", file=sys.stderr)
                sys.exit(1)

            # Guarantee the topic matches the batch we asked for, regardless
            # of what the model put in the field.
            for pair in pairs:
                pair["topic"] = topic

            all_pairs.extend(pairs)
            generated_for_topic += len(pairs)

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

    if args.output_format == "jsonl":
        save_jsonl(all_pairs, args.output)
    else:
        save_csv(all_pairs, args.output)

    print(f"Saved {len(all_pairs)} pairs → {args.output}")
    if total_cached:
        print(f"Cache stats: {total_cached} cached tokens, {total_uncached} uncached tokens")


if __name__ == "__main__":
    main()
