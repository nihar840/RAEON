"""
raeon.py — RAEON: The Face of Draeghir's Cognitive Mind

Usage:
    python raeon.py                        # face window only
    python raeon.py --db D:/Draeghir/data  # with SpaceDB learning
    python raeon.py --preset curiosity     # start with expression

Controls:
    1-7        : Emotion presets
    R          : Reset neutral
    Arrow keys : Head tilt / gaze
    ESC        : Quit

This is baby RAEON. The face that learns.
"""

import sys
import time
import argparse
import threading

from face import RaeonWindow, ExpressionVector


def run_demo_loop(window: RaeonWindow, db=None, bridge=None):
    """
    Demo mode: cycle through expressions so you can see RAEON alive.
    Runs in background thread while window renders.
    """
    sequence = [
        ("neutral",   3.0),
        ("curiosity", 3.5),
        ("joy",       3.0),
        ("thinking",  4.0),
        ("focus",     3.0),
        ("surprise",  2.5),
        ("empathy",   3.5),
        ("neutral",   2.0),
    ]

    print("[RAEON] Starting expression demo loop...")
    time.sleep(1.5)

    while True:
        for preset_name, hold_s in sequence:
            print(f"[RAEON] Expression: {preset_name}")
            window.push_preset(preset_name)
            time.sleep(hold_s)


def run_interactive(window: RaeonWindow, db, bridge):
    """
    Interactive mode: type text, RAEON reacts with expressions from memory.
    """
    print("\n[RAEON] Interactive mode. Type something and watch RAEON react.")
    print("        RAEON learns from each interaction.")
    print("        Type 'quit' to exit.\n")

    while True:
        try:
            text = input("  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            window.stop()
            break

        if text.lower() in ("quit", "exit", "q"):
            window.stop()
            break

        if not text:
            continue

        # Process through SpaceDB bridge -> expression
        ev = bridge.process_input(text)
        window.push_expression(ev)

        print(f"  RAEON: [eye={ev.eye_openness:.2f} "
              f"lip={ev.lip_curve:.2f} "
              f"brow={ev.eyebrow_angle:.2f} "
              f"tilt={ev.head_tilt:.2f}]")
        print(f"         Memories: {bridge.memory_count()}")
        print()


def main():
    parser = argparse.ArgumentParser(description="RAEON — The Face of CMA")
    parser.add_argument("--db",     type=str,  default=None,
                        help="Path to SpaceDB data folder (enables learning)")
    parser.add_argument("--preset", type=str,  default="neutral",
                        help="Starting expression preset")
    parser.add_argument("--demo",   action="store_true",
                        help="Run auto-demo expression loop")
    parser.add_argument("--width",  type=int,  default=720)
    parser.add_argument("--height", type=int,  default=900)
    args = parser.parse_args()

    print("""
  ╔══════════════════════════════════════════════════╗
  ║                                                  ║
  ║   RAEON  v0.1.0  —  The Face of Draeghir         ║
  ║   Nothing fixed. Everything evolving.            ║
  ║                                                  ║
  ╚══════════════════════════════════════════════════╝
    """)

    # Create window
    window = RaeonWindow(width=args.width, height=args.height, title="RAEON")
    window.push_preset(args.preset)

    db     = None
    bridge = None

    # Connect SpaceDB if path provided
    if args.db:
        try:
            from spacedb import SpaceClient
            from mind.db_bridge import ExpressionBridge

            print(f"[RAEON] Connecting to SpaceDB: {args.db}")
            client = SpaceClient(args.db, silent=True)
            space  = client["raeon_mind"]
            db     = space
            bridge = ExpressionBridge(space)
            print(f"[RAEON] SpaceDB connected. "
                  f"Memories: {space.status()['blocks']}")
            space.drift.start(idle_seconds=8)
        except ImportError:
            print("[RAEON] SpaceDB not installed — running face-only mode.")

    # Start background thread
    if args.demo or db is None:
        bg = threading.Thread(
            target=run_demo_loop,
            args=(window, db, bridge),
            daemon=True
        )
        bg.start()
    elif db is not None:
        bg = threading.Thread(
            target=run_interactive,
            args=(window, db, bridge),
            daemon=True
        )
        bg.start()

    # Run window (blocks until closed)
    window.run()


if __name__ == "__main__":
    main()
