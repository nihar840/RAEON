"""
mind/db_bridge.py -- SpaceDB <-> RAEON Expression Bridge

This is where the magic happens.

Every expression RAEON shows gets stored as a MemoryBlock in SpaceDB.
Expression blocks get reinforced alongside the input blocks that triggered them.
Over time, W matrix learns: certain inputs -> certain expressions.
Expression clusters emerge -> expression personalities form.
RAEON starts expressing emotions automatically from memory -- not rules.

Compatible with SpaceDB v0.2.0+ (chainable query API).
"""

import logging
from typing import Optional

from face.expression import ExpressionVector

log = logging.getLogger("raeon.bridge")


class ExpressionBridge:
    """
    Connects RAEON's expression system to SpaceDB.

    Flow:
        1. CMA processes input -> generates context embedding
        2. ExpressionBridge queries SpaceDB for closest expression memory
        3. Returns ExpressionVector to render
        4. Stores new expression block in SpaceDB
        5. Reinforces input_block <-> expression_block
        6. W matrix evolves. Expression personalities emerge.
    """

    SENSORY_TYPE = "expression"

    def __init__(self, space):
        """
        Parameters
        ----------
        space : spacedb.Space
            An open Space instance from SpaceClient.
        """
        self.space = space
        self._last_input_block_id: Optional[str] = None

    # -- core: input -> expression ----------------------------------------

    def process_input(self, text: str,
                      fallback_preset: str = "neutral") -> ExpressionVector:
        """
        Given raw text input:
        1. Ingest into SpaceDB as a language block
        2. Query for related expression blocks
        3. If expression memory found -> use it
        4. Else -> use fallback preset
        5. Store expression + reinforce the pair
        Returns ExpressionVector for face renderer.
        """

        # Ingest language block
        lang_block = self.space.ingest(text, sensory_type="text")
        self._last_input_block_id = lang_block.id

        # Personality switching: pick the personality closest to input context
        pid = self._pick_personality(text)

        # Query SpaceDB for related expression memories (v0.2.0 chainable API)
        q = self.space.query(text).within(ms=200).limit(5)
        if pid:
            q = q.as_personality(pid)
        expr_results = q.fetch()

        # Filter for expression-type blocks only
        expr_hits = [
            r for r in expr_results
            if r["sensory_type"] == self.SENSORY_TYPE
        ]

        if expr_hits:
            # RAEON remembers how it felt about something like this
            best = expr_hits[0]
            ev = self._decode_expression(best["token"])
            log.info("Memory recall: '%s'  score=%.3f",
                     best["token"][:40], best["score"])
        else:
            # No memory yet -> use keyword heuristic for bootstrapping
            ev = self._heuristic_expression(text, fallback_preset)
            log.info("Heuristic expression: %s", ev.to_dict())

        # Store this expression as a new memory block
        expr_token = self._encode_expression(ev)
        expr_block = self.space.ingest(expr_token, sensory_type=self.SENSORY_TYPE)

        # Reinforce: language block <-> expression block
        # W matrix learns their connection
        self.space.reinforce(lang_block.id, expr_block.id, strength=0.015)

        return ev

    def process_expression_vector(self, ev: ExpressionVector,
                                   trigger_block_id: Optional[str] = None):
        """
        Store an expression directly (e.g. from external CMA output).
        Reinforce with last input block if available.
        """
        expr_token = self._encode_expression(ev)
        expr_block = self.space.ingest(expr_token, sensory_type=self.SENSORY_TYPE)

        link_id = trigger_block_id or self._last_input_block_id
        if link_id:
            self.space.reinforce(link_id, expr_block.id, strength=0.015)

        return expr_block

    # -- expression encoding -----------------------------------------------

    def _encode_expression(self, ev: ExpressionVector) -> str:
        """
        Encode ExpressionVector as a text token for SpaceDB.
        Format: 'expr::eye=0.8,lip=0.6,brow=0.3,...'
        This gets embedded by sentence-transformers like any other block.
        """
        d = ev.to_dict()
        parts = [f"{k[:4]}={v:.2f}" for k, v in d.items()
                 if k != "extras" and abs(v) > 0.01]
        return "expr::" + ",".join(parts)

    def _decode_expression(self, token: str) -> ExpressionVector:
        """Decode stored expression token back to ExpressionVector."""
        if not token.startswith("expr::"):
            return ExpressionVector()
        body = token[6:]
        d = {}
        key_map = {
            "eye_o": "eye_openness",
            "eyeb":  "eyebrow_angle",
            "brow":  "brow_scrunch",
            "lip_c": "lip_curve",
            "lip_p": "lip_part",
            "jaw_":  "jaw_tension",
            "nose":  "nose_flare",
            "chee":  "cheek_raise",
            "head":  "head_tilt",
            "gaze":  "gaze_direction",
        }
        for part in body.split(","):
            if "=" in part:
                k, v = part.split("=", 1)
                for prefix, full_name in key_map.items():
                    if k.startswith(prefix[:3]):
                        d[full_name] = float(v)
                        break
        return ExpressionVector.from_dict(d)

    def _heuristic_expression(self, text: str,
                               fallback: str = "neutral") -> ExpressionVector:
        """
        Bootstrap expression before RAEON has enough memories.
        Simple keyword matching -> preset.
        Replaced by learned W matrix over time.
        """
        text_lower = text.lower()
        if any(w in text_lower for w in ["?", "what", "how", "why", "curious"]):
            return ExpressionVector.from_dict({
                "eye_openness": 0.85, "eyebrow_angle": 0.4,
                "head_tilt": 0.3, "lip_curve": 0.1
            })
        elif any(w in text_lower for w in ["love", "happy", "great", "awesome", "joy"]):
            return ExpressionVector.from_dict({
                "eye_openness": 0.8, "lip_curve": 0.9,
                "cheek_raise": 0.7, "lip_part": 0.25
            })
        elif any(w in text_lower for w in ["code", "build", "think", "solve", "debug"]):
            return ExpressionVector.from_dict({
                "eye_openness": 0.5, "brow_scrunch": 0.4,
                "lip_curve": -0.05, "jaw_tension": 0.1
            })
        elif any(w in text_lower for w in ["wow", "oh", "whoa", "really", "seriously"]):
            return ExpressionVector.from_dict({
                "eye_openness": 1.0, "eyebrow_angle": 0.8,
                "lip_part": 0.6, "nose_flare": 0.4
            })
        else:
            return ExpressionVector.from_dict({
                "eye_openness": 0.6, "eyebrow_angle": 0.0,
                "lip_curve": 0.05
            })

    # -- personality switching ------------------------------------------------

    def _pick_personality(self, text: str) -> Optional[str]:
        """
        Choose which personality to invoke based on input context.
        Quick query per personality — pick the one with highest avg score.
        Returns personality cluster_id, or None if no personalities exist.
        """
        personalities = self.space.clusters.personalities()
        if not personalities:
            return None
        best_pid, best_score = None, -1.0
        for p in personalities:
            results = (
                self.space.query(text)
                    .as_personality(p["id"])
                    .within(ms=50)
                    .limit(3)
                    .fetch()
            )
            if results:
                avg = sum(r["score"] for r in results) / len(results)
                if avg > best_score:
                    best_score = avg
                    best_pid = p["id"]
        if best_pid:
            log.info("Personality switch -> %s (score=%.3f)", best_pid, best_score)
        return best_pid

    # -- introspection ------------------------------------------------------

    def expression_clusters(self) -> list:
        """Return expression-related clusters from SpaceDB."""
        return [
            c for c in self.space.clusters.all()
            if (c.get("name") or "").startswith("expr")
        ]

    def memory_count(self) -> int:
        """Total memory blocks in the space."""
        return self.space.status()["blocks"]
