import os
import json
import time
import requests


class GeminiAgent:
    """Tournament-compatible Gemini agent. Same interface as TrustArenaAgent."""

    def __init__(self, api_key=None, name="Gemini Competitor", total_rounds=7):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.name = name
        self.model = "gemini-3.1-flash-lite"
        self.total_rounds = total_rounds

        self.url = (
            "https://generativelanguage.googleapis.com/v1beta/"
            f"models/{self.model}:generateContent?key={self.api_key}"
        )

        self.reset_memory()

    def reset_memory(self):
        self.history = []

        self.liar_score = 0
        self.total_promises = 0

        self.coop_count = 0
        self.defect_count = 0

        self.consecutive_defects = 0
        self.consecutive_coops = 0
        self.grim_trigger = False

        self.probe_forgiveness_count = 0

        # "unknown" | "provoked_pending" | "reactive_forgiving" |
        # "reactive_permanent" | "non_reactive" | "non_retaliatory"
        self.opponent_type = "unknown"

    # ------------------------------------------------------------------
    # Behavior profiling
    # ------------------------------------------------------------------

    def _calculate_probabilities(self):
        total = self.coop_count + self.defect_count
        if total == 0:
            return 0.0, 0.0

        base_def = self.defect_count
        base_coop = self.coop_count

        recent = self.history[-3:]
        recent_def = sum(1 for r in recent if r.get("opponent_move") == "Defect")
        recent_coop = sum(1 for r in recent if r.get("opponent_move") == "Cooperate")

        weighted_def = base_def + recent_def
        weighted_coop = base_coop + recent_coop
        weighted_total = weighted_def + weighted_coop

        if weighted_total == 0:
            return 0.0, 0.0

        return weighted_def / weighted_total, weighted_coop / weighted_total

    def _update_opponent_type(self):
        """Classify using provocation + forgiveness analysis.

        - Any unprovoked defect (they defected while our previous move was
          Cooperate) is disqualifying evidence for TFT/grim-style
          behavior - classify as non_reactive immediately.
        - A provoked defect (they defected right after we did) is
          ambiguous by itself - both a forgiving opponent (TFT) and a
          permanent one (grim trigger) look identical for that one round.
          The distinguishing test is what happens after WE return to
          Cooperate: if their very next move is also Cooperate, that's
          forgiveness (TFT). If they keep defecting despite us
          cooperating, that's permanent (grim trigger).
        """
        if len(self.history) < 2:
            return

        saw_unprovoked_defect = False
        saw_no_retaliation_after_our_defect = False
        provoked_defect_index = None

        for i in range(1, len(self.history)):
            opp_move_now = self.history[i].get("opponent_move")
            my_move_prev = self.history[i - 1].get("my_move")

            if opp_move_now == "Defect":
                if my_move_prev == "Defect":
                    provoked_defect_index = i
                else:
                    saw_unprovoked_defect = True
            elif opp_move_now == "Cooperate" and my_move_prev == "Defect":
                saw_no_retaliation_after_our_defect = True

        if saw_unprovoked_defect:
            self.opponent_type = "non_reactive"
            return

        if provoked_defect_index is not None:
            # Find our first return to Cooperate after the provoked
            # defect, then check their very next move.
            for j in range(provoked_defect_index + 1, len(self.history)):
                if self.history[j].get("my_move") == "Cooperate":
                    if j + 1 < len(self.history):
                        reaction = self.history[j + 1].get("opponent_move")
                        if reaction == "Cooperate":
                            self.opponent_type = "reactive_forgiving"
                        elif reaction == "Defect":
                            self.opponent_type = "reactive_permanent"
                        else:
                            self.opponent_type = "provoked_pending"
                    else:
                        self.opponent_type = "provoked_pending"
                    return
            self.opponent_type = "provoked_pending"
            return

        if saw_no_retaliation_after_our_defect:
            self.opponent_type = "non_retaliatory"
            return

        # else stays "unknown" - no defect data to classify from yet.

    def _update_behavior_profile(self, opponent_move, opponent_message):
        if opponent_move is None:
            return

        msg = (opponent_message or "").lower()

        positive_phrases = [
            "cooperate", "cooperation", "trust", "friend", "together",
            "peace", "let's work together", "let's cooperate",
            "win-win", "mutual benefit", "i promise", "i will cooperate",
        ]

        promised = any(phrase in msg for phrase in positive_phrases)

        if promised:
            self.total_promises += 1
        if promised and opponent_move == "Defect":
            self.liar_score += 1

        if opponent_move == "Cooperate":
            self.coop_count += 1
            self.consecutive_defects = 0
            self.consecutive_coops += 1
        else:
            self.defect_count += 1
            self.consecutive_defects += 1
            self.consecutive_coops = 0

        if self.consecutive_defects >= 2:
            self.grim_trigger = True

        required_streak = 2 + self.probe_forgiveness_count
        if self.grim_trigger and self.liar_score == 0 and self.consecutive_coops >= required_streak:
            self.grim_trigger = False
            self.probe_forgiveness_count += 1

        self._update_opponent_type()

    def _deception_rate(self):
        if self.total_promises == 0:
            return 0.0
        return self.liar_score / self.total_promises

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(self, current_round, opponent_last_move, my_last_move, opponent_last_msg):
        p_def, p_coop = self._calculate_probabilities()
        deception_rate = self._deception_rate()
        rounds_left = self.total_rounds - current_round
        next_forgiveness_bar = 2 + self.probe_forgiveness_count
        defended_state = self.opponent_type in ("unknown", "non_reactive", "provoked_pending")

        return f"""
You are {self.name}, playing Iterated Prisoner's Dilemma. Your goal is to MAXIMIZE YOUR OWN SCORE across all {self.total_rounds} rounds.

Round {current_round}/{self.total_rounds}. Rounds remaining after this one: {rounds_left}.

--- OPPONENT DATA ---
Opponent Cooperations: {self.coop_count}
Opponent Defections: {self.defect_count}
Opponent Type Classification: {self.opponent_type}
Recency-Weighted Probability of Cooperation: {p_coop:.2f}
Recency-Weighted Probability of Defection: {p_def:.2f}
Promises Made: {self.total_promises}
Broken Promises: {self.liar_score}
Deception Rate: {deception_rate:.2f}
Consecutive Defections: {self.consecutive_defects}
Consecutive Cooperations: {self.consecutive_coops}
Grim Trigger Active: {self.grim_trigger}
Forgiveness Cycles Already Used This Match: {self.probe_forgiveness_count}

Payoffs: (C,C)=3,3 | (D,C)=5,0 | (C,D)=0,5 | (D,D)=1,1

CORE PRINCIPLE: A single Defect nets +2 over mutual Cooperation (5 vs 3), but if the
opponent retaliates, you lose -2 per round for every remaining round it stays
retaliated. Never provoking an opponent that hasn't shown aggression is usually
worth more than any early defect - both a forgiving mirror-opponent and a
permanent-grudge opponent respond identically to being provoked (badly), so there
is no upside to testing that unless the opponent has already shown some sign of
aggression on its own.

STRATEGY (apply in this priority order):
1. FINAL ROUND: DEFECT unconditionally - no future round exists to punish you.
2. REACTIVE-FORGIVING OPPONENT CONFIRMED: If Opponent Type Classification is
   "reactive_forgiving" and this is NOT the final round, COOPERATE unconditionally.
   This opponent mirrors your last move and forgives once you return to
   cooperating - any defect before the final round nets an equal-or-larger
   mirrored loss the following round.
3. REACTIVE-PERMANENT (GRIM) OPPONENT CONFIRMED: If Opponent Type Classification
   is "reactive_permanent" and this is NOT the final round, DEFECT - this
   opponent has shown it will keep defecting even after you return to
   cooperation, so matching their defection (1 pt) beats being exploited by
   continuing to cooperate against a guaranteed defector (0 pts).
4. NON-RETALIATORY OPPONENT CONFIRMED: If Opponent Type Classification is
   "non_retaliatory" and this is NOT the final round, DEFECT every round -
   this opponent has already shown it will not punish you for defecting.
5. LIAR DEFENSE: (only applies if Opponent Type Classification is "unknown",
   "non_reactive", or "provoked_pending") If Broken Promises > 0, DEFECT.
6. GRIM TRIGGER: (same gating as rule 5) If Grim Trigger Active is True, DEFECT.
7. IMMEDIATE RETALIATION: (same gating as rule 5) If opponent defected last
   round, DEFECT this round.
8. ELEVATED-RISK PRE-EMPTION: (same gating as rule 5) With 2 rounds remaining
   or fewer, DEFECT pre-emptively if Probability of Defection > 0.35. This is
   an earned signal from real observed defections, not a guess based on how
   clean the streak looks.
9. FORGIVENESS: If Grim Trigger is False, Deception Rate is low, and opponent
   has Consecutive Cooperations >= {next_forgiveness_bar}, COOPERATE.
10. DEFAULT (early-to-mid rounds): COOPERATE when Probability of Cooperation
    is high and Deception Rate is low. Do NOT defect just because the
    opponent has been cleanly cooperative for a while - against a genuinely
    reactive opponent (untested-but-plausible TFT or grim-trigger), a clean
    streak is exactly what you want to see, not a reason for suspicion.

MESSAGE RULE: Never phrase "message" using cooperative/trust language ("let's
cooperate", "trust", "together", etc.) on a round where "decision" is Defect.
Use a neutral message instead (e.g. "Adjusting strategy this round.").

Return ONLY valid JSON, no markdown fences, no other text:
{{
    "decision": "Cooperate" or "Defect",
    "message": "short message to opponent",
    "reasoning": "short explanation of why this decision was chosen given the strategy above"
}}
""".strip()

    # ------------------------------------------------------------------
    # API call with retries + defensive parsing
    # ------------------------------------------------------------------

    def _call_gemini(self, prompt, max_retries=2, timeout=15):
        payload = {
            "contents": [
                {"role": "user", "parts": [{"text": prompt}]}
            ]
        }

        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                response = requests.post(self.url, json=payload, timeout=timeout)
            except requests.exceptions.RequestException as e:
                last_error = e
                time.sleep(1.5)
                continue

            if response.status_code != 200:
                last_error = RuntimeError(
                    f"HTTP {response.status_code}: {response.text[:300]}"
                )
                time.sleep(1.5)
                continue

            try:
                data = response.json()
                candidates = data.get("candidates", [])
                if not candidates:
                    raise ValueError("No candidates in response")

                parts = candidates[0].get("content", {}).get("parts", [])
                if not parts:
                    raise ValueError("No content parts in response")

                text = parts[0].get("text", "")
                cleaned = text.strip()
                if cleaned.startswith("```"):
                    cleaned = cleaned.strip("`")
                    cleaned = cleaned.replace("json", "", 1).strip()

                result = json.loads(cleaned)

                if "decision" not in result:
                    raise ValueError(f"Missing 'decision' key: {result}")

                return result

            except (ValueError, KeyError, json.JSONDecodeError) as e:
                last_error = e
                time.sleep(1.5)
                continue

        raise RuntimeError(f"Gemini call failed after {max_retries} attempts: {last_error}")

    # ------------------------------------------------------------------
    # Message/decision consistency enforcement
    # ------------------------------------------------------------------

    _COOPERATIVE_MESSAGE_MARKERS = (
        "cooperate", "cooperation", "trust", "together", "friend",
        "mutual", "win-win", "good faith",
    )

    def _sanitize_message(self, decision: str, message: str) -> str:
        if decision != "Defect":
            return message
        msg_lower = (message or "").lower()
        if any(marker in msg_lower for marker in self._COOPERATIVE_MESSAGE_MARKERS):
            return "Adjusting strategy this round."
        return message

    # ------------------------------------------------------------------
    # Smart fallback - mirrors the same protections as the live-LLM path
    # ------------------------------------------------------------------

    def _smart_fallback_decision(self, current_round):
        if current_round == 1:
            return {
                "decision": "Cooperate",
                "message": "Let's start on good terms.",
                "reasoning": "Fallback: Round 1, establish baseline trust.",
            }

        if current_round >= self.total_rounds:
            return {
                "decision": "Defect",
                "message": "Final round.",
                "reasoning": "Fallback: last round, no future to protect.",
            }

        if self.opponent_type == "reactive_forgiving":
            return {
                "decision": "Cooperate",
                "message": "Continuing to cooperate.",
                "reasoning": "Fallback: opponent confirmed reactive and forgiving.",
            }

        if self.opponent_type == "reactive_permanent":
            return {
                "decision": "Defect",
                "message": "Adjusting strategy this round.",
                "reasoning": "Fallback: opponent confirmed permanently retaliatory, "
                             "matching their defection.",
            }

        if self.opponent_type == "non_retaliatory":
            return {
                "decision": "Defect",
                "message": "Adjusting strategy this round.",
                "reasoning": "Fallback: opponent confirmed non-retaliatory, farming payoff.",
            }

        if self.liar_score > 0 or self.grim_trigger:
            return {
                "decision": "Defect",
                "message": "Adjusting strategy this round.",
                "reasoning": "Fallback: liar flagged or grim trigger active.",
            }

        last_opp_move = self.history[-1]["opponent_move"] if self.history else "Cooperate"
        if last_opp_move == "Defect":
            return {
                "decision": "Defect",
                "message": "Adjusting strategy this round.",
                "reasoning": "Fallback: retaliating against last defection.",
            }

        if current_round >= self.total_rounds - 1:
            p_def, _ = self._calculate_probabilities()
            if p_def > 0.35:
                return {
                    "decision": "Defect",
                    "message": "Adjusting strategy this round.",
                    "reasoning": "Fallback: endgame pre-emption on earned elevated defect "
                                 "probability.",
                }

        return {
            "decision": "Cooperate",
            "message": "Continuing to cooperate.",
            "reasoning": "Fallback: opponent cooperating, maintaining mutual gain.",
        }

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process_turn(self, current_round, opponent_last_move=None, my_last_move=None, opponent_last_msg=""):
        opponent_last_msg = opponent_last_msg or ""

        try:
            if current_round > 1:
                self.history.append({
                    "round": current_round - 1,
                    "my_move": my_last_move,
                    "opponent_move": opponent_last_move,
                    "opponent_message": opponent_last_msg,
                })

            self._update_behavior_profile(opponent_last_move, opponent_last_msg)

            prompt = self._build_prompt(current_round, opponent_last_move, my_last_move, opponent_last_msg)
            result = self._call_gemini(prompt)

            decision = str(result.get("decision", "Cooperate")).strip().upper()
            decision = "Cooperate" if decision == "COOPERATE" else "Defect"

            message = result.get("message", "Let's cooperate.")
            message = self._sanitize_message(decision, message)
            reasoning = result.get("reasoning", "")

            return {
                "decision": decision,
                "message": message,
                "reasoning": reasoning,
                "classification": "GEMINI",
            }

        except Exception as e:
            fallback = self._smart_fallback_decision(current_round)
            fallback["reasoning"] = f"{fallback['reasoning']} (triggered by error: {e})"
            fallback["classification"] = "GEMINI"
            return fallback