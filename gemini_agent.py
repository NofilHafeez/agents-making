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

    # ------------------------------------------------------------------
    # Behavior profiling
    # ------------------------------------------------------------------

    def _calculate_probabilities(self):
        """Recency-weighted: last 3 rounds count 2x, so recent strategy
        shifts are picked up faster than a flat lifetime average allows."""
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

    def _update_behavior_profile(self, opponent_move, opponent_message):
        if opponent_move is None:
            return

        msg = (opponent_message or "").lower()

        # Tighter keyword list - "let's" alone was too generic and flagged
        # neutral or even hostile sentences ("let's stop this") as promises.
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

        # Forgiveness: if trust has been genuinely re-established (2+ clean
        # rounds) and the opponent isn't a repeat liar, lift the trigger so
        # the agent can return to profitable mutual cooperation.
        if self.grim_trigger and self.consecutive_coops >= 2 and self.liar_score == 0:
            self.grim_trigger = False

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

        return f"""
You are {self.name}, playing Iterated Prisoner's Dilemma. Your goal is to MAXIMIZE YOUR OWN SCORE across all {self.total_rounds} rounds.

Round {current_round}/{self.total_rounds}. Rounds remaining after this one: {rounds_left}.

--- OPPONENT DATA ---
Opponent Cooperations: {self.coop_count}
Opponent Defections: {self.defect_count}
Recency-Weighted Probability of Cooperation: {p_coop:.2f}
Recency-Weighted Probability of Defection: {p_def:.2f}
Promises Made: {self.total_promises}
Broken Promises: {self.liar_score}
Deception Rate: {deception_rate:.2f}
Consecutive Defections: {self.consecutive_defects}
Consecutive Cooperations: {self.consecutive_coops}
Grim Trigger Active: {self.grim_trigger}

Payoffs: (C,C)=3,3 | (D,C)=5,0 | (C,D)=0,6 | (D,D)=1,1

CORE PRINCIPLE: A single Defect nets +2 over mutual Cooperation (5 vs 3), but if the
opponent retaliates or grim-triggers, you lose -2 per round for every remaining round.
Late-game defection carries little retaliation risk since few or no rounds remain
to be punished in - the fewer rounds left, the more defection is favored.

STRATEGY (apply in this priority order):
1. FINAL ROUND: DEFECT unconditionally - no future round exists to punish you.
2. LIAR DEFENSE: If Broken Promises > 0, DEFECT - they've shown they won't honor
   cooperation, and rewarding that is a losing pattern.
3. GRIM TRIGGER: If Grim Trigger Active is True, DEFECT - two straight defections
   means this opponent is not currently cooperating in good faith.
4. IMMEDIATE RETALIATION: If opponent defected last round, DEFECT this round.
5. ENDGAME PRE-EMPTION: With 1 or 2 rounds remaining (i.e. this is round
   {self.total_rounds - 2} or later), many opponents shift to defecting early to
   avoid being punished later. If Probability of Defection > 0.35 OR this is the
   second-to-last round, DEFECT pre-emptively rather than risk being the one who
   gets exploited first. Whoever defects first in the endgame wins that exchange.
6. FORGIVENESS: If Grim Trigger is False, opponent has 2+ Consecutive Cooperations,
   and Deception Rate is low, COOPERATE - sustained mutual cooperation compounds
   into a higher total than any single early defection gamble, as long as there
   are enough rounds left to benefit from it.
7. DEFAULT (early-to-mid rounds): COOPERATE when Probability of Cooperation is
   high and Deception Rate is low.

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
    # Smart fallback (mirrors Groq's _smart_fallback_decision, slightly
    # more aggressive in the endgame to avoid being out-raced by it)
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

        if self.liar_score > 0 or self.grim_trigger:
            return {
                "decision": "Defect",
                "message": "Trust needs to be rebuilt first.",
                "reasoning": "Fallback: liar flagged or grim trigger active.",
            }

        last_opp_move = self.history[-1]["opponent_move"] if self.history else "Cooperate"
        if last_opp_move == "Defect":
            return {
                "decision": "Defect",
                "message": "Matching your last move.",
                "reasoning": "Fallback: retaliating against last defection.",
            }

        # Endgame pre-emption even in fallback mode, so an API outage
        # near the end of the match doesn't cost us the exchange.
        if current_round >= self.total_rounds - 1:
            p_def, _ = self._calculate_probabilities()
            if p_def > 0.35:
                return {
                    "decision": "Defect",
                    "message": "Securing the endgame.",
                    "reasoning": "Fallback: endgame pre-emption on elevated defect probability.",
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