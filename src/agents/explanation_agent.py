"""
explanation_agent.py
--------------------
Phase 4: Generate a Suspicious Activity Report (SAR) narrative via Groq LLM.

Input : AMLAgentState fields (risk score, patterns, features, subgraph stats)
Output: SAR narrative string + structured report dict

Low-risk accounts (tier = "LOW") receive a minimal exit summary — no LLM call.
"""

import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-70b-8192")
GROQ_TEMPERATURE = 0.2
GROQ_MAX_TOKENS = 1200

SYSTEM_PROMPT = """You are a senior Anti-Money Laundering (AML) investigation analyst.
Your task is to write a structured Suspicious Activity Report (SAR) based on graph \
analysis data provided to you.

Guidelines:
- Be factual and precise. Do not speculate beyond the data provided.
- Use formal banking and regulatory language throughout.
- For HIGH risk cases: use urgent language and recommend immediate escalation.
- For MEDIUM risk cases: use cautious language and recommend monitoring.

Always structure your report with exactly these five numbered sections:
1. SUBJECT SUMMARY
2. SUSPICIOUS ACTIVITY DESCRIPTION
3. GRAPH EVIDENCE
4. RISK ASSESSMENT
5. RECOMMENDED ACTION"""

USER_PROMPT_TEMPLATE = """Account under investigation: {account_id}
Risk Score: {risk_score:.2f} / 1.00
Risk Tier: {risk_tier}
Detected Laundering Patterns: {patterns}
Pattern Confidence Scores: {pattern_confidence}

Graph Metrics:
  - Subgraph size: {subgraph_node_count} accounts, {subgraph_edge_count} transactions
  - In-degree: {in_degree}  |  Out-degree: {out_degree}
  - In/Out Ratio: {in_out_ratio:.2f}
  - Betweenness Centrality: {betweenness:.4f}
  - Transaction Velocity: {txn_velocity:.2f} transactions/day
  - Burst Score: {burst_score:.2f}
  - Net Fund Flow: ${net_flow:,.2f}
  - Average Transaction Amount: ${avg_amount:,.2f}
  - Amount Std Deviation: ${amount_std:,.2f}
  - Cross-border transaction ratio: {cross_border_ratio:.1%}
  - Cycles detected in transaction graph: {has_cycle}
  - Intermediary accounts: {num_intermediaries}
  - Max transaction chain length: {max_path_length} hops

Risk Score Components:
  - Anomaly component:     {comp_anomaly:.3f}
  - Pattern component:     {comp_pattern:.3f}
  - Velocity component:    {comp_velocity:.3f}
  - Cross-border component:{comp_cross_border:.3f}
  - Structuring component: {comp_structuring:.3f}

Write the SAR report now."""


def generate_sar_report(
    account_id: str,
    risk_score: float,
    risk_tier: str,
    features: dict,
    pattern_result: dict,
    risk_result: dict,
) -> dict:
    """
    Generate a SAR report for a flagged account.

    For LOW risk, returns a minimal exit summary without calling the LLM.
    For MEDIUM/HIGH risk, calls the Groq API and returns a structured report.

    Args:
        account_id:    Account under investigation.
        risk_score:    Final risk score (0–1).
        risk_tier:     "LOW" | "MEDIUM" | "HIGH"
        features:      Feature dict from feature_agent.
        pattern_result: Pattern dict from pattern_agent.
        risk_result:   Risk dict from risk_agent.

    Returns:
        Structured report dict.
    """
    timestamp = datetime.now(timezone.utc).isoformat()

    if risk_tier == "LOW":
        return _low_risk_exit(account_id, risk_score, timestamp)

    narrative = _call_groq(
        account_id, risk_score, risk_tier, features, pattern_result, risk_result
    )

    return {
        "account_id": account_id,
        "risk_score": risk_score,
        "risk_tier": risk_tier,
        "detected_patterns": pattern_result.get("detected_patterns", []),
        "pattern_confidence": pattern_result.get("pattern_confidence", {}),
        "sar_narrative": narrative,
        "score_components": risk_result.get("score_components", {}),
        "report_generated_at": timestamp,
        "model_used": GROQ_MODEL,
    }


def _call_groq(
    account_id: str,
    risk_score: float,
    risk_tier: str,
    features: dict,
    pattern_result: dict,
    risk_result: dict,
    max_retries: int = 3,
) -> str:
    """Call the Groq API and return the SAR narrative string."""
    try:
        from groq import Groq, RateLimitError
    except ImportError:
        logger.error("groq package not installed. Run: pip install groq")
        return "[ERROR] groq package not installed."

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.error("GROQ_API_KEY not set in environment.")
        return "[ERROR] GROQ_API_KEY not configured."

    client = Groq(api_key=api_key)
    prompt = _build_prompt(
        account_id, risk_score, risk_tier, features, pattern_result, risk_result
    )

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=GROQ_TEMPERATURE,
                max_tokens=GROQ_MAX_TOKENS,
            )
            narrative = response.choices[0].message.content.strip()
            logger.info(
                f"SAR generated for {account_id} "
                f"(~{len(narrative.split())} words)."
            )
            return narrative

        except Exception as e:
            if "RateLimitError" in type(e).__name__ and attempt < max_retries:
                wait = 2 ** attempt
                logger.warning(f"Rate limit hit — retrying in {wait}s.")
                time.sleep(wait)
            else:
                logger.error(f"Groq API error: {e}")
                return f"[ERROR] SAR generation failed: {e}"

    return "[ERROR] SAR generation failed after all retries."


def _build_prompt(
    account_id: str,
    risk_score: float,
    risk_tier: str,
    features: dict,
    pattern_result: dict,
    risk_result: dict,
) -> str:
    components = risk_result.get("score_components", {})
    return USER_PROMPT_TEMPLATE.format(
        account_id=account_id,
        risk_score=risk_score,
        risk_tier=risk_tier,
        patterns=", ".join(pattern_result.get("detected_patterns", [])) or "None",
        pattern_confidence=pattern_result.get("pattern_confidence", {}),
        subgraph_node_count=features.get("subgraph_node_count", 0),
        subgraph_edge_count=features.get("subgraph_edge_count", 0),
        in_degree=features.get("in_degree", 0),
        out_degree=features.get("out_degree", 0),
        in_out_ratio=features.get("in_out_ratio", 0.0),
        betweenness=features.get("betweenness", 0.0),
        txn_velocity=features.get("txn_velocity", 0.0),
        burst_score=features.get("burst_score", 0.0),
        net_flow=features.get("net_flow", 0.0),
        avg_amount=features.get("avg_amount", 0.0),
        amount_std=features.get("amount_std", 0.0),
        cross_border_ratio=features.get("cross_border_ratio", 0.0),
        has_cycle=features.get("has_cycle", False),
        num_intermediaries=features.get("num_intermediaries", 0),
        max_path_length=features.get("max_path_length", 0),
        comp_anomaly=components.get("anomaly", 0.0),
        comp_pattern=components.get("pattern", 0.0),
        comp_velocity=components.get("velocity", 0.0),
        comp_cross_border=components.get("cross_border", 0.0),
        comp_structuring=components.get("structuring", 0.0),
    )


def _low_risk_exit(account_id: str, risk_score: float, timestamp: str) -> dict:
    return {
        "account_id": account_id,
        "risk_score": risk_score,
        "risk_tier": "LOW",
        "detected_patterns": [],
        "pattern_confidence": {},
        "sar_narrative": None,
        "exit_summary": (
            "Transaction analysis complete. Risk score below threshold. "
            "No suspicious laundering patterns detected. "
            "No further investigation required at this time."
        ),
        "score_components": {},
        "report_generated_at": timestamp,
        "model_used": None,
    }