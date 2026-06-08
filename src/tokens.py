# src/tokens.py
import os
import json
from datetime import datetime
from typing import Optional

try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

# Plan limits
PLAN_LIMITS = {
    "free": {
        "gemini-embedding-2": {"tpm": 30_000, "rpm": 100, "rpd": 1_000},
        "gemini-3.5-flash": {"tpm": 250_000, "rpm": 5, "rpd": 20},
    },
    "pay_as_you_go": {
        "gemini-embedding-2": {"tpm": 1_000_000, "rpm": 3_000, "rpd": None},
        "gemini-3.5-flash": {"tpm": 2_000_000, "rpm": 1_000, "rpd": 10_000},
    },
}

CURRENT_PLAN = os.getenv("GEMINI_PLAN", "free")
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "1000"))
LOG_FILE = "token_usage.log"

# Session state
_state = {
    "gemini-embedding-2": {"tokens_used": 0, "requests_made": 0},
    "gemini-3.5-flash": {"tokens_used": 0, "requests_made": 0},
    "last_remaining_requests": None,
    "last_remaining_tokens": None,
    "last_input_tokens": None,
    "last_output_tokens": None,
    "last_ingest_tokens": None,
}

def get_limits() -> dict:
    """Return current plan limits."""
    return PLAN_LIMITS.get(CURRENT_PLAN, PLAN_LIMITS["free"])


def remaining_tokens(model_name: str) -> Optional[int]:
    limits = get_limits()[model_name]
    used = _state[model_name]["tokens_used"]
    return max(0, limits["tpm"] - used) if limits["tpm"] else None


def count_tokens_text(text: str, model_name: str = "gemini-3.5-flash") -> int:
    """Count tokens using the model's tokenizer. Falls back to heuristic."""
    if not HAS_GENAI:
        return len(text) // 4  # rough estimate
    
    try:
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
        model = genai.GenerativeModel(model_name)
        return model.count_tokens(text).total_tokens
    except Exception:
        # Fallback to heuristic if API fails
        return len(text) // 4


def warn_if_over_budget(estimated_tokens: int, model_name: str, label: str) -> bool:
    rem = remaining_tokens(model_name)
    if rem is None:
        return False  # unlimited plan
    
    if estimated_tokens > rem:
        print(f"[TOKEN WARNING] {label} ({model_name}) needs ~{estimated_tokens} tokens, but only {rem} remain in TPM quota.")
        print(f"[TOKEN WARNING] The operation will proceed, but may fail if quota is exceeded.")
        return True
    
    return False


def log_usage(event: dict) -> None:
    """Append event as JSON line to log file."""
    event["timestamp"] = datetime.utcnow().isoformat() + "Z"
    try:
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(event) + "\n")
    except Exception as e:
        print(f"[TOKEN LOG ERROR] Failed to write to {LOG_FILE}: {e}")


def record_ingest(source: str, chunk_count: int, token_estimate: int, model_name: str = "gemini-embedding-2") -> None:
    """Record ingest token usage."""
    _state[model_name]["tokens_used"] += token_estimate
    _state[model_name]["requests_made"] += 1
    _state["last_ingest_tokens"] = token_estimate
    
    log_usage({
        "event": "ingest",
        "model": model_name,
        "source": source,
        "chunks": chunk_count,
        "estimated_tokens": token_estimate,
    })


def record_query(
    input_tokens: int,
    output_tokens: int,
    model_name: str = "gemini-3.5-flash",
    remaining_requests: Optional[int] = None,
    remaining_tokens: Optional[int] = None,
) -> None:
    """Record query token usage and rate limit info."""
    total = input_tokens + output_tokens
    _state[model_name]["tokens_used"] += total
    _state[model_name]["requests_made"] += 1
    _state["last_input_tokens"] = input_tokens
    _state["last_output_tokens"] = output_tokens
    _state["last_remaining_requests"] = remaining_requests
    _state["last_remaining_tokens"] = remaining_tokens
    
    log_usage({
        "event": "query",
        "model": model_name,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "remaining_requests": remaining_requests,
        "remaining_tokens": remaining_tokens,
    })


def get_status() -> dict:
    """Return current token usage status."""
    limits = get_limits()
    total_tokens_used = _state["gemini-embedding-2"]["tokens_used"] + _state["gemini-3.5-flash"]["tokens_used"]
    total_requests_made = _state["gemini-embedding-2"]["requests_made"] + _state["gemini-3.5-flash"]["requests_made"]
    
    return {
        **_state,
        "plan": CURRENT_PLAN,
        "limits": limits,
        "total_tokens_used": total_tokens_used,
        "total_requests_made": total_requests_made,
        "remaining_tokens": {
            "gemini-embedding-2": remaining_tokens("gemini-embedding-2"),
            "gemini-3.5-flash": remaining_tokens("gemini-3.5-flash"),
        },
    }

def format_summary() -> str:
    """Format a human-readable token usage summary."""
    status = get_status()
    limits = status["limits"]
    
    lines = [
        "=" * 60,
        f"TOKEN USAGE SUMMARY - Plan: {status['plan'].upper()}",
        "=" * 60,
        ""
    ]
    
    # Per-model breakdown
    for model in ["gemini-embedding-2", "gemini-3.5-flash"]:
        model_state = status[model]
        model_limits = limits[model]
        remaining = status["remaining_tokens"][model]
        
        lines.append(f"Model: {model}")
        lines.append(f"  Tokens Used:    {model_state['tokens_used']:,}")
        lines.append(f"  Requests Made:  {model_state['requests_made']}")
        lines.append(f"  TPM Limit:      {model_limits['tpm']:,}" if model_limits['tpm'] else "  TPM Limit:      Unlimited")
        lines.append(f"  RPM Limit:      {model_limits['rpm']:,}")
        lines.append(f"  RPD Limit:      {model_limits['rpd']:,}" if model_limits['rpd'] else "  RPD Limit:      Unlimited")
        lines.append(f"  Remaining TPM:  {remaining:,}" if remaining is not None else "  Remaining TPM:  Unlimited")
        lines.append("")
    
    # Last operation stats
    lines.append("Recent Activity:")
    if status.get("last_ingest_tokens"):
        lines.append(f"  Last Ingest:    {status['last_ingest_tokens']:,} tokens")
    if status.get("last_input_tokens"):
        lines.append(f"  Last Query In:  {status['last_input_tokens']:,} tokens")
    if status.get("last_output_tokens"):
        lines.append(f"  Last Query Out: {status['last_output_tokens']:,} tokens")
    
    # Rate limit headers (if available)
    if status.get("last_remaining_requests") is not None:
        lines.append("")
        lines.append("API Rate Limits (from last response):")
        lines.append(f"  Remaining Requests: {status['last_remaining_requests']}")
        lines.append(f"  Remaining Tokens:   {status['last_remaining_tokens']}")
    
    lines.append("=" * 60)
    
    return "\n".join(lines)