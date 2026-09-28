"""Coverage and precision. This is the only file in the method that reads gold."""
from .data import match


def metrics(pools, state, contested_ids):
    """coverage  = pools with an answer / all pools
    precision = pools whose answer matches gold / pools with an answer
    contested_accuracy = the same, over the contested pools that got an answer
    """
    answered = [i for i in contested_ids if state[i]["answer"] is not None]
    hits = sum(match(state[i]["answer"], pools[i]["gold"]) for i in answered)
    total_committed = sum(1 for s in state.values() if s["answer"] is not None)
    total_hits = sum(match(s["answer"], pools[i]["gold"])
                     for i, s in state.items() if s["answer"] is not None)
    n = len(state)
    return {
        "contested_answered": len(answered),
        "contested_accuracy": round(hits / max(len(answered), 1), 4),
        "coverage": round(total_committed / n, 4),
        "precision": round(total_hits / max(total_committed, 1), 4),
    }
