"""Check a flight log against its on-chain black box on Monad.

    python verify_flight.py web\\static\\videos\\demo4_copilot_log.json
    python verify_flight.py <log> --tamper 12      # edit entry 12 in memory and show the check fail

Recomputes the hash chain from the full off-chain log (every scene summary, decision and reason) and
compares it with the head stored by FlightRecorder. If they differ, it walks the on-chain Decision
events to name the first entry that was changed.
"""
import argparse
import json

from terrain_nav.copilot import ACTIONS
from terrain_nav.monad import ZERO, canonical, chain_step, connect, contract, decision_fields, keccak, load_env


def recompute(log: list) -> list[bytes]:
    heads, h = [], ZERO
    for e in log:
        h = chain_step(h, *decision_fields(e, ACTIONS))
        heads.append(h)
    return heads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--report", help="mission report .md to check against the on-chain report hash")
    ap.add_argument("--tamper", type=int, help="change this entry's reason before checking (demo)")
    a = ap.parse_args()
    load_env()

    data = json.load(open(a.log, encoding="utf-8"))
    chain, log = data.get("chain"), data["log"]
    if not chain:
        raise SystemExit("This log was recorded without the Monad recorder (no 'chain' section).")
    if a.tamper is not None:
        log[a.tamper]["decision"]["reason"] += " (edited)"
        print(f"Tampered with entry {a.tamper} (t={log[a.tamper]['t']}s) in memory.")

    w3 = connect()
    if w3.eth.chain_id != chain["chain_id"]:
        raise SystemExit(f"RPC is chain {w3.eth.chain_id}, log was recorded on chain {chain['chain_id']}. "
                         f"Set MONAD_NETWORK / MONAD_RPC_URL.")
    c = contract(w3, chain["contract"])
    m = c.functions.missions(chain["mission_id"]).call()
    operator, _, ended, n_on, ovr_on, brief_on, head_on, report_on, name = m
    print(f"Mission #{chain['mission_id']} '{name}' by {operator} on chain {chain['chain_id']}")
    print(f"  on-chain: {n_on} decisions, {ovr_on} guard overrides, closed={bool(ended)}")

    brief = data.get("mission")
    brief_ok = (keccak(canonical(brief)) if brief else ZERO) == brief_on
    print(f"  mission brief hash   {'OK' if brief_ok else 'MISMATCH'}")

    heads = recompute(log)
    ok = len(log) == n_on and heads and heads[-1] == head_on
    print(f"  decision log ({len(log)} entries) {'OK: matches on-chain head' if ok else 'MISMATCH'}")
    if not ok:
        on = {}
        for tx in chain["decision_txs"]:
            for ev in c.events.Decision().process_receipt(w3.eth.get_transaction_receipt(tx)):
                if ev["args"]["id"] == chain["mission_id"]:
                    on[ev["args"]["seq"]] = ev["args"]["head"]
        bad = next((i for i, h in enumerate(heads) if on.get(i) != h), None)
        if bad is not None:
            print(f"  first altered entry: #{bad} at t={log[bad]['t']}s")

    if a.report:
        rep = open(a.report, "rb").read()
        print(f"  mission report hash  {'OK' if keccak(rep) == report_on else 'MISMATCH'}")
    raise SystemExit(0 if ok and brief_ok else 1)


if __name__ == "__main__":
    main()
