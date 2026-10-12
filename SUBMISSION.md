# Metropolis submission kit

Deadline: **14 Oct 2026 03:59 UTC** (09:29 IST, 14 Oct). Portal: hackathon.monad.xyz

## Checklist

- [ ] Fund the burner wallet (address from `python deploy_recorder.py --wallet`) at https://faucet.monad.xyz
- [ ] `python deploy_recorder.py` → contract address in `.env`
- [ ] `python record.py --source videos\aerial.webm --name demo4 --seconds 30` (about 6 min on the RTX 4050)
- [ ] `python verify_flight.py web\static\videos\demo4_copilot_log.json --report web\static\videos\demo4_report.md` prints OK
- [ ] `python build_site.py demo4`, deploy `site/` (Vercel / GitHub Pages), open it, press VERIFY, then TAMPER → VERIFY
- [ ] Public GitHub repo (`.env` must NOT be in it: `git status` before the first push)
- [ ] Demo video + pitch video uploaded (unlisted YouTube is fine) and opened in a private window
- [ ] Fill in the bounty questions in the portal (copy them here first)

## Form answers (draft)

**Name:** Pathfinder Black Box

**One-liner:** A tamper-evident flight recorder for autonomous drones: every AI copilot decision is
written to Monad, once per second, while the drone is still flying.

**Problem:** Drones and rovers now make their own navigation decisions with AI. After an incident, the
only record of what the AI decided, and whether the safety system overrode it, is a log file controlled
by the operator, who is the party with the most reason to edit it. Insurers, regulators and customers
have to take that log on trust.

**Solution:** Pathfinder runs perception on the vehicle (terrain, metric depth, 3D obstacles, mapping,
path planning), and an AI copilot picks a manoeuvre once per second, behind a deterministic safety
guard. Each decision is one transaction to a FlightRecorder contract on Monad, which folds it into a
per-mission keccak hash chain: the action, what the AI proposed, risk, whether the guard overrode it,
and the hash of the full log entry. After the flight, anyone can recompute the chain from the log and
compare it with the contract. Our site does it in the browser. Editing one word in one entry breaks the
match, and the verifier names the entry.

**How Monad is used:** FlightRecorder.sol on Monad testnet (chain 10143) at `<address>`. One
`logDecision` per copilot decision (1 Hz). That rate needs sub-second blocks and low fees: on Monad
the on-chain record trails the vehicle by about a second (measured confirmation p50 `<x>` ms), so a
crash or a seized laptop loses at most one decision. The client sends from a background thread with
local nonce management so perception never waits on the chain, and it keeps the gas limit tight
because Monad charges for the limit.

**Tech:** Python, PyTorch (CLIP, Depth Anything V2, YOLO-World, BootsTAPIR), Solidity 0.8.24,
web3.py, py-solc-x, ethers.js v6 (in-browser verification), Flask.

**Pre-existing work (disclose):** The perception stack (terrain/depth/obstacles/planner/mapping) is
from my terrain-nav project, started 6 Oct 2026, inside the hackathon window. The AI copilot and safety
guard were built 13 Oct 2026 for another hackathon entry (Nebius x NVIDIA). Everything Monad-specific
is new for Metropolis: FlightRecorder.sol, `terrain_nav/monad.py`, deploy and verify scripts, and the
verification site.

**What's next:** a fleet-owner smart account with per-flight session keys, the scene summary signed by a
secure element on the flight computer, MAVLink output to PX4/ArduPilot, and an insurer-facing
dashboard that indexes `MissionEnded` events across fleets.

## Demo video (≤ 3 min) shot list

1. 0:00 Site hero + the recorded flight playing (perception overlays, copilot panel showing "ON-CHAIN" ticking up).
2. 0:25 Terminal: `record.py` running; split screen with the contract on the explorer, transactions arriving every second.
3. 0:55 Site timeline: click a red-outlined decision (guard override) → its Monad tx opens in the explorer.
4. 1:20 Press VERIFY: three green checks (log head, brief hash, report hash).
5. 1:40 Press TAMPER, then VERIFY: red mismatch, the first altered entry is named and highlighted on the timeline.
6. 2:05 Terminal `verify_flight.py --tamper 12` shows the same result from the CLI.
7. 2:20 The contract code (logDecision, 12 lines) and the hash chain formula; close on the 3D map.

## Pitch video (≤ 2 min) script

> Drones are starting to make their own decisions. When one crashes into a field, a roof or a person,
> the only record of what its AI decided is a log file on the operator's laptop, and the operator is
> the one with the most to lose.
>
> Pathfinder is a black box for autonomous drones, built on Monad. The drone sees with one camera,
> running terrain, depth and obstacle models on board. An AI copilot picks the next move every second,
> and a hard-coded safety guard can veto it. Every one of those decisions is written to Monad while the
> drone is still in the air.
>
> That's only possible because Monad is fast. One transaction per second per drone, confirmed in under
> a second, at a fraction of a cent. On a slower chain you'd batch, and the last minute before a crash
> is exactly the minute you'd lose.
>
> Afterwards an insurer or a regulator downloads the log, and their browser checks it against the chain.
> Change one word and it says exactly where.
>
> Next: fleet accounts, signed sensor data, and real flight controllers. Pathfinder: AI that flies, with
> a record that can't be quietly edited.
