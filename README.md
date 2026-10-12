# Pathfinder Black Box: a tamper-evident flight recorder for autonomous drones, on Monad

**Monad Hackathon · Metropolis**

Autonomous drones and rovers are starting to make their own decisions: survey drones, farm rovers,
delivery quads, search-and-rescue. When something goes wrong, someone has to answer *what did the
AI decide, when, and did the safety system step in?* Today that answer is a log file on the operator's
laptop, and the operator is the party with the most reason to edit it.

Pathfinder makes the AI copilot's decisions **append-only and publicly verifiable while the vehicle is
still flying**. Every decision (one per second) is a transaction to a `FlightRecorder` contract on Monad
that folds it into a per-mission hash chain. Afterwards anyone (insurer, regulator, customer, incident
investigator) can take the full log, recompute the chain and compare it with the contract. If one
reason was edited, an override dropped or two steps reordered, the check fails and names the first
entry that changed.

**Why Monad:** a 1 Hz control loop needs a ledger that keeps up with it. With sub-second blocks
and low fees, one transaction per decision is practical, and the on-chain record trails the vehicle by
about a second. On a 12-second-block chain the log would have to be batched, and a crash or a
confiscated laptop could take the last minute of decisions with it. On Monad the drone loses at most
the decision it was making.

```
camera ─┬─ CLIP terrain ─────────────┐
        ├─ Depth Anything V2 ────────┤
        ├─ YOLO-World 3D obstacles ──┼─► scene summary (~1 KB JSON, no pixels)
        ├─ visual odometry map ──────┤            │
        └─ Dijkstra planner ─────────┘            ▼
                                AI copilot (Nemotron, or local rules) @ 1 Hz
                                                  │ action, risk, reason
                                                  ▼
                                safety guard (hard rules can veto the AI) ─► HUD
                                                  │
                     ┌────────────────────────────┴───────────────────────────┐
                     ▼                                                        ▼
        full log entry (scene + decision)             FlightRecorder.logDecision() on Monad
        kept off-chain, shared later                  head = keccak(head ‖ fields ‖ entryHash)
                     └──────────── verify_flight.py / site recompute and compare ────┘
```

## What's on-chain

`contracts/FlightRecorder.sol` (Solidity 0.8.24, MIT)

| call | when | stores / emits |
|---|---|---|
| `startMission(name, briefHash)` | take-off | operator, start time, hash of the preflight mission brief |
| `logDecision(id, tMs, action, proposed, riskBps, overridden, entryHash)` | every copilot decision (1 Hz) | updates `head = keccak256(head ‖ tMs ‖ action ‖ proposed ‖ riskBps ‖ overridden ‖ entryHash)`, counts guard overrides, emits `Decision` |
| `endMission(id, reportHash)` | landing | end time, hash of the mission report, emits `MissionEnded` |

`entryHash` is keccak256 of the canonical JSON of the full log entry (scene summary, decision, reason,
latency), so the off-chain detail is committed without paying to store it. Only the operator who
opened a mission can append to it, and a closed mission can't be changed. One `logDecision` costs about
100k gas. Monad charges for the gas limit rather than the gas used, so the recorder estimates the limit
once per mission and keeps it tight.

## Files

| file | role |
|---|---|
| `contracts/FlightRecorder.sol` | the black box contract |
| `terrain_nav/monad.py` | recorder client: canonical hashing, hash chain, background transaction sender with local nonces, so video never waits on the network |
| `deploy_recorder.py` | burner wallet, compile (py-solc-x), deploy, writes `.env` |
| `verify_flight.py` | recomputes a flight log's chain, compares it with the contract and names the first altered entry (`--tamper N` demo) |
| `build_site.py` + `site/` | static, judge-facing page: video, decision timeline with a transaction per decision, **in-browser verification against Monad** (ethers.js, public RPC), and a tamper button |
| `terrain_nav/copilot.py` | 1 Hz copilot: Nemotron via Nebius when `NEBIUS_API_KEY` is set, else a local rule-based copilot. Streams each decision to the recorder |
| `terrain_nav/safety.py` | deterministic guard that can veto the AI. Every veto is counted on-chain |
| `record.py`, `main.py` | record a video run / fly live from a webcam. Both open and close a mission on Monad |
| `web/` | local Flask demo site (live camera feed, debrief, mission planner) |

## Setup

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
$env:PYTHONPATH = "tapnet_src"

python deploy_recorder.py --wallet      # burner key -> .env, prints the address
#   fund it with testnet MON at https://faucet.monad.xyz
python deploy_recorder.py               # compile + deploy FlightRecorder, address -> .env
```

`MONAD_NETWORK=testnet` (chain 10143) is the default. Set `MONAD_NETWORK=mainnet` (chain 143) in `.env`
to use mainnet. Optional: `NEBIUS_API_KEY` switches the copilot from local rules to NVIDIA Nemotron.

## Run

```powershell
# fly: record a run; every decision goes to Monad while it plays
python record.py --source videos\aerial.webm --name demo4 --seconds 30
#   -> web\static\videos\demo4.mp4, demo4_copilot_log.json (with tx hashes), demo4_report.md

# audit: anyone with the log can check it against the contract
python verify_flight.py web\static\videos\demo4_copilot_log.json --report web\static\videos\demo4_report.md
python verify_flight.py web\static\videos\demo4_copilot_log.json --tamper 12     # watch it fail

# publish the judge site (static, verifies in the browser)
python build_site.py demo4              # -> site/
python -m http.server -d site 8000

python main.py                          # live from the webcam, also recorded on Monad
```

## Known limits

- The recorder proves *the log wasn't changed after it was written*. It can't prove the camera saw the
  truth: a compromised vehicle could log lies from the start. Signing the scene summary inside a
  secure element on the flight computer is the next step.
- Burner key on the vehicle: whoever holds it is the operator. A production version would use
  a smart account that a fleet owner controls, with session keys per flight.
- Decisions are advisory (HUD + log). MAVLink output to PX4/ArduPilot behind the same guard is next.
- Perception is zero-shot (CLIP terrain, monocular depth), so it confuses look-alike terrain and drifts on long flights.

## License

MIT, see [LICENSE](LICENSE). `tapnet_src/` is Google DeepMind's TAPNet under Apache 2.0 (its own LICENSE).
