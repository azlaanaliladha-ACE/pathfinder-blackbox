"""On-chain flight recorder on Monad.

Each copilot decision becomes one `logDecision` transaction while the vehicle is still flying.
Monad's sub-second blocks and low fees are what make one transaction per decision (1 Hz) practical;
on a 12 s chain the log would have to be batched and a crash could lose the last minute.

The contract keeps a running hash chain per mission, so the full off-chain log (scene summaries,
reasons, latencies) can be verified later against the single on-chain `head`. See verify_flight.py.

Needs MONAD_PRIVATE_KEY and FLIGHT_RECORDER_ADDRESS (deploy_recorder.py writes it to .env).
Without them the recorder is off and everything else runs as before.
"""
import json
import os
import queue
import threading
import time

import config

ABI_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "contracts", "FlightRecorder.json")
ZERO = b"\x00" * 32


def load_env(path: str = ".env"):
    """Minimal .env loader (KEY=VALUE lines) so keys never have to be typed into the shell."""
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def canonical(obj) -> bytes:
    """Deterministic JSON that a browser can reproduce: sorted keys, no spaces, UTF-8,
    integral floats written as ints (JSON.stringify prints 12.0 as 12)."""
    def norm(x):
        if isinstance(x, float) and x.is_integer():
            return int(x)
        if isinstance(x, dict):
            return {k: norm(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [norm(v) for v in x]
        return x
    return json.dumps(norm(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def keccak(data: bytes) -> bytes:
    from eth_utils import keccak as _k
    return _k(data)


def chain_step(head: bytes, t_ms: int, action: int, proposed: int, risk_bps: int, overridden: bool,
               entry_hash: bytes) -> bytes:
    """Same as the contract: keccak256(abi.encodePacked(head, tMs, action, proposed, riskBps, overridden, entryHash))."""
    packed = (head + t_ms.to_bytes(4, "big") + bytes([action, proposed]) + risk_bps.to_bytes(2, "big")
              + bytes([1 if overridden else 0]) + entry_hash)
    return keccak(packed)


def decision_fields(entry: dict, actions: list[str]) -> tuple:
    """(tMs, action, proposed, riskBps, overridden, entryHash) for one copilot log entry."""
    d = entry["decision"]
    body = {"t": entry["t"], "scene": entry["scene"], "decision": d}
    return (int(round(entry["t"] * 1000)), actions.index(d["action"]),
            actions.index(d["proposed"]) if d["proposed"] in actions else actions.index("HOLD"),
            int(round(d["risk"] * 10000)), bool(d["override"]), keccak(canonical(body)))


def connect():
    from web3 import Web3
    w3 = Web3(Web3.HTTPProvider(config.MONAD_RPC_URL, request_kwargs={"timeout": 20}))
    return w3


def contract(w3, address: str):
    with open(ABI_PATH, encoding="utf-8") as f:
        abi = json.load(f)["abi"]
    return w3.eth.contract(address=w3.to_checksum_address(address), abi=abi)


class Recorder:
    """Streams decisions to FlightRecorder on a background thread; never blocks the video loop."""

    def __init__(self, actions: list[str]):
        load_env()
        self.actions = actions
        self.enabled = False
        self.status = "OFF"
        self.mission_id = None
        self.head = ZERO
        self.txs: list[str] = []
        self.sent = self.confirmed = self.failed = 0
        self.confirm_ms: list[float] = []
        key, addr = os.environ.get("MONAD_PRIVATE_KEY"), os.environ.get("FLIGHT_RECORDER_ADDRESS")
        if not (key and addr):
            self.status = "SET MONAD KEY"
            return
        self.w3 = connect()
        self.acct = self.w3.eth.account.from_key(key)
        self.c = contract(self.w3, addr)
        self.address = self.c.address
        self.chain_id = self.w3.eth.chain_id
        self.enabled = True
        self.status = "READY"
        self._q: queue.Queue = queue.Queue()
        self._worker: threading.Thread | None = None

    # --- transactions ---

    def _send(self, fn, gas: int) -> str:
        tx = fn.build_transaction({"from": self.acct.address, "nonce": self._nonce, "gas": gas,
                                   "chainId": self.chain_id, **self._fees})
        self._nonce += 1
        signed = self.acct.sign_transaction(tx)
        return self.w3.to_hex(self.w3.eth.send_raw_transaction(signed.raw_transaction))

    def _fees_now(self) -> dict:
        base = self.w3.eth.get_block("latest").get("baseFeePerGas", 0) or 0
        tip = self.w3.eth.max_priority_fee or 0
        return {"maxFeePerGas": base * 2 + tip, "maxPriorityFeePerGas": tip}

    def start(self, name: str, brief: dict | None):
        if not self.enabled:
            return
        self._nonce = self.w3.eth.get_transaction_count(self.acct.address, "pending")
        self._fees = self._fees_now()
        brief_hash = keccak(canonical(brief)) if brief else ZERO
        fn = self.c.functions.startMission(name[:64], brief_hash)
        tx = self._send(fn, int(fn.estimate_gas({"from": self.acct.address}) * 1.3))
        rcpt = self.w3.eth.wait_for_transaction_receipt(tx, timeout=60)
        self.mission_id = self.c.events.MissionStarted().process_receipt(rcpt)[0]["args"]["id"]
        self.head, self.txs, self.start_tx = ZERO, [], tx
        # Monad charges the gas *limit*, so estimate one decision once and keep the limit tight
        self._log_gas = int(self.c.functions.logDecision(self.mission_id, 0, 0, 0, 0, True, ZERO)
                            .estimate_gas({"from": self.acct.address}) * 1.25)
        self.status = f"MISSION #{self.mission_id}"
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def log(self, entry: dict):
        """Queue one copilot log entry; stores its chain head in the entry for the site and verifier."""
        if not self.enabled or self.mission_id is None:
            return
        f = decision_fields(entry, self.actions)
        self.head = chain_step(self.head, *f)
        entry["chain"] = {"seq": len(self.txs), "head": "0x" + self.head.hex()}
        self.txs.append("")
        self._q.put((len(self.txs) - 1, f, entry))

    def _run(self):
        while True:
            item = self._q.get()
            if item is None:
                return
            i, f, entry = item
            t0 = time.perf_counter()
            try:
                tx = self._send(self.c.functions.logDecision(self.mission_id, *f), self._log_gas)
                self.txs[i] = entry["chain"]["tx"] = tx
                self.sent += 1
                self.status = f"#{self.mission_id} TX {self.sent}"
                threading.Thread(target=self._confirm, args=(tx, t0), daemon=True).start()
            except Exception as e:
                self.failed += 1
                self.status = f"TX ERR {type(e).__name__}"[:22]
                self._nonce = self.w3.eth.get_transaction_count(self.acct.address, "pending")

    def _confirm(self, tx: str, t0: float):
        try:
            r = self.w3.eth.wait_for_transaction_receipt(tx, timeout=60, poll_latency=0.1)
            if r.status == 1:
                self.confirmed += 1
                self.confirm_ms.append((time.perf_counter() - t0) * 1000)
            else:
                self.failed += 1
        except Exception:
            self.failed += 1

    def end(self, report: str | None) -> dict | None:
        """Flush the queue, close the mission on-chain and return a summary for the flight log."""
        if not self.enabled or self.mission_id is None:
            return None
        self._q.put(None)
        self._worker.join()
        deadline = time.time() + 60
        while self.confirmed + self.failed < self.sent and time.time() < deadline:
            time.sleep(0.2)
        report_hash = keccak(report.encode("utf-8")) if report else ZERO
        tx = self._send(self.c.functions.endMission(self.mission_id, report_hash), 120_000)
        self.w3.eth.wait_for_transaction_receipt(tx, timeout=60)
        on_chain = self.c.functions.missions(self.mission_id).call()
        lat = sorted(self.confirm_ms)
        return {"chain_id": self.chain_id, "rpc": config.MONAD_RPC_URL, "contract": self.address,
                "explorer": config.MONAD_EXPLORER, "mission_id": self.mission_id, "operator": self.acct.address,
                "start_tx": self.start_tx, "end_tx": tx, "decision_txs": [t for t in self.txs if t],
                "head": "0x" + self.head.hex(), "on_chain_head": "0x" + on_chain[6].hex(),
                "report_hash": "0x" + report_hash.hex(), "confirmed": self.confirmed, "failed": self.failed,
                "confirm_ms_p50": round(lat[len(lat) // 2]) if lat else None,
                "confirm_ms_p95": round(lat[int(len(lat) * 0.95)]) if lat else None}
