"""Compile and deploy contracts/FlightRecorder.sol to Monad.

    python deploy_recorder.py --wallet     # make a burner key in .env (fund it from the faucet)
    python deploy_recorder.py --compile    # only rebuild contracts/FlightRecorder.json
    python deploy_recorder.py              # compile + deploy, writes FLIGHT_RECORDER_ADDRESS to .env
"""
import argparse
import json
import os

import config
from terrain_nav.monad import ABI_PATH, connect, load_env

SRC = os.path.join("contracts", "FlightRecorder.sol")
SOLC = "0.8.24"


def compile_contract() -> dict:
    import solcx
    if SOLC not in [str(v) for v in solcx.get_installed_solc_versions()]:
        solcx.install_solc(SOLC)
    out = solcx.compile_files([SRC], output_values=["abi", "bin"], solc_version=SOLC,
                              optimize=True, optimize_runs=200, evm_version="cancun")
    art = next(v for k, v in out.items() if k.endswith(":FlightRecorder"))
    with open(ABI_PATH, "w") as f:
        json.dump({"abi": art["abi"], "bytecode": "0x" + art["bin"], "solc": SOLC}, f, indent=1)
    print(f"Compiled -> {ABI_PATH}")
    return art


def set_env(key: str, value: str):
    lines = open(".env", encoding="utf-8").read().splitlines() if os.path.exists(".env") else []
    lines = [l for l in lines if not l.startswith(key + "=")] + [f"{key}={value}"]
    open(".env", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    os.environ[key] = value


def wallet():
    from eth_account import Account
    load_env()
    if os.environ.get("MONAD_PRIVATE_KEY"):
        print("Already have MONAD_PRIVATE_KEY:", Account.from_key(os.environ["MONAD_PRIVATE_KEY"]).address)
        return
    acct = Account.create()
    set_env("MONAD_PRIVATE_KEY", acct.key.hex())
    print(f"New burner wallet {acct.address} saved to .env (never commit .env).")
    print("Fund it with testnet MON: https://faucet.monad.xyz")


def deploy():
    load_env()
    art = compile_contract()
    w3 = connect()
    acct = w3.eth.account.from_key(os.environ["MONAD_PRIVATE_KEY"])
    bal = w3.eth.get_balance(acct.address)
    print(f"Network {config.MONAD_NETWORK} (chain {w3.eth.chain_id}), deployer {acct.address}, "
          f"balance {w3.from_wei(bal, 'ether')} MON")
    C = w3.eth.contract(abi=art["abi"], bytecode=art["bin"])
    tx = C.constructor().build_transaction({"from": acct.address,
                                            "nonce": w3.eth.get_transaction_count(acct.address)})
    tx["gas"] = int(w3.eth.estimate_gas(tx) * 1.2)
    h = w3.eth.send_raw_transaction(acct.sign_transaction(tx).raw_transaction)
    r = w3.eth.wait_for_transaction_receipt(h, timeout=120)
    set_env("FLIGHT_RECORDER_ADDRESS", r.contractAddress)
    print(f"FlightRecorder deployed at {r.contractAddress}\n  {config.MONAD_EXPLORER}/address/{r.contractAddress}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wallet", action="store_true")
    ap.add_argument("--compile", action="store_true")
    a = ap.parse_args()
    wallet() if a.wallet else compile_contract() if a.compile else deploy()
