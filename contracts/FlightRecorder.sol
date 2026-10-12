// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @title FlightRecorder: a tamper-evident black box for autonomous drones and rovers on Monad.
/// @notice Every copilot decision (1 Hz) is appended to a per-mission hash chain on-chain while the
///         vehicle is still flying. Anyone holding the off-chain flight log can recompute the chain and
///         compare it with `missions(id).head`: one edited, dropped or reordered entry breaks the match.
contract FlightRecorder {
    struct Mission {
        address operator;
        uint64 startedAt;
        uint64 endedAt;
        uint32 decisions;
        uint32 overrides;       // decisions where the safety guard vetoed the AI
        bytes32 briefHash;      // keccak256 of the preflight mission brief (0 = free flight)
        bytes32 head;           // running hash chain over every decision
        bytes32 reportHash;     // keccak256 of the post-flight report
        string name;
    }

    Mission[] public missions;

    event MissionStarted(uint256 indexed id, address indexed operator, bytes32 briefHash, string name);
    event Decision(uint256 indexed id, uint32 indexed seq, uint32 tMs, uint8 action, uint8 proposed,
                   uint16 riskBps, bool overridden, bytes32 entryHash, bytes32 head);
    event MissionEnded(uint256 indexed id, uint32 decisions, uint32 overrides, bytes32 head, bytes32 reportHash);

    error NotOperator();
    error MissionClosed();

    modifier onlyOpen(uint256 id) {
        Mission storage m = missions[id];
        if (m.operator != msg.sender) revert NotOperator();
        if (m.endedAt != 0) revert MissionClosed();
        _;
    }

    function startMission(string calldata name, bytes32 briefHash) external returns (uint256 id) {
        id = missions.length;
        Mission storage m = missions.push();
        m.operator = msg.sender;
        m.startedAt = uint64(block.timestamp);
        m.briefHash = briefHash;
        m.name = name;
        emit MissionStarted(id, msg.sender, briefHash, name);
    }

    /// @param tMs       flight time of the decision in milliseconds
    /// @param action    final action after the safety guard (index into the copilot's ACTIONS list)
    /// @param proposed  what the AI proposed before the guard
    /// @param riskBps   AI risk estimate, 0..10000
    /// @param entryHash keccak256 of the canonical JSON of the full log entry (scene + decision + reason)
    function logDecision(uint256 id, uint32 tMs, uint8 action, uint8 proposed, uint16 riskBps,
                         bool overridden, bytes32 entryHash) external onlyOpen(id) {
        Mission storage m = missions[id];
        bytes32 h = keccak256(abi.encodePacked(m.head, tMs, action, proposed, riskBps, overridden, entryHash));
        m.head = h;
        uint32 seq = m.decisions++;
        if (overridden) m.overrides++;
        emit Decision(id, seq, tMs, action, proposed, riskBps, overridden, entryHash, h);
    }

    function endMission(uint256 id, bytes32 reportHash) external onlyOpen(id) {
        Mission storage m = missions[id];
        m.endedAt = uint64(block.timestamp);
        m.reportHash = reportHash;
        emit MissionEnded(id, m.decisions, m.overrides, m.head, reportHash);
    }

    function missionCount() external view returns (uint256) {
        return missions.length;
    }
}
