"""MARITRACE Cryptographic Audit Logging and Provenance Trail.

Implements a SHA-256 tamper-evident hash-chained audit log that tracks
dataset checksums, model checkpoints, preprocessing parameters, fusion decisions,
and forensic abstentions.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple


class AuditRecord:
    """Individual tamper-evident record in the audit chain."""

    def __init__(
        self,
        index: int,
        timestamp_utc: str,
        event_type: str,
        actor: str,
        details: Dict[str, Any],
        previous_hash: str,
        record_hash: Optional[str] = None,
    ):
        self.index = index
        self.timestamp_utc = timestamp_utc
        self.event_type = event_type
        self.actor = actor
        self.details = details
        self.previous_hash = previous_hash
        self.hash = record_hash or self.calculate_hash()

    def calculate_hash(self) -> str:
        """Compute SHA-256 hash over canonical serialized fields."""
        payload = {
            "index": self.index,
            "timestamp_utc": self.timestamp_utc,
            "event_type": self.event_type,
            "actor": self.actor,
            "details": self.details,
            "previous_hash": self.previous_hash,
        }
        canonical_str = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary representation."""
        return {
            "index": self.index,
            "timestamp_utc": self.timestamp_utc,
            "event_type": self.event_type,
            "actor": self.actor,
            "details": self.details,
            "previous_hash": self.previous_hash,
            "hash": self.hash,
        }


class AuditTrail:
    """Tamper-evident hash-linked log with verification capabilities."""

    def __init__(self, log_dir: str = r"D:\dataset maritrace\outputs\audit"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.chain: List[AuditRecord] = []
        self._initialize_genesis()

    def _initialize_genesis(self) -> None:
        """Create genesis block if chain is empty."""
        if not self.chain:
            genesis = AuditRecord(
                index=0,
                timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                event_type="SYSTEM_GENESIS",
                actor="MARITRACE_CORE",
                details={
                    "system": "MARITRACE Maritime Pollution Forensics",
                    "provenance_policy": "ISO 27037 Digital Evidence Admissibility Guidelines",
                    "disclaimer": (
                        "A cryptographic hash chain guarantees post-hoc tamper-evident "
                        "logging of recorded events, but does not independently prove the "
                        "truthfulness or physical accuracy of upstream sensor inputs."
                    ),
                },
                previous_hash="0" * 64,
            )
            self.chain.append(genesis)

    def log_event(
        self,
        event_type: str,
        details: Dict[str, Any],
        actor: str = "MARITRACE_PIPELINE",
    ) -> AuditRecord:
        """Append a new event block to the tamper-evident chain."""
        prev_record = self.chain[-1]
        new_record = AuditRecord(
            index=len(self.chain),
            timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            event_type=event_type,
            actor=actor,
            details=details,
            previous_hash=prev_record.hash,
        )
        self.chain.append(new_record)
        self.save()
        return new_record

    def verify_integrity(self) -> Tuple[bool, Optional[str]]:
        """Verify that the cryptographic hash chain is uncompromised and sequential."""
        for i in range(len(self.chain)):
            rec = self.chain[i]
            # Check recomputed hash
            recomputed = rec.calculate_hash()
            if rec.hash != recomputed:
                return (
                    False,
                    f"Block {i} hash mismatch: recorded={rec.hash}, recomputed={recomputed}",
                )

            # Check pointer to previous block
            if i > 0:
                prev_rec = self.chain[i - 1]
                if rec.previous_hash != prev_rec.hash:
                    return (
                        False,
                        f"Block {i} previous_hash mismatch: expected={prev_rec.hash}, got={rec.previous_hash}",
                    )
        return True, None

    def save(self, filename: str = "audit_log.json") -> str:
        """Persist audit log to disk."""
        filepath = os.path.join(self.log_dir, filename)
        data = [rec.to_dict() for rec in self.chain]
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return filepath

    def load(self, filepath: str) -> bool:
        """Load audit trail from JSON file."""
        if not os.path.exists(filepath):
            return False
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.chain = [
            AuditRecord(
                index=item["index"],
                timestamp_utc=item["timestamp_utc"],
                event_type=item["event_type"],
                actor=item["actor"],
                details=item["details"],
                previous_hash=item["previous_hash"],
                record_hash=item["hash"],
            )
            for item in data
        ]
        return True


if __name__ == "__main__":
    audit = AuditTrail()
    audit.log_event("TEST_EVENT", {"status": "operational", "test_id": 101})
    valid, err = audit.verify_integrity()
    print(f"Audit log integrity check passed: {valid} (Records: {len(audit.chain)})")
