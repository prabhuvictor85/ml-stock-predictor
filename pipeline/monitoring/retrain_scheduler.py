"""
RetrainingScheduler — queues and manages model retraining runs.
Triggered by FeatureDriftMonitor when PSI thresholds are breached.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from pipeline.utils.logging import get_logger

log = get_logger(__name__)

QUEUE_FILE = Path("monitoring") / "retrain_queue.json"


class RetrainingScheduler:
    """
    Simple file-backed retrain queue.

    Usage:
        scheduler = RetrainingScheduler(retrain_fn=run_training_pipeline)
        scheduler.queue("drift_triggered", market="nse")
        scheduler.process_queue()
    """

    def __init__(self, retrain_fn: Optional[Callable] = None) -> None:
        self.retrain_fn = retrain_fn
        QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)

    def queue(self, reason: str, market: str) -> None:
        """Add a retrain job to the queue."""
        entry = {
            "queued_at": datetime.utcnow().isoformat(),
            "reason": reason,
            "market": market,
            "status": "pending",
        }
        jobs = self._load_queue()
        # Deduplicate: don't re-queue if pending job already exists for same market
        pending = [j for j in jobs if j["market"] == market and j["status"] == "pending"]
        if pending:
            log.info(f"Retrain already queued for market='{market}', skipping duplicate.")
            return
        jobs.append(entry)
        self._save_queue(jobs)
        log.warning(f"Retrain queued: market='{market}', reason='{reason}'")

    def process_queue(self) -> None:
        """Execute all pending retrain jobs."""
        jobs = self._load_queue()
        for job in jobs:
            if job["status"] != "pending":
                continue
            log.info(f"Processing retrain job: {job}")
            job["status"] = "running"
            self._save_queue(jobs)
            try:
                if self.retrain_fn is not None:
                    self.retrain_fn(market=job["market"])
                    job["status"] = "completed"
                    job["completed_at"] = datetime.utcnow().isoformat()
                    log.info(f"Retrain completed for market='{job['market']}'")
                else:
                    log.warning("No retrain_fn registered. Job will remain as 'pending'.")
                    job["status"] = "pending"
            except Exception as e:
                job["status"] = "failed"
                job["error"] = str(e)
                log.warning(f"Retrain failed for market='{job['market']}': {e}")
            self._save_queue(jobs)

    def _load_queue(self) -> list:
        if QUEUE_FILE.exists():
            with open(QUEUE_FILE) as f:
                return json.load(f)
        return []

    def _save_queue(self, jobs: list) -> None:
        with open(QUEUE_FILE, "w") as f:
            json.dump(jobs, f, indent=2)

