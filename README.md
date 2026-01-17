# Snoonu Smart Dispatch (Hackathon)

A lightweight **dispatch simulation + dashboard** to compare two policies on the **same scenario**:

- **Baseline (Myopic):** nearest / single-order assignment  
- **Batch Dispatch (OBA):** safe bundling + multi-stop routing (with guardrails)

This tool is built to **de-risk batching** by replaying scenarios and comparing KPI trade-offs before production rollout.

---

## Quick Start

### 1) Backend (FastAPI)

```bash
cd snoonu-hackathon
python -m uvicorn app.main:app --reload --port 8000
