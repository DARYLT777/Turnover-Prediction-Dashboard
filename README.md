# AI-Enabled Predictive Turnover Dashboard (MVP)

This repository contains a small, working prototype of an AI-enabled predictive turnover dashboard
for **IOP 563 — AI in I-O Psychology (Milestone 3)**.

The app is built with **Streamlit + Python** and uses **synthetic data** to simulate a mid-sized
hospitality organization (hotels, food & beverage, spa).

## How to Run Locally

1. Clone this repo.
2. Create a virtual environment (optional but recommended).
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
streamlit run main.py
rivacy & Governance

All data in this MVP is synthetic (no real employee data).

The app enforces an aggregation rule: groups with fewer than 5 employees are not displayed to protect privacy.

The dashboard is intended for proactive retention and well-being support, not for discipline or termination.

Model outputs should always be combined with human HR judgment and organizational policies.

Purpose

This MVP supports Milestone 3 by providing:

A working end-to-end slice of an AI-enabled predictive turnover tool.

Basic fairness checks (Adverse Impact Ratio by tenure band).

A basic validity check (correlation between predicted risk and actual turnover).
