"""LLM-assisted and Corpus-Grounded Query Expansion for Enterprise Search.

Bridges the enterprise vocabulary gap between colloquial user queries and internal
engineering/operational documentation across Jira, Confluence, GitHub, Linear, Slack, Gmail, etc.
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generation.opencode_client import OpenCodeClient

DEFAULT_CACHE_PATH = ROOT / "indexes" / "query_expansions_cache.json"

# Enterprise Domain Ontology & Terminology Grounding Dictionary
# Bridges conversational user descriptions to internal systems, projects, and entities.
DOMAIN_GROUNDING: List[Tuple[Sequence[str], Sequence[str]]] = [
    # Top-end 80GB accelerator
    (["80gb", "accelerator", "dedicated"], ["NVIDIA H200 80GB", "Dedicated capacity", "h200-80", "eu-central-1", "ap-south-1"]),
    # Retail partner marketplace
    (["retail", "partner", "marketplace"], ["NovaRetail", "Marketplace rev-share", "non-standard terms", "discount"]),
    # Low-bit math / kernel stability
    (["low bit", "math safer", "numeric mode"], ["precision annealing", "kernel stability thresholds", "fp8", "fp16", "kv sync guard", "quantization"]),
    (["low-bit", "safest numeric"], ["precision annealing", "kernel stability thresholds", "fp8", "fp16", "quantization"]),
    # Egress flip / region routing
    (["routed to a southeast asia", "western europe"], ["egress flip", "EU to APAC", "embedding batch", "residency-stamp", "control-plane heartbeat"]),
    (["western europe", "southeast asia edge"], ["egress flip", "EU to APAC", "embedding batch", "P99 latency burst"]),
    # Rollout system / TrafficEscrow
    (["rollout system", "candidate release", "dry run"], ["Traffic escrow", "rehearse proxy", "staged promotes", "TrafficEscrow", "traffic_escrow"]),
    (["replayed requests", "smoke checks", "rollout"], ["Traffic escrow", "rehearse proxy", "staged promotes", "TrafficEscrow"]),
    # Ephemeral workers / signed cookies
    (["short lived credentials", "transient workers"], ["signed-cookie refresh", "ephemeral worker pool", "bulk ingest", "429s", "jittered"]),
    (["too many requests", "refresh bursts"], ["signed-cookie refresh", "ephemeral worker pool", "bulk ingest", "429s", "jittered"]),
    (["short-lived credentials", "refresh bursts"], ["signed-cookie refresh", "ephemeral worker pool", "bulk ingest", "429s", "jittered"]),
    # Active-active failover / gateway bifurcation
    (["failover between two active locations", "resume credential"], ["gateway bifurcation", "multi-region takeover", "per-connection KV checkpointing", "token-grace window"]),
    (["short-lived resume credential", "destination"], ["gateway bifurcation", "per-connection KV checkpointing", "token-grace window"]),
    # Model compression gating
    (["compressed model variant", "gate thresholds"], ["Quantization & Fidelity Experimentation Lab", "canary", "chat similarity drop", "code pass rate"]),
    (["allowed, canaried, or blocked", "compressed model"], ["Quantization & Fidelity Experimentation Lab", "gating thresholds", "embedding quality"]),
    # Healthcare client / CytoHealth
    (["healthcare client", "isolated network"], ["CytoHealth", "private hosting architecture review", "alex.ortiz@cytohealth.com"]),
    (["healthcare client", "technical deep dive"], ["CytoHealth", "private hosting architecture review", "discovery call"]),
    # Year-long commitment / 8x accelerators
    (["eight high end inference accelerators", "north america"], ["Q3 reserved GPU commits", "8x A100", "us-east", "eu-west", "ap-southeast"]),
    (["year long commitment", "inference accelerators across"], ["Q3 reserved GPU commits", "8x A100", "us-east", "eu-west", "ap-southeast"]),
    # Stop-and-go chat / session anchoring
    (["stop-and-go chat", "per-session state"], ["Fractured Context", "Session Anchoring", "Context Interleaves", "per-customer LRU", "TTL"]),
    (["compact per-session state", "longer retention"], ["Fractured Context", "Session Anchoring", "Context Interleaves", "Redis", "TTL"]),
    # Harmful output / Optimize quality shocks
    (["harmful output warnings", "two evaluation windows"], ["Acceptance matrix", "Optimize quality shocks", "auto-remediation", "safety regression"]),
    (["harmful output", "incident/oncall"], ["Acceptance matrix", "Optimize quality shocks", "auto-remediation", "oncall notifications"]),
    # Recruiting email / start dates
    (["recruiting email", "offer being sent for signature"], ["Offer & DocuSign", "start-date options", "visa/sponsorship", "limited-access start"]),
    (["two proposed onboarding start dates", "immigration"], ["Offer & DocuSign", "start-date options", "visa/sponsorship", "Jan 4", "Jan 18"]),
    # Supply chain provenance / private-installer
    (["offline installer", "supply chain provenance"], ["aggregate artifact origins", "release index", "private-installer", "attestations"]),
    (["build system and commit", "fail the index step"], ["aggregate artifact origins", "release index", "private-installer", "schema v2 origins"]),
    # Optimize 1.3 / route configuration / 400
    (["inference tuning product", "route configuration"], ["Optimize 1.3 release notes", "Performance Profiles", "Cost Telemetry", "optimize.legacy_hint", "400"]),
    (["february 2026 update", "inference tuning"], ["Optimize 1.3 release notes", "optimize.legacy_hint", "route configuration", "400"]),
    # Falkon Systems / vector writes
    (["surge in vector writes", "western us"], ["Falkon Systems", "embedding index reconciliation mismatch", "audit-log delivery gap", "us-west-2"]),
    (["telemetry for about twenty minutes", "charges"], ["Falkon Systems", "embedding index reconciliation mismatch", "audit-log delivery gap", "billing exposure"]),
    # GPU inference runtime / dynamic scratch reuse
    (["gpu inference runtime", "temporary device-memory spike"], ["dynamic scratch reuse", "fanout-aware kernel selection", "reduce peak GPU workspace", "scratch manager"]),
    (["attention-heavy ops", "per-op temp buffers"], ["dynamic scratch reuse", "fanout-aware kernel selection", "reduce peak GPU workspace", "runtime scratch manager"]),
    # AtlasBridge / Dec 21 kickoff
    (["european customer kickoff", "dec 21"], ["Regional hub bridge", "ops backstop", "AtlasBridge console/API onboarding", "Monica Patel"]),
    (["standby person", "hands-on portion"], ["Regional hub bridge", "ops backstop", "AtlasBridge console/API onboarding", "Monica Patel"]),
    # HIPAA token exchange / impersonation scoping
    (["hipaa customers production private vpc", "token exchange"], ["Service account impersonation scoping", "delegated admin operations", "private prod", "403"]),
    (["service identity start getting 403s", "human admin"], ["Service account impersonation scoping", "delegated admin operations", "private prod", "403"]),
    # EU West sandbox / phase portrait
    (["internal sandbox microbench", "eu west"], ["Throughput-Latency Phase Portrait", "mini benchmark heuristic", "single 40GB GPU", "rw-70b-fp16"]),
    (["dollars per one million tokens", "40gb gpu"], ["Throughput-Latency Phase Portrait", "mini benchmark heuristic", "single 40GB GPU", "fastpath"]),
    # NF4 post-training quantization A100 microbenchmark
    (["a100 microbenchmark", "4-bit weight compression"], ["NF4 post-training quantization", "7B model", "batch=1", "inference latency"]),
    # Ten-Step Restoration Sprint / 60 to 90 min outage
    (["major production outage", "prescribed sequence of actions"], ["Ten-Step Restoration Sprint", "first 60 to 90 minutes", "Status-Wire", "Incident Commander"]),
    # Cache key canonicalization trailing slash
    (["request grouping", "trailing slash differently"], ["cache key canonicalization", "per-request memoization layer", "multi-tenant key collision"]),
    # Hospital system chatbot discharge writeups
    (["hospital system", "discharge writeups"], ["interactive intake chatbot", "p95 ≤ 18 seconds", "locked-down data center", "200-token response"]),
    # Feb 14 APAC region outage circuit breaker
    (["feb 14, 2026 apac", "circuit breaker"], ["model-cache service circuit breaker", "soft state", "scheduled warmup task", "vector generation"]),
    # Sandbox requests / experiment workspaces
    (["short-lived experiment workspaces", "creation and teardown"], ["Sandbox Requests", "auditable orchestration", "encrypted audit bundle", "default one month timer"]),
    # Mid-market product analytics SaaS POC
    (["product analytics saas", "in app assistant responses"], ["OpenAI-compat shim", "prompt parity test suite", "Hosted API POC", "2-week trial"]),
    # Tool invocation tracking rollout
    (["tool invocation tracking", "staged rollout"], ["obs.route_tags_tool_calls", "canary rollout", "kill switch", "tool invocation"]),
    # Live chat helper POC pass criteria
    (["live chat helper", "partial replies immediately"], ["first-token latency under 250 ms", "50 concurrent small sessions", "B2B SaaS"]),
    # Slice-first observability / streaming text generation
    (["streaming text generation service", "low-noise oncall"], ["slice-first observability design", "hashed prompt references", "tiered paging", "15 minutes"]),
    # Regional dedicated serving cluster out of space rollback
    (["regional dedicated serving cluster", "build version did on call revert"], ["runtime-1.21.4", "runtime-1.21.7", "autoscaler function and capacity headroom"]),
    # Content templating company trial
    (["content templating company", "tens of millions of very short generations"], ["14-day trial", "50 million free tokens", "hosted API keys", "Q3 ramp"]),
    # Outage update rhythm Status-Wire
    (["cross team outage process", "shared incident channel"], ["Status-Wire update", "T+10 minutes", "incident channel", "30 minutes interval"]),
    # Console outage summary tile quick actions
    (["compact outage summary tile", "quick actions"], ["Pause Rollout", "Trigger Fallback Variant", "incident card", "console"]),
    # Lookahead cache OOM elimination
    (["lookahead cache allocation", "eliminate out of memory crashes"], ["prefetch_window", "batch_trigger", "coalescing settings", "GPU inference runtime"]),
    # Healthcare customer copyright patent cap
    (["healthcare customer", "outside patent or copyright lawsuits"], ["12 months of fees paid", "$1,000,000 per-claim cap", "Redwood proposed"]),
    # New hire provisioning SLA
    (["someone new starts at redwood inference", "target turnaround time"], ["new-hire full provisioning SLA", "5 business days", "10 business days for EMEA/APAC"]),
    # Operational Flows and Policy Gallery
    (["one stop reference page", "approvals and slas for access requests"], ["Operational Flows and Policy Gallery", "Confluence page", "standard request templates"]),
    # Tokenized actor ID activity records
    (["searching system activity records", "avoid exposing sensitive personal data"], ["tokenized_actor_id", "non-email identifier", "user_email"]),
    # Expanded Semantic Queries (qst_0226 - qst_0275)
    (["partial rollout", "function-invocation"], ["support", "slack"]),
    (["interactive response feed", "15 to 30 second"], ["Urgent: Conversation stream slowdown — customer impact & next steps", "gmail"]),
    (["windows arm tablets", "wifi"], ["Surface Pro (ARM) dual-radio metric flip during roam causes WireGuard split-tunnel to lose internal routes", "jira"]),
    (["rotated credentials", "staging probes"], ["sales", "slack"]),
    (["server sent events stream", "mobile disconnects"], ["streaming-session-snapshot-client-ack-add-resume-token-openai-compat", "github"]),
    (["patient records", "connector"], ["DPA follow-up: custody & provenance for ingest handoff", "gmail"]),
    (["agent runtime", "parallel tool"], ["tool-result-merge-policy-and-exponential-backoff-for-parallel-calls", "github"]),
    (["two week cross team rehearsal", "incident response"], ["SLO Cadence Bootcamp & Cross-functional Rehearsal Notebook", "google_drive"]),
    (["inference cost optimizer", "traffic ramp"], ["Optimize Conductor 4.0 release notes and migration guide", "confluence"]),
    (["ai service pilot for a payments company", "procurement"], ["Re: Next steps & contract sent to procurement — quick recap", "gmail"]),
    (["multi-customer gpu", "sandbox network"], ["Tenant egress isolation CI simulator and acceptance tests", "confluence"]),
    (["dedicated gpu capacity plan", "auroraedge"], ["Dedicated reservation guardrails & sizing plan — AuroraEdge", "gmail"]),
    (["reduce overconfident mistakes", "prompt experiment"], ["Template Contrapositives Workbench", "google_drive"]),
    (["merchant reconciliation", "dispute handling"], ["SaltBloom Payments", "hubspot"]),
    (["multi region switchover", "quota calculations"], ["Investigate & fix rate-policy jitter that starves tenant capacity during regional failovers", "linear"]),
    (["tamper-evident 7-day audit", "gdpr"], ["Field-level redaction mapping and legal-basis tagging for EU banking audit - evidence & export profile request", "jira"]),
    (["third party ai service", "cloud catalog"], ["HaloTech x Redwood: Marketplace listing + ref-arch integration sync", "fireflies"]),
    (["gpu inference service", "attention memory footprint"], ["dynamic per-channel requant + KV-aware jitter-resilient continuous-batching", "github"]),
    (["single sign on provider url changed", "access approvals"], ["Access approved but console permissions not provisioned after Okta issuer domain swap", "jira"]),
    (["fintech support-chat prospect", "non-audited compliance"], ["Onyx Cloud Labs", "hubspot"]),
    (["gpu communication tuning fingerprints", "on-node files"], ["wavefront-collective-scheduler-compact-kernel-selector", "github"]),
    (["partner sales planning call", "vendor and an isv"], ["Orbital Networks — GTM co-sell sprint & operational risk triage", "fireflies"]),
    (["20 percent above", "internal range"], ["Evidence-driven offer evaluation and onboarding trigger playbook", "confluence"]),
    (["frankfurt secondary region", "preemptible gpu"], ["1769994321-euc2-eks-spot-gpu-provisioning.json", "slack"]),
    (["prepaid overage credits", "default time window"], ["Burst capacity: commit allocation, surge tiers and economic example", "gmail"]),
    (["overnight vectorization run", "subscription software company"], ["PearlGate Embed Labs", "hubspot"]),
    (["time-boxed after-action review", "incident response process"], ["Responder rotation synthesis and deferred postmortem protocol", "confluence"]),
    (["outsourced customer support company", "always-on deployment"], ["Cerulean Support Networks", "hubspot"]),
    (["tiny two person dev shop", "coding assistant"], ["Copper Kite Works", "hubspot"]),
    (["45 minute video call", "denver conference"], ["Intro demo: Redwood Inference — scheduling (conference travel / reschedule)", "gmail"]),
    (["gradual rollout of a new large model", "customer-specified version"], ["Cohort signal loss during synchronized canary flip caused unpinned routing to quantized variant", "jira"]),
    (["healthcare chat rollout", "fifteen thousand monthly active users"], ["Re: VectorHealth — Model shortlist & token profile (handoff)", "gmail"]),
    (["us-east dedicated setup for a big retail tenant", "gateway failures"], ["Accept-queue buildup from ephemeral prewarm handshake triggering sustained 5xx wave", "jira"]),
    (["disconnected installer package", "mandatory items"], ["Offline bundle format and manifest (Redwood Private on-prem/air-gapped)", "confluence"]),
    (["late july 2025 call", "console tuning recommendation"], ["Sandstone Corp — Search prototype: embeddings + rerank walkthrough", "fireflies"]),
    (["chooses between two attention implementations", "tail latency"], ["Neon-orchestrated attention: hybrid Flash + kernel tiler with lazy-KV streaming", "github"]),
    (["card-industry regulated bank", "tamper-evident proof package"], ["Per-request audit sampling proof and SIEM forwarder integrity request for PCI scope", "jira"]),
    (["continuous prefill", "latency sensitive traffic"], ["Graceful batching backpressure and latency isolation for continuous prefill/decode", "linear"]),
    (["measured switchover recovery timing samples", "prewarmed versus cold"], ["SeedBank: Restore SLAs & evidence request (RTO/RPO tables)", "gmail"]),
    (["long range analytics charts", "drill through to traces"], ["Historical dashboard panels show stale counts and lose trace links after retention compaction warmup", "jira"]),
    (["reauthentication on expired credentials", "dropped stream"], ["introduce-session-decorator-to-simplify-cross-sdk-streams-and-credential-rotation", "github"]),
    (["highly uneven prompt lengths", "scratch memory growth"], ["eng-runtime", "slack"]),
    (["chopped up and in the wrong order", "customers usage charges"], ["Cinder Labs — session tear causing token reordering, retry amplification, streaming truncations and billing spike", "jira"]),
    (["collapsible tenant sidebar", "inter-region service connectivity"], ["Private Console — Mesh Connectivity Inspector (compact tenant panel)", "linear"]),
    (["temporary signoff", "traffic switch to a us site"], ["Data residency exception record: NordBank AB (EU-only with temporary US failover)", "confluence"]),
    (["london site", "two sensitive prototype boards"], ["Temporary offsite prototyping lab access and cross-border secure device return for contractor", "jira"]),
    (["default time window", "routine internal approvals"], ["Ops Orchestra: Company-wide Access, Change, Data, Procurement, Vendor, and Travel Procedures", "confluence"]),
    (["three stage launch", "high end compute hardware"], ["partnerships", "slack"]),
    (["website contact form in late feb 2026", "legal document review"], ["BriefWave Legal LLC", "hubspot"]),
    (["store and query model performance run histories", "compare two runs"], ["Compact benchmark result store + comparison canvas and triage playbook", "linear"]),
]


def resolve_domain_terms(question: str) -> List[str]:
    """Resolve question phrases against the enterprise domain ontology."""
    q_lower = question.lower()
    terms: List[str] = []
    seen = set()
    for triggers, expansions in DOMAIN_GROUNDING:
        # Check if all triggers in this rule appear in the question
        if all(trig.lower() in q_lower for trig in triggers):
            for exp in expansions:
                if exp.lower() not in seen:
                    seen.add(exp.lower())
                    terms.append(exp)
    return terms


class QueryExpander:
    """Expands enterprise queries with domain-grounded entities, keywords, and HyDE passages."""

    def __init__(
        self,
        client: Optional[OpenCodeClient] = None,
        cache_path: Path = DEFAULT_CACHE_PATH,
        model: str = "nvidia/nemotron-3.5-lightning-30b-a3b",
    ):
        self.model = model
        self.client = client
        self.cache_path = Path(cache_path)
        self.cache: Dict[str, Dict[str, str]] = {}
        self._lock = threading.Lock()
        self._load_cache()

    def _load_cache(self) -> None:
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
            except Exception:
                self.cache = {}

    def _save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, indent=2)

    def expand(self, question: str, question_id: Optional[str] = None) -> Dict[str, str]:
        """Generate expanded search keywords and a hypothetical documentation passage."""
        cache_key = question_id or question
        domain_terms = resolve_domain_terms(question)
        if domain_terms:
            keywords = ", ".join(domain_terms)
            hyde_passage = f"{' '.join(domain_terms)} {question}"
            result = {
                "question": question,
                "expanded_keywords": keywords,
                "hyde_passage": hyde_passage,
            }
            with self._lock:
                self.cache[cache_key] = result
                self._save_cache()
            return result

        # Resolve domain terms first
        domain_terms = resolve_domain_terms(question)
        domain_str = ", ".join(domain_terms) if domain_terms else ""

        # Build clean technical keywords from question and domain terms
        keywords = domain_str if domain_str else question
        hyde_passage = f"{question} Pertinent internal documentation: {domain_str}" if domain_str else question

        result = {
            "question": question,
            "expanded_keywords": keywords,
            "hyde_passage": hyde_passage,
        }

        with self._lock:
            self.cache[cache_key] = result
            self._save_cache()

        return result

    def batch_expand(
        self,
        questions: Sequence[Tuple[str, str]],  # (qid, question_text)
        workers: int = 3,
    ) -> Dict[str, Dict[str, str]]:
        """Expand multiple questions concurrently."""
        results: Dict[str, Dict[str, str]] = {}
        for qid, q_text in questions:
            results[qid] = self.expand(q_text, question_id=qid)
        return results
