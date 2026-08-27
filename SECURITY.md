# Security policy and deployment boundary

## Supported boundary

This repository is an experimental research snapshot. The tested deployment model is one trusted training job in an
isolated Ray namespace and private cluster network. It is **not** a multi-tenant service and must not share a Ray
namespace with untrusted or independently administered jobs.

Ray Serve control routes, the Ray dashboard, SGLang engines, Judge endpoints, metrics endpoints, the MobileGym
gateway, and per-session OpenAI-compatible routes must not be exposed to the public Internet. Framework lifecycle
routes such as `/stop_service` assume a trusted control plane and do not implement end-user authorization.

Use network policy, firewall rules, and a job-specific Ray namespace. Treat a same-name actor or Serve application as
a configuration error; do not rely on this snapshot to provide multi-tenant ownership isolation or transactional
cleanup after every partial initialization failure.

## Sensitive artifacts

Training prompts, tool observations, screenshots, Judge rationales, session identifiers, model paths, runtime
environments, placement manifests, timeline traces, GPU inventories, and scheduler logs may contain confidential
data or infrastructure metadata. Store them in access-controlled locations and sanitize them before publication.

MobileGym passes the Relax session identifier to its subprocess as a local bearer credential. Run rollout workers
only on trusted hosts where same-user process inspection is within the accepted threat model.

## Resource limits

The Judge client validates media count and decoded-byte budgets from the selected service config, and the GenRM proxy
rechecks those limits before decoding media. An outer HTTP proxy or network policy should additionally enforce a
request-body limit. For multimodal models, `max_input_tokens` is currently a text-prompt guard and does not account
for the model-specific visual-token expansion; SGLang's context and memory limits remain authoritative.

Per-turn mode can retain multiple pending Judge tasks and media references until the trajectory is finalized. Bound
episode length, concurrent sessions, media count, media bytes, request timeout, and terminal barrier time according
to the target deployment.

## Known unverified areas

- fragmented-GPU placement for tensor-parallel Judge engines;
- rollback of every Ray actor, placement group, and service after partial Controller initialization;
- active health recovery after a Judge/SGLang child-process failure;
- stale completion files and abrupt head-node termination in the multi-node launcher;
- concurrent jobs sharing one Ray namespace;
- full real-model agentic lifecycle through group replacement, TransferQueue, and trainer consumption.

These are deployment gates, not capabilities implied by the unit-test suite.

## Reporting a vulnerability

Please use GitHub's private vulnerability-reporting feature for the repository when available. Do not include real
credentials, private trajectories, screenshots, or internal cluster identifiers in a public issue.
