# Specification Quality Checklist: Bet execution

**Purpose**: Validate specification completeness and quality before proceeding to planning
**Created**: 2026-07-16
**Feature**: [spec.md](../spec.md)

## Content Quality

- [x] No implementation details (languages, frameworks, APIs)
- [x] Focused on user value and business needs
- [x] Written for non-technical stakeholders
- [x] All mandatory sections completed

## Requirement Completeness

- [x] No [NEEDS CLARIFICATION] markers remain
- [x] Requirements are testable and unambiguous
- [x] Success criteria are measurable
- [x] Success criteria are technology-agnostic (no implementation details)
- [x] All acceptance scenarios are defined
- [x] Edge cases are identified
- [x] Scope is clearly bounded
- [x] Dependencies and assumptions identified

## Feature Readiness

- [x] All functional requirements have clear acceptance criteria
- [x] User scenarios cover primary flows
- [x] Feature meets measurable outcomes defined in Success Criteria
- [x] No implementation details leak into specification

## Notes

- No venue, SDK, browser driver or library was named anywhere in the spec at authoring time.
  Those are technology choices and belong to `/speckit-plan` research, which settled them on
  Betfair and Playwright. See [research.md](../research.md) D1 and D4.
- The Assumptions section now names Betfair, added after planning. Judged to pass: planning
  falsified two of the spec's assumptions, and an Assumptions section that keeps asserting a
  false thing is worse than one that names what resolved it. The requirements themselves stay
  venue-agnostic, and the venue contract (FR-005) is what every venue implements.
- The spec does name `sportsbet.execution`, the bettor, and the three surfaces (Python
  API, CLI, MCP server). For this project these are domain vocabulary rather than
  technology choices: the product *is* a library, its surfaces are mandated by
  Constitution Principle I, and FR-001/FR-002 exist precisely to fix an architectural
  boundary (placement must not sit on an estimator). Judged to pass in substance.
- Zero [NEEDS CLARIFICATION] markers were needed at authoring time. A `/speckit-clarify`
  session on 2026-07-16 then resolved the five highest-impact open decisions and wrote them
  into the spec: the site-driven scope (FR-007), who may authorise real money (FR-009), where
  the placement record lives (FR-015), what makes two bets the same (FR-014), and how the
  worst acceptable price is expressed (FR-013). See the Clarifications section of the spec.
- RESOLVED, and both answers surprised us. The reference venue is Betfair, and it is the only
  exchange of four surveyed that can carry a caller reference at all, so FR-015 was
  unimplementable at the others. It takes two references rather than one, and neither alone
  satisfies FR-014, so the contract carries both.
- FALSIFIED: the spec assumed the reference exchange "offers a sandbox or equivalent test
  facility". No exchange does. Betfair's delayed application key is widely believed to be a
  sandbox and is not one: it places real bets on the live exchange. FR-026 already forbade
  touching a venue, so no requirement changed, but the reason is now written into the spec so
  nobody relaxes it later.
- FALSIFIED: planning assumed the sanctioned path was the one the maintainer would use. All four
  surveyed exchanges are closed to Greece, verified empirically. This reorders which user story
  serves whom without changing the spec, and both paths still ship.
- Constitution alignment: the feature was checked against v1.1.0. Principle I is preserved
  by FR-001/FR-002 (bettors stay estimators) and FR-020 (surface parity), and the
  agent-is-a-client rule by FR-022. The optional-extra rule for credentialled side effects
  is carried by FR-004.
- The money-safety requirements (FR-009 to FR-016) and the boundary requirements
  (FR-023, FR-024) are the ones most worth a reviewer's attention. FR-023 forbidding any
  evasion of a venue's automation controls is a hard boundary, not a preference.
- CONTRADICTION found during planning, and it needs a reviewer's eye before `/speckit-tasks`.
  FR-005 requires every venue to implement `place`. FR-007 requires the site-driven path to be
  generic primitives with the agent supplying site knowledge. A `place` on a website has to find
  the market, click the price, fill the stake and confirm, all of which is site knowledge, so it
  is either per-site code (FR-007 forbids) or an LLM in the package (FR-022 forbids). The plan
  resolves this by making the site path a `BrowserSession` that is not a venue and offers no
  `place`. Consequence a reviewer must accept or reject: FR-014's once-only and FR-011's ceilings
  hold on the API path and are the agent's responsibility on the site path. The docs state this
  rather than implying parity. If that is unacceptable, FR-007 is what has to change.
- OPEN, recorded in [plan.md](../plan.md) rather than decided: in-play. The session design serves
  live markets, but bettors are fitted on pregame odds and cannot see a score, so an in-play stake
  would trace back through FR-016 to a value bet that stopped being true at kickoff. The plan
  recommends in-play stays out of scope for 006.
