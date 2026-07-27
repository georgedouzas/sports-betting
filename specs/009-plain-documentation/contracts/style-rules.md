# Contract: The Plain Style Rules

These rules replace the terse literary voice. They are concrete, so a reviewer can check them by eye. They become the
constitution's Writing Style rules and the review checklist.

## Do

- Write short, direct sentences. One idea per sentence.
- Say who does what in the normal order: subject, then verb, then object. "The library places the bet at the venue."
- Use plain words. Prefer the common word over the clever one.
- State a real risk once, in the right place, plainly. Then move on.
- Give each page headings and subsections, so a reader can scan it. Split long blocks.
- Make every code example run. Use sample data, a placeholder, or a fake.

## Do not

- Do not invert word order for effect. Not "A venue with an API is placed at by the library."
- Do not use the passive voice unless the doer is unknown or does not matter.
- Do not use idioms, metaphors, or literary flourish.
- Do not hedge or repeat a warning. Do not sound defensive.
- Do not write a page as one undivided block of text.
- Do not write a code example that cannot run. No fragments, no pseudo-code, no demo that needs a secret or the network.

## Kept from before

- A line is at most 120 characters.
- A sentence uses no semicolon and no dash as punctuation. Hyphenated words are fine.

## Rewrite examples

- Before: "A venue with an official betting API is placed at by the library."
  After: "The library places bets at a venue that has a betting API."
- Before: a single block that mixes the basic install, the MCP extra, the execution extra, and the development install.
  After: four subsections, one per install, each with its own heading and its own commands.
- Before: "A bettor finds value bets and stops. Execution acts on one of them. The single-event unit takes a fitted
  bettor, one upcoming match, and a fixed stake, watches that one event, and at the moment the model was fitted for
  places the model's bet, once, at a bookmaker where you hold an account."
  After: "A bettor finds value bets. The single-event unit places one of them. It takes a fitted bettor, one match, and
  a stake. It watches the match. At the moment the model was fitted for, it places the bet, once. You place the bet at a
  bookmaker where you hold an account. This spends real money."
