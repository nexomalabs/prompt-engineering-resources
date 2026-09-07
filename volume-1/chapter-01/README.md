# Volume I · Chapter 1 — Lab: Mapping the AI Landscape

**Type: Paper exercise.** No code, no setup.

## What You Will Build

A written audit of five AI systems you already use, classifying each by capability and
naming a limitation you have personally observed. The deliverable is a table and a short
analysis — the first piece of engineering judgement this series asks of you.

## Learning Outcomes

- Distinguish narrow AI from general AI using concrete examples rather than definitions.
- Identify which AI capability a product is applying.
- Separate a system's marketed capability from its observed behaviour.
- Articulate a limitation in terms a system designer could act on.

## Prerequisites

- Chapter 1 read in full
- **Estimated API cost: $0.00** — no code, no model calls
- A text editor

## Setup

None. Open a document.

## Running It

Complete this table for five AI-powered products or services you use regularly:

| Product | AI capability | Evidence you observed | Limitation you observed |
|---------|---------------|-----------------------|-------------------------|
|         |               |                       |                         |

Then pick one product and find the company's public engineering documentation or
technical blog. Compare what they claim about capabilities and limitations with what
you wrote.

## Expected Result

A completed table plus roughly 300 words answering:

1. Were any of your five products using general AI? If not, why not?
2. Which limitations did you observe that the company does not publicly acknowledge?
3. If you were designing one of these systems, which limitations would have to be
   handled in the architecture rather than fixed in the model?

## Verifying Your Work

There is no automated check. Your table is adequate if a reader who has never used the
product could tell, from your entry alone, what the system does and where it fails.

Question 3 is the one that matters. It is the question the rest of Volume I answers.

## If It Does Not Work

| Symptom | Cause | Fix |
|---------|-------|-----|
| Cannot identify the AI capability | Looking at the product, not the task | Ask what input becomes what output |
| Every limitation is "it makes mistakes" | Too general to act on | Name the input class that triggers the failure |
| Cannot find engineering documentation | Not all companies publish | Substitute a published model card or research paper |

## Going Further

Repeat the audit for a system in a regulated domain — banking, healthcare, hiring.
Note which limitations become compliance problems rather than quality problems.
Volume IV Chapter 78 returns to this.
