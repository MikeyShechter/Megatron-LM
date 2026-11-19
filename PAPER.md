# Paper Guidelines

Rules for working on the paper with the user. Read this file first whenever the
user asks about the paper.

The paper's location and file layout are in the Paper section of `AGENTS.md`
(`CLAUDE.md` is a symlink to it). `PAPER_OUTLINE.md` is the map of the paper.

When the user states a new preference about the paper, add it to this file.
Keep each entry short.

## Roles

- The user writes the paper. Claude does the tasks the user gives, and nothing
  else. Do not assume that any part of the paper is Claude's to write.
- Claude is not the only editor. The user and the co-authors change the files
  too. Read a file again before working on it.
- The user decides, for example what goes in the main text and what goes in
  the appendix. Sometimes they ask for Claude's opinion.

## Target and scope

- Target venue: ICML. The first version goes to arXiv.
- For now there are no page limits. Everything is written where it belongs.
- Results come only from the Megatron runs. `experiments.tex`, the tables and
  the sweep appendix are old (modded-nanogpt, May 2026).
- Tables: the user adds them, or asks Claude to build them from the logs.

## Stages

The user works on the paper in stages.

1. Structure and content.
   - Get a good overall structure.
   - Get all the relevant information into the paper: what the method is, which
     experiments were run, the setting, and so on.
   - Do not get stuck on phrasing. Good phrasing is welcome. It is just not the
     goal of this stage.
2. Refinement. Refine the structure further for readability. Improve the
   phrasing.

The stages are a guideline, not a rule. Sometimes better phrasing makes the
intended structure clearer.

Current stage: 1 (since 2026-11-18).

## Writing

- The goal: the reader understands easily, and the story flows. This matters
  more than any strict rule about sentences. The user does not want strict
  rules here.
- Avoid overly complex sentences. The user marked this as very important and
  said Claude tends to write them. A long sentence is fine when it flows well,
  but that is not the usual case.
- No em dashes. No semicolons.

## Edits

1. Never change anything the user did not explicitly ask for.
2. When the user asks for suggestions, do not edit the text at all. Answer only
   in the chat.
3. When the user asks for an edit, use the iron skill first. For now this
   applies to every edit request. The user may relax it if it becomes too much.
4. Every change goes through the user before it is made, even a typo fix. The
   user wants to know exactly what changed. Git shows it too, but it is easier
   for them when the change passes through them explicitly.
5. Small fixes (typos, grammar) do not need a question per word. Say what is
   off and ask to fix it. Example: "The grammar in that paragraph is a bit off.
   Is it ok to fix it now?" The user then knows to look at that place. The same
   holds for a small problem Claude notices on its own: report it and ask.
   Never fix it silently.

## Notes in the text

- `\mikey{...}` and `\daniel{...}` are the authors' notes. Generally do not
  edit them. If a note looks resolved, ask whether to remove it.
- `\claude{...}` is for Claude's notes. The user allowed it for now. Do not
  overdo it. For reference, the user writes a `\mikey{}` note when they are not
  sure how to continue, or when they need to refer to something that is not
  written yet.
- The `\claude` macro is defined in `macros.tex`, right after `\mikey` and
  `\daniel` (added 2026-11-18, dark orange).

## Skills

`skills/iron/` and `skills/pace/` were added on 2026-11-18. They are unchanged
copies from https://github.com/tomshlomo/skills.

- iron: iron out the details before implementing. Ask questions one at a time,
  in plain chat text, with numbered options. Do not edit until the user says to
  proceed.
- pace: break a long explanation into small steps. The user invokes it with
  `/pace`.

Both files set `disable-model-invocation: true`. Only the user can load them,
with `/iron` and `/pace`. The Skill tool refuses to load iron for Claude. Its
error also says not to replicate the workflow by other means (seen on
2026-11-18). So rule 3 of Edits cannot be followed as written yet. The user was
asked on 2026-11-18 whether to remove the flag from the local copy of iron.
Until they answer, follow rules 4 and 5 of Edits: state every change and get
the user's approval before making it.
