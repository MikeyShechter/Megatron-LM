---
name: pace
description: >-
  Avoids long dumps by breaking the explanation of the current topic into
  smaller steps so the user can steer, ask, or check along the way. Use when
  the user invokes /pace.
disable-model-invocation: true
---

# Pace

Use this skill only when the user explicitly invokes `/pace`.

The purpose of this skill is to avoid long dumps.

When the user is trying to understand something, one long explanation is a problem. A question, something they do not understand, or something they disagree with often shows up near the start. They then want to discuss that in the next message. A dump makes that next turn assume the rest was already read, so the early point never gets a real discussion.

Breaking the explanation into smaller steps leaves more chances to steer, ask, and check along the way. How to break it up depends on the context.

Apply this only to the topic in progress, including follow-ups on it. When that topic is finished, or the user moves on, return to normal replies.
