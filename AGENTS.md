# Repository Guidelines

This is a research fork of Megatron-LM, not an upstream contribution workflow.
Treat the repository as an active experiment workspace.

## General Instructions

- When I ask you to do something, do that and nothing else. If you believe you
  should do something else as well, ask first.
- Do not add checks, preflights, guards, validation steps, or similar launch
  logic unless I specifically ask for it or specifically approve it after you
  ask.
- Sometimes I will ask you to run things yourself, or to iterate, make changes,
  and test until it works. That is a very strong tool when requested. If I do not ask for
  that, do not use that style because it uses too many tokens: make the requested change, and if you think you need to test things to make it work, tell me. The thing is, simply running a small test run is often much cheaper and faster for me than waiting for you to cover all the test cases. 
- Take your time and think about the correct solution before implementing. The instructions above do not mean you should rush. 
- If something is unclear, you have two options: either ask about it before implementing or if you believe you know what to do, simply do it and mention the decision you made at the end.

## Runtime Environment

- Runs target an offline GPU node.
- The container used by `run_megatron.sh` is:
  `/e/project1/laionize/shechter1/containers/megatron-lm-dev.sif`.
- The main entry point for training runs is:
  `run_megatron.sh`.

## Submitting Runs

- I use the conda environment `megatron-submit` to submit runs to SLURM.
- Submissions are done with the `submit_multiple.py` script. You don't have to read it unless we change something in the submission.
- You are not expected to submit jobs with this workflow unless I explicitly ask,
  but this context should guide changes related to configs or launch behavior.
- An example of a valid config file for `submit_multiple.py` is:
  `configs/test_JSC.yaml`.

## Research Context

The current research focus is load balancing loss.

Relevant arguments:

- `moe_router_load_balancing_type`
- `load_balance_ste_width`
- `moe_aux_loss_coeff`
