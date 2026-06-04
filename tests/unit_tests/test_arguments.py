import sys

import pytest
import yaml

from megatron.training.arguments import parse_args


def _parse_args(monkeypatch, cli_args):
    monkeypatch.setattr(sys, 'argv', ['pretrain_gpt.py', *cli_args])
    return parse_args()


def test_output_dir_sets_save_and_load(monkeypatch):
    args = _parse_args(monkeypatch, ['--output-dir', '/tmp/megatron-run'])

    assert args.output_dir == '/tmp/megatron-run'
    assert args.save == '/tmp/megatron-run'
    assert args.load == '/tmp/megatron-run'


@pytest.mark.parametrize('checkpoint_arg', ['--save', '--load'])
def test_output_dir_rejects_checkpoint_args(monkeypatch, checkpoint_arg):
    with pytest.raises(SystemExit) as exc_info:
        _parse_args(monkeypatch, ['--output-dir', '/tmp/megatron-run', checkpoint_arg, '/tmp/ckpt'])

    assert exc_info.value.code == 2


def test_run_name_sets_wandb_exp_name(monkeypatch):
    args = _parse_args(monkeypatch, ['--run-name', 'smoke-run'])

    assert args.run_name == 'smoke-run'
    assert args.wandb_exp_name == 'smoke-run'


def test_run_name_rejects_wandb_exp_name(monkeypatch):
    with pytest.raises(SystemExit) as exc_info:
        _parse_args(monkeypatch, ['--run-name', 'smoke-run', '--wandb-exp-name', 'explicit-run'])

    assert exc_info.value.code == 2


def test_flat_config_injects_output_dir_and_run_name(monkeypatch, tmp_path):
    output_dir = tmp_path / 'megatron-run'
    config_path = tmp_path / 'spec.yaml'
    config_path.write_text(
        yaml.safe_dump(
            {
                'output_dir': str(output_dir),
                'run_name': 'config-run',
            }
        )
    )

    args = _parse_args(monkeypatch, ['--config', str(config_path)])

    assert args.output_dir == str(output_dir)
    assert args.save == str(output_dir)
    assert args.load == str(output_dir)
    assert args.run_name == 'config-run'
    assert args.wandb_exp_name == 'config-run'


def test_checkpoint_latest_and_pre_decay_flags(monkeypatch):
    args = _parse_args(monkeypatch, ['--save-only-latest', '--save-pre-decay'])

    assert args.save_only_latest
    assert args.save_pre_decay


def test_flat_config_injects_checkpoint_latest_flags(monkeypatch, tmp_path):
    config_path = tmp_path / 'spec.yaml'
    config_path.write_text(
        yaml.safe_dump(
            {
                'save_only_latest': True,
                'save_pre_decay': True,
            }
        )
    )

    args = _parse_args(monkeypatch, ['--config', str(config_path)])

    assert args.save_only_latest
    assert args.save_pre_decay
