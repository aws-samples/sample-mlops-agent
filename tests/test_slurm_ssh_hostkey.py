"""Security regression — Slurm skill SSH host-key verification (CSO 2026-07-25, MEDIUM).

Covers the fix in lambda/skills/slurm/handler.py: _ssh_run must NOT trust unknown
head-node host keys. It now requires SLURM_KNOWN_HOSTS to be set, loads that
known_hosts file, and installs paramiko.RejectPolicy (never AutoAddPolicy).
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types

import pytest


def _import_slurm():
    """Load the slurm handler under a unique module name (collision-proof)."""
    path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "lambda", "skills", "slurm", "handler.py")
    )
    spec = importlib.util.spec_from_file_location("_slurm_handler_hostkey", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_slurm_handler_hostkey"] = mod
    spec.loader.exec_module(mod)
    return mod


class _RejectPolicy:
    """Stand-in for paramiko.RejectPolicy so we can assert it (not AutoAdd) is used."""


class _AutoAddPolicy:
    """Stand-in for paramiko.AutoAddPolicy — its use in _ssh_run would be a regression."""


def _install_fake_paramiko(monkeypatch, client):
    """Register a fake `paramiko` module exposing the two policies and an SSHClient."""
    fake = types.ModuleType("paramiko")
    fake.RejectPolicy = _RejectPolicy
    fake.AutoAddPolicy = _AutoAddPolicy
    fake.SSHClient = lambda: client
    monkeypatch.setitem(sys.modules, "paramiko", fake)


class _FakeSSHClient:
    """Records host-key configuration so the test can assert on it."""

    def __init__(self):
        self.loaded_host_keys = None
        self.policy = None
        self.connected = False

    def load_host_keys(self, path):
        self.loaded_host_keys = path

    def set_missing_host_key_policy(self, policy):
        self.policy = policy

    def connect(self, **kwargs):
        self.connected = True

    def exec_command(self, cmd):
        class _Stream:
            def read(self_inner):
                return b""

            channel = types.SimpleNamespace(recv_exit_status=lambda: 0)

        return None, _Stream(), _Stream()

    def close(self):
        pass


def test_ssh_run_refuses_without_known_hosts(monkeypatch):
    """With SLURM_KNOWN_HOSTS unset, _ssh_run must fail loudly before connecting."""
    h = _import_slurm()
    h.MOCK_MODE = False
    h.SLURM_HOST = "head.example.internal"
    h.SLURM_KNOWN_HOSTS = ""
    with pytest.raises(RuntimeError, match="SLURM_KNOWN_HOSTS"):
        h._ssh_run("squeue")


def test_ssh_run_pins_host_key_and_rejects_unknown(monkeypatch, tmp_path):
    """With SLURM_KNOWN_HOSTS set, _ssh_run loads it and installs RejectPolicy."""
    h = _import_slurm()
    client = _FakeSSHClient()
    _install_fake_paramiko(monkeypatch, client)

    known_hosts = tmp_path / "known_hosts"
    known_hosts.write_text("head.example.internal ssh-ed25519 AAAA...\n")

    h.MOCK_MODE = False
    h.SLURM_HOST = "head.example.internal"
    h.SLURM_KNOWN_HOSTS = str(known_hosts)
    # _load_ssh_key hits Secrets Manager — stub it out.
    h._load_ssh_key = lambda: str(tmp_path / "id")

    h._ssh_run("squeue")

    assert client.loaded_host_keys == str(known_hosts)
    assert isinstance(client.policy, _RejectPolicy)
    assert not isinstance(client.policy, _AutoAddPolicy)
    assert client.connected is True
