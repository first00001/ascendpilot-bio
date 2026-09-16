#!/usr/bin/env python3
"""Execute commands or upload files through the HiDevLab SSH jump host."""

from __future__ import annotations

import argparse
import os
import socket
import sys
from pathlib import Path

import paramiko


def connect(jump_user: str, password: str, target: str, port: int):
    sock = socket.create_connection(("113.47.8.48", 2234), timeout=20)
    jump = paramiko.Transport(sock)
    jump.start_client(timeout=20)
    jump.auth_none(jump_user)
    channel = jump.open_channel("direct-tcpip", (target, port), ("127.0.0.1", 0))
    target_transport = paramiko.Transport(channel)
    target_transport.start_client(timeout=20)
    target_transport.auth_password("root", password)
    return jump, target_transport


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="199.93.56.238")
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--command")
    parser.add_argument("--command-file", type=Path)
    parser.add_argument("--upload", nargs=2, metavar=("LOCAL", "REMOTE"))
    parser.add_argument("--download", nargs=2, metavar=("REMOTE", "LOCAL"))
    args = parser.parse_args()
    jump_user = os.environ.get("TARGET_JUMP_USER")
    password = os.environ.get("TARGET_PASSWORD")
    if not jump_user or not password:
        parser.error("TARGET_JUMP_USER and TARGET_PASSWORD must be set")

    jump = target_transport = None
    try:
        jump, target_transport = connect(jump_user, password, args.target, args.port)
        if args.upload:
            local, remote = Path(args.upload[0]), args.upload[1]
            with paramiko.SFTPClient.from_transport(target_transport) as sftp:
                sftp.put(str(local), remote, confirm=True)
            print(f"uploaded {local} -> {remote}")
        exit_status = 0
        command = args.command
        if args.command_file:
            if command:
                parser.error("use either --command or --command-file")
            command = args.command_file.read_text(encoding="utf-8")
        if command:
            channel = target_transport.open_session()
            channel.exec_command(command)
            while not channel.exit_status_ready():
                if channel.recv_ready():
                    sys.stdout.buffer.write(channel.recv(65536))
                    sys.stdout.buffer.flush()
                if channel.recv_stderr_ready():
                    sys.stderr.buffer.write(channel.recv_stderr(65536))
                    sys.stderr.buffer.flush()
            while channel.recv_ready():
                sys.stdout.buffer.write(channel.recv(65536))
            while channel.recv_stderr_ready():
                sys.stderr.buffer.write(channel.recv_stderr(65536))
            exit_status = channel.recv_exit_status()
        if args.download:
            remote, local = args.download[0], Path(args.download[1])
            local.parent.mkdir(parents=True, exist_ok=True)
            with paramiko.SFTPClient.from_transport(target_transport) as sftp:
                sftp.get(remote, str(local))
            print(f"downloaded {remote} -> {local}")
        raise SystemExit(exit_status)
    finally:
        if target_transport:
            target_transport.close()
        if jump:
            jump.close()


if __name__ == "__main__":
    main()
