#!/usr/bin/env python3
"""Run a native validation command and sample its Linux RSS and DRM allocations.

Run inside the same distrobox as the executable. This is sampled telemetry, not
an allocator high-water mark: combine it with nr_allocation_stats in the check.
"""
import argparse
import json
import pathlib
import subprocess
import time


def fields(path):
    try:
        return dict(line.split(':', 1) for line in path.read_text().splitlines() if ':' in line)
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return {}


def kib(value):
    parts = value.split()
    return int(parts[0]) * 1024 if len(parts) == 2 and parts[1] in ('kB', 'KiB') else 0


def sample(pid):
    proc = pathlib.Path('/proc') / str(pid)
    status = fields(proc / 'status')
    clients = {}
    try:
        entries = list((proc / 'fdinfo').iterdir())
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        entries = []
    for entry in entries:
        info = fields(entry)
        client = info.get('drm-client-id')
        if client is not None:
            key = (info.get('drm-pdev', '').strip(), client.strip())
            clients[key] = sum(kib(info.get(k, '')) for k in ('drm-memory-vram', 'drm-memory-gtt'))
    available = kib(fields(pathlib.Path('/proc/meminfo')).get('MemAvailable', ''))
    return dict(rss_bytes=kib(status.get('VmRSS', '')),
                rss_high_water_bytes=kib(status.get('VmHWM', '')),
                drm_allocated_bytes=sum(clients.values()),
                drm_clients=len(clients), host_available_bytes=available)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('a native executable command is required')
    start = time.monotonic()
    peaks = {k: 0 for k in ('rss_bytes', 'rss_high_water_bytes', 'drm_allocated_bytes', 'drm_clients')}
    minimum_available = None
    count = 0
    child = subprocess.Popen(command)
    try:
        while True:
            row = sample(child.pid)
            count += 1
            for key in peaks:
                peaks[key] = max(peaks[key], row[key])
            available = row['host_available_bytes']
            if available:
                minimum_available = available if minimum_available is None else min(available, minimum_available)
            rc = child.poll()
            if rc is not None:
                break
            time.sleep(0.05)
    except BaseException:
        child.terminate()
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
        raise
    result = dict(command=command, returncode=rc, seconds=time.monotonic()-start,
                  samples=count, sample_interval_seconds=0.05, peaks=peaks,
                  minimum_host_available_bytes=minimum_available,
                  scope='direct child only; sampled DRM allocations include runtime buffers; RSS and DRM may overlap on shared-memory GPUs')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    raise SystemExit(rc if rc >= 0 else 128 - rc)


if __name__ == '__main__':
    main()
