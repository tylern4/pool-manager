# Examples

This directory contains example configuration files and test data for
pool-manager.

## Directory contents

| File | Description |
|------|-------------|
| `slurm-perlmutter.yaml` | SLURM config for NERSC Perlmutter |
| `pbs-alcf.yaml` | PBS config for ALCF (Crux / Polaris) |
| `local-test.yaml` | Local subprocess backend for testing |
| `sample_jobs.json` | Sample `condor_q -json` output for `test-strategy` |
| `worker.sh` | Minimal worker script example |

## How to run

### 1. Local test (no HPC cluster required)

```bash
# From the repo root
cp examples/local-test.yaml pool-manager.yaml
cp examples/worker.sh /tmp/worker.sh
chmod +x /tmp/worker.sh
```

Edit `pool-manager.yaml` so `worker_script` points to `/tmp/worker.sh`, then:

```bash
pool-manager run --log-level DEBUG
```

The local backend starts your script as a subprocess. Press Ctrl+C to stop.

### 2. test-strategy (dry-run placement against sample data)

```bash
pool-manager test-strategy examples/sample_jobs.json -c examples/local-test.yaml
```

With running counts:

```bash
pool-manager test-strategy examples/sample_jobs.json \
  -c examples/local-test.yaml \
  --running 3 \
  --running-type small=2 --running-type large=1
```

### 3. SLURM (NERSC Perlmutter)

Copy `slurm-perlmutter.yaml`, edit the paths, credentials, and account, then:

```bash
pool-manager run -c examples/slurm-perlmutter.yaml
```

### 4. PBS (ALCF Crux / Polaris)

Copy `pbs-alcf.yaml`, edit the paths and project, then:

```bash
pool-manager run -c examples/pbs-alcf.yaml
```

## Worker script

The `worker.sh` example is a placeholder that sleeps for 60 seconds. In
production, this script should connect back to the HTCondor startd or run a
job payload appropriate for your site.
