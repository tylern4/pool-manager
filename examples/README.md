# Examples

This directory contains example configuration files and test data for
pool-manager.

## Directory contents

| File | Description |
|------|-------------|
| `slurm-perlmutter.yaml` | SLURM config for NERSC Perlmutter |
| `pbs-alcf.yaml` | PBS config for ALCF (Crux / Polaris) |
| `sample_jobs.json` | Sample `condor_q -json` output for `test-strategy` |
| `worker.sh` | Minimal worker script example |

## How to run

### 1. test-strategy (dry-run placement against sample data)

```bash
pool-manager test-strategy examples/sample_jobs.json -c examples/slurm-perlmutter.yaml
```

With running counts:

```bash
pool-manager test-strategy examples/sample_jobs.json \
  -c examples/slurm-perlmutter.yaml \
  --running 3 \
  --running-type small=2 --running-type large=1
```

### 2. SLURM (NERSC Perlmutter)

Copy `slurm-perlmutter.yaml`, edit the paths, credentials, and account, then:

```bash
pool-manager run -c examples/slurm-perlmutter.yaml
```

### 3. PBS (ALCF Crux / Polaris)

Copy `pbs-alcf.yaml`, edit the paths and project, then:

```bash
pool-manager run -c examples/pbs-alcf.yaml
```

## Worker script

The `worker.sh` example is a placeholder that sleeps for 60 seconds. In
production, this script should connect back to the HTCondor startd or run a
job payload appropriate for your site.
