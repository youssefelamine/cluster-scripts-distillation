# Kadeploy Runtime Workflow

These scripts prepare and validate a reusable Ubuntu 24.04 Kadeploy image for
the repository's real Mininet/DDQN training workflow.

## Path Layout

```text
/root/project                         cloned project, retained in the image
/root/project/kadeploy_runtime        these scripts
/root/project/results                 training logs and train_<timestamp> runs
/root/project/tmp                     temporary PCAP/flow processing data
/opt/ddos-rl-venv                     Python virtual environment
/opt/ddos-rl-tools/MHDDoS             MHDDoS checkout
/opt/ddos-rl-tools/CICFlowMeter       built CICFlowMeter checkout
```

Setup also creates compatibility symlinks required by hardcoded project paths:

```text
/home/user12/myenv                    -> /opt/ddos-rl-venv
/home/user12/Documents/CICFlowMeter   -> /opt/ddos-rl-tools/CICFlowMeter
/root/MHDDoS                          -> /opt/ddos-rl-tools/MHDDoS
```

D-ITG and `tcpdump` are not installed because the active training path uses the
project's TCP client/server, MHDDoS, and TShark instead.

## Build And Capture

Run reservation and deployment commands from the Kadeploy/OAR host:

```bash
oarsub -t deploy -l /nodes=1,walltime=6:00:00 -I
NODE="$(cat /tmp/deploynodes.${OAR_JOBID})"
kaenv3 -p ubuntu2404-base
kadeploy3 -e ubuntu2404-base -f /tmp/deploynodes.${OAR_JOBID}
ssh-keygen -f ~/.ssh/known_hosts -R "$NODE"
ssh root@"$NODE"
```

On the deployed node, clone the repository at the required path:

```bash
git clone REPOSITORY_URL /root/project
cd /root/project
```

Run the image workflow as root:

```bash
/root/project/kadeploy_runtime/01_setup_image.sh
/root/project/kadeploy_runtime/02_check_image.sh
/root/project/kadeploy_runtime/03_smoke_test.sh
/root/project/kadeploy_runtime/04_clean_before_capture.sh
```

The smoke test runs the real entrypoint with one episode and one step:

```bash
/opt/ddos-rl-venv/bin/python3 reinforcement/Main.py \
  -a '[h1]' -e 1 -s 1 --checkpoint-every 1 --keep-last-checkpoints 1
```

It takes several minutes because the project performs real Mininet traffic,
packet capture, CICFlowMeter processing, and network-metric calculation.
`04_clean_before_capture.sh` removes only the tracked smoke run/log and
temporary caches. It preserves code, tools, the virtualenv, and any other
results.

Capture the cleaned node:

```bash
/root/create_custom_image.sh ddos-rl-ubuntu2404.tar.bz2
ls -lh /image/ddos-rl-ubuntu2404.tar.bz2
scp /image/ddos-rl-ubuntu2404.tar.bz2 YOUR_LOGIN@cluster.common.lip6.fr:~
```

The archive under `/image` is temporary. Copy it to persistent storage before
the OAR reservation ends. Register a matching private `.env` descriptor from
the Kadeploy host, then redeploy-test it before considering shared visibility.
See `/root/project/clusterdoc.md` for the descriptor and registration details.

## Later Training

Inside a node deployed from the captured image, start a default project run:

```bash
/root/project/kadeploy_runtime/start_training.sh
```

Pass normal entrypoint arguments through whitespace-separated `TRAIN_ARGS`:

```bash
TRAIN_ARGS="-a [h1] -e 50 -s 100 --checkpoint-every 5" \
  /root/project/kadeploy_runtime/start_training.sh
```

A non-interactive deploy job can deploy the captured environment and invoke the
same command over SSH. This assumes Kadeploy injects your SSH key for root
access; verify that `ssh -o BatchMode=yes root@NODE true` works before relying
on unattended jobs.

```bash
oarsub -t deploy -l /nodes=1,walltime=12:00:00 \
  'set -e
   NODE="$(cat /tmp/deploynodes.${OAR_JOBID})"
   kadeploy3 -e ddos-rl-ubuntu2404 -f /tmp/deploynodes.${OAR_JOBID}
   ssh-keygen -f "$HOME/.ssh/known_hosts" -R "$NODE" || true
   set +e
   ssh -o BatchMode=yes -o StrictHostKeyChecking=no root@"$NODE" \
     "TRAIN_ARGS='\''-a [h1] -e 50 -s 100 --checkpoint-every 5'\'' /root/project/kadeploy_runtime/start_training.sh"
   TRAIN_EXIT=$?
   scp -o BatchMode=yes -o StrictHostKeyChecking=no -r \
     root@"$NODE":/root/project/results "$HOME/ddos-rl-results-${OAR_JOBID}" || true
   exit "$TRAIN_EXIT"'
```

Adjust walltime and training arguments for the experiment. Results stored only
on the deployed node are temporary, so the example copies the complete results
directory to persistent home storage before the job ends.

## Logs And Results

The project creates each run under:

```text
/root/project/results/train_<timestamp>/
```

That directory contains figures, CSV data, CICFlowMeter outputs, model weights
and checkpoints, RL statistics, and configuration output. `start_training.sh`
moves its console log into the detected run directory as `training.log` and
writes `status.txt` plus `exit_code.txt` there. If training fails before the
project creates a run directory, those files remain in `/root/project/results`.

## Build Log: Successful Ubuntu 24.04 Image Capture

This section records the successful interactive build and capture session for the
custom DDOS-RL Ubuntu 24.04 image.

### Completed Work

- Deployed the clean Ubuntu 24.04 base environment and cloned the project to:

  ```text
  /root/project
  ```

- Ran the image setup script:

  ```bash
  ./kadeploy_runtime/01_setup_image.sh
  ```

  The setup installed and configured the Python environment, Mininet, Open
  vSwitch, Java, Gradle, CICFlowMeter, MHDDoS, TShark, and other required
  dependencies.

- Fixed Java 8 registration. Java 8 was installed, but it was not registered
  with `update-alternatives`, so `java` and `javac` were registered and selected
  manually. The validated Java version was:

  ```text
  Java 1.8.0_492
  ```

- Successfully ran the image validation script:

  ```bash
  ./kadeploy_runtime/02_check_image.sh
  ```

  It confirmed the required Python imports and validated Mininet, Open vSwitch,
  TShark, Java 8, `javac`, Maven, Gradle, CICFlowMeter, MHDDoS, `libpcap`,
  `jnetpcap`, and the required compatibility links.

- Ran the first smoke test. The test successfully created the Mininet network,
  generated TCP traffic, ran the ICMP attack, and captured packets. It later
  timed out with exit code `124` while invoking CICFlowMeter.

- Tested and fixed CICFlowMeter independently using:

  ```text
  /tmp/tshark_out.pcap
  ```

  Initially, Gradle printed:

  ```text
  Please select pcap!
  ```

  The project was passing the source and destination PCAP paths as Gradle
  properties, while CICFlowMeter expected command arguments. A Gradle
  compatibility configuration was added so those properties are passed correctly
  to the `exeCMD` task. The Gradle daemon was also disabled. After the fix, the
  exact project-style command succeeded and generated:

  ```text
  tshark_out.pcap_Flow.csv
  ```

- Diagnosed a Mininet cgroup failure on Ubuntu 24.04. Ubuntu 24.04 uses cgroup
  v2, while the virtualenv originally contained Mininet `2.3.0.dev6`, whose
  cgroup detection only supported cgroup v1. This caused:

  ```text
  cgroups not mounted on /sys/fs/cgroup
  ```

- Upgraded the Mininet package inside the virtualenv to the current upstream
  version. Final versions:

  ```text
  System Mininet:      2.3.0
  Virtualenv Mininet:  2.3.1b4
  ```

  The new virtualenv Mininet correctly supports cgroup v2.

- Fixed headless X terminal support. The project uses Mininet terminal commands
  for traffic generation and MHDDoS. On the headless node, `xhost` was missing
  or hanging, so the required X utilities were installed and a virtual display
  was started with:

  ```bash
  export DISPLAY=:99
  Xvfb :99 -ac -screen 0 1280x800x24
  ```

  The `-ac` option allowed Mininet terminal processes to start without X
  authorization problems.

- Cleaned stale processes between attempts, including suspended smoke-test jobs,
  Flask/EntryPoint processes, port `5000`, Mininet state, Open vSwitch state,
  and Gradle daemon processes.

- Successfully completed the smoke training. The final smoke test completed and
  printed:

  ```text
  (Reinforcement) ================> Main Ended
  ```

  The smoke test used one episode and one reinforcement-learning step. It
  successfully produced network traffic and attack data, CICFlowMeter results,
  network metrics, CSV files, training figures, model weights, and checkpoints.

- Copied the successful smoke-test result directory to persistent cluster
  storage using `scp`.

- Cleaned the node before capture with:

  ```bash
  ./kadeploy_runtime/04_clean_before_capture.sh
  ```

  This preserved the project code, Python virtualenv, installed tools, and
  required dependencies.

- Captured and copied the customized Ubuntu image. The image was captured as:

  ```text
  /image/ddos-rl-ubuntu2404.tar.bz2
  ```

  It was approximately `3.4 GB` and was copied to persistent storage at:

  ```text
  /home/elamine/ddos-rl-ubuntu2404.tar.bz2
  ```

  The image-creation process successfully verified the archive integrity.

### Notes For Future Rebuilds

- Do not treat an early smoke-test timeout as an installation failure if the log
  shows that Mininet, traffic generation, packet capture, and the attack all ran.
  Check CICFlowMeter and shutdown behavior first.
- Keep Java 8 explicitly registered with `update-alternatives`.
- Keep the virtualenv Mininet version at `2.3.1b4` or newer for Ubuntu 24.04
  cgroup v2 compatibility.
- Keep the Xvfb `:99` display configuration for headless Mininet terminal
  support.
- Disable the Gradle daemon for CICFlowMeter reliability in short smoke-test and
  non-interactive job contexts.
- Before capturing, always clean stale Mininet, Open vSwitch, Flask, Gradle,
  `tcpdump`/TShark, MHDDoS, and training processes.

## Deployment Image and Environment Inventory

This section documents the Kadeploy images and environment descriptors currently available on the cluster. It records which image should be used for each stage of the project, what changed between image versions, and how the current working v2 image was produced.

The goal is to avoid confusion between the base Ubuntu environment, the first custom DDOS-RL image, the fixed v2 image used for teacher training, and the planned v3 image for student and distillation experiments.

### Current Inventory

As of the current teacher-training run, the available deployment files are:

```text
/home/elamine/ubuntu2404-base.env
/home/elamine/ddos-rl-ubuntu2404.env
/home/elamine/ddos-rl-ubuntu2404.tar.bz2
/home/elamine/ddos-rl-ubuntu2404-v2.env
/home/elamine/ddos-rl-ubuntu2404-v2.tar.bz2
```

| Version | Image | Env file | Status | Purpose |
|---|---|---|---|---|
| Base | none listed | `/home/elamine/ubuntu2404-base.env` | Keep | Clean Ubuntu 24.04 base environment |
| v1 | `/home/elamine/ddos-rl-ubuntu2404.tar.bz2` | `/home/elamine/ddos-rl-ubuntu2404.env` | Backup only | First custom DDOS-RL image |
| v2 | `/home/elamine/ddos-rl-ubuntu2404-v2.tar.bz2` | `/home/elamine/ddos-rl-ubuntu2404-v2.env` | Current stable | Fixed image used for teacher training |
| v3 | `/home/elamine/ddos-rl-ubuntu2404-v3.tar.bz2` | `/home/elamine/ddos-rl-ubuntu2404-v3.env` | Not created yet | Planned student/distillation image |

---

## Base Ubuntu 24.04 Environment

### Env file

`/home/elamine/ubuntu2404-base.env`

### Purpose

This is the clean Ubuntu 24.04 base Kadeploy environment. It is useful only if the full DDOS-RL image must be rebuilt from scratch.

### Use case

Use this only as a clean base if rebuilding everything manually again.

---

## v1 — Original DDOS-RL Image

### Image

`/home/elamine/ddos-rl-ubuntu2404.tar.bz2`

### Env file

`/home/elamine/ddos-rl-ubuntu2404.env`

### Purpose

This was the first custom DDOS-RL image built for the project.

It contains the main project and runtime dependencies, including:

- `/root/project`
- Python virtual environment
- Mininet
- Open vSwitch
- CICFlowMeter
- MHDDoS
- training/runtime scripts

### Problem found

The image itself was mostly functional, but passive deployment had an SSH automation problem.

The frontend SSH public key was not properly baked into:

```text
/root/.ssh/authorized_keys
```

Because of this, after passive deployment, the job script could not automatically SSH into the deployed node as root without manual intervention.

This made the image inconvenient for automated training jobs.

### Status

This image is kept as a historical backup. Prefer v2 for real training jobs.

---

## How v2 Was Created

### Goal

The goal of v2 was to create a fixed version of v1 where the frontend SSH key is already present inside the deployed root account.

This allows passive jobs to:

1. reserve a node,
2. deploy the image,
3. SSH into the node as root,
4. start training,

without manually entering a password or injecting the SSH key again.

### Base used

v2 was created by first deploying the original v1 image:

```bash
kadeploy3 -d --env-file /home/elamine/ddos-rl-ubuntu2404.env -f "$OAR_NODEFILE"
```

### Node used during recapture

The deployed node used during the v2 fix was:

```bash
big1
```

During that session:

```bash
NODE=big1
```

### SSH key injection

The frontend public SSH key was manually injected into the deployed node:

```bash
cat /home/elamine/.ssh/id_ed25519.pub | ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null root@"$NODE" \
  'mkdir -p /root/.ssh && chmod 700 /root/.ssh && cat >> /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys && grep elamine@cluster /root/.ssh/authorized_keys'
```

This ensured that the deployed root account contained the frontend key in:

```text
/root/.ssh/authorized_keys
```

### Passwordless SSH verification

After injecting the key, passwordless SSH was verified with:

```bash
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes root@"$NODE" \
  'hostname; whoami; ls -ld /root/project; grep -c elamine@cluster /root/.ssh/authorized_keys'
```

Expected successful behavior:

- `hostname` prints the deployed node name
- `whoami` prints `root`
- `/root/project` exists
- `/root/.ssh/authorized_keys` contains the frontend key

Duplicate key entries were cleaned so that the key appeared exactly once:

```text
1
```

### Cleanup before capture

Before capturing the image, old runtime outputs were removed:

```bash
rm -rf /root/project/results/* /root/project/tmp/* /tmp/tshark_out.pcap /tmp/xvfb-ddos.log
```

This made sure the new image did not include old experiment outputs or temporary capture files.

### Capture command

The fixed deployed system was captured using:

```bash
/root/create_custom_image.sh ddos-rl-ubuntu2404-v2.tar.bz2
```

The capture produced:

```text
/image/ddos-rl-ubuntu2404-v2.tar.bz2
```

The image was then copied back to the frontend home directory as:

```text
/home/elamine/ddos-rl-ubuntu2404-v2.tar.bz2
```

### v2 env file creation

The v1 env descriptor was copied and updated to point to the v2 image.

The resulting env file is:

```text
/home/elamine/ddos-rl-ubuntu2404-v2.env
```

The important image field is:

```yaml
image:
    file: /home/elamine/ddos-rl-ubuntu2404-v2.tar.bz2
    kind: tar
    compression: bzip2
```

The env name may still be similar to v1, but the important part is that the `image.file` points to the v2 tarball.

---

## v2 — Current Working Teacher-Training Image

### Image

`/home/elamine/ddos-rl-ubuntu2404-v2.tar.bz2`

### Env file

`/home/elamine/ddos-rl-ubuntu2404-v2.env`

### Purpose

v2 is the current stable working image for automated teacher training.

It is based on v1, but fixes the passive SSH issue.

### Fixes compared to v1

v2 includes the frontend SSH public key inside the deployed root account:

```text
/root/.ssh/authorized_keys
```

This means passive deployment can automatically SSH into the node as root.

### Verified behavior

v2 was successfully deployed with:

```bash
kadeploy3 -d --env-file /home/elamine/ddos-rl-ubuntu2404-v2.env -f "$OAR_NODEFILE"
```

After deployment, passwordless SSH was verified with:

```bash
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes root@"$NODE" \
  'hostname; whoami; ls -ld /root/project; grep -c elamine@cluster /root/.ssh/authorized_keys; ls -l /root/project/kadeploy_runtime/start_training.sh'
```

The verification confirmed:

- `/root/project` exists
- `/root/.ssh/authorized_keys` contains the frontend key exactly once
- `/root/project/kadeploy_runtime/start_training.sh` exists
- `/root/project/kadeploy_runtime/start_training.sh` is executable

### Current usage

v2 is currently being used by the teacher training job:

```text
OAR job: 1293333
Node: tall1
Env: /home/elamine/ddos-rl-ubuntu2404-v2.env
Image: /home/elamine/ddos-rl-ubuntu2404-v2.tar.bz2
```

The training output confirmed that the job uses:

```text
/home/elamine/ddos-rl-ubuntu2404-v2.env
```

The run is writing results under a folder like:

```text
/root/project/results/train_2026_06_15_09_24_12/
```

Example result files observed:

```text
/root/project/results/train_2026_06_15_09_24_12/cic/Episode 1 - Step 1 - CIC results.csv
/root/project/results/train_2026_06_15_09_24_12/cic/Episode 1 - Step 26 - CIC results.csv
```

This confirms that the teacher training job is actively producing CIC result files.

### Status

v2 is the current trusted image for teacher baseline training.

Do not overwrite v2.

Keep v2 as the stable fallback image.

---

## Planned v3 — Student / Distillation Image

### Planned image

`/home/elamine/ddos-rl-ubuntu2404-v3.tar.bz2`

### Planned env file

`/home/elamine/ddos-rl-ubuntu2404-v3.env`

### Purpose

v3 will be based on v2, but will include additional code for student and model-compression experiments.

Planned additions:

- Student A training support
- Student B training support
- teacher model path loading
- distillation alpha parameter
- seed control
- variant launch scripts
- smoke tests for student runs

---

## Planned Student Experiments

### Teacher model

The teacher is the baseline DDQN model currently being trained with v2.

After the teacher job finishes, the saved teacher model path will be used by the student scripts.

### Student A

Student A is the compressed student model.

Purpose:

Test reduced capacity while preserving the main teacher architecture pattern.

Expected script:

```text
run_student_a.sh
```

Expected inputs:

```text
TEACHER_MODEL_PATH
DISTILL_ALPHA
SEED
```

### Student B

Student B is the structural variant.

Purpose:

Test the effect of removing one layer / changing architecture shape.

Expected script:

```text
run_student_b.sh
```

Expected inputs:

```text
TEACHER_MODEL_PATH
DISTILL_ALPHA
SEED
```

---

## Planned Workflow for v3

The intended workflow is:

1. Let current teacher job finish.
2. Confirm that the teacher model was saved and synced back to `/home/elamine`.
3. Keep v2 unchanged as backup.
4. Deploy v2 again in a maintenance job.
5. Modify `/root/project` inside that deployed node.
6. Add Student A and Student B code.
7. Add `run_student_a.sh` and `run_student_b.sh`.
8. Run tiny smoke tests, not full training.
9. Recapture the modified system as v3.
10. Create `/home/elamine/ddos-rl-ubuntu2404-v3.env`.
11. Submit student jobs using v3.

---

## Important Rules

### Do not overwrite v2

v2 is the current working teacher-training image.

It should remain unchanged.

Any future student/distillation modifications should become v3.

### Do not edit the running teacher node

The current teacher training job is running on:

```text
tall1
```

Do not modify `/root/project` on `tall1` while job `1293333` is training.

### v3 must be a new image

Use a new image name:

```text
ddos-rl-ubuntu2404-v3.tar.bz2
```

Use a new env file:

```text
ddos-rl-ubuntu2404-v3.env
```

---

## Useful Commands

### List available images and env files

```bash
ls -lh /home/elamine/*.tar.bz2 /home/elamine/*.env 2>/dev/null
```

### Deploy v2 manually inside a maintenance job

```bash
kadeploy3 -d --env-file /home/elamine/ddos-rl-ubuntu2404-v2.env -f "$OAR_NODEFILE"
```

### Verify passwordless SSH after deploying v2

```bash
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes root@"$NODE" \
  'hostname; whoami; ls -ld /root/project; grep -c elamine@cluster /root/.ssh/authorized_keys; ls -l /root/project/kadeploy_runtime/start_training.sh'
```

### Submit teacher training using current v2 default

```bash
cd /home/elamine
CLUSTER=any WALLTIME=100:00:00 JOBNAME=ddos-rl-teacher ./submit_deploy.sh
```

### Check current OAR jobs

```bash
oarstat -u
```

### Check teacher job output

```bash
tail -n 60 /home/elamine/OAR.ddos-rl-teacher.1293333.stdout
```

### Count current CIC result files on the training node

```bash
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null root@tall1 \
  'find /root/project/results/train_2026_06_15_09_24_12/cic -type f | wc -l'
```

---

## Final Notes

v1 proved that the image content and training environment mostly worked.

v2 fixed the missing SSH key issue and is now confirmed to support passive automated teacher training.

v3 should be created only after teacher training completes and after the student/distillation code is ready and smoke-tested.