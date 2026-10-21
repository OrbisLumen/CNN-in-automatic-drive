# DonkeyCar Setup

MyDeZero and vehicle control share a root `.venv`. Each member creates their own environment.
Use the same Python and dependency versions on Windows and macOS, with the simulator for each platform.

## Files and Versions

| Item | Location |
| --- | --- |
| Third-party interface source | `tools/gym-donkeycar/` |
| Simulator | `tools/donkey-sim/` |
| Project code | The semester's `proj/` |
| Requirements and templates | The semester's `request/` |
| Software engineering reports and figures | The semester's `work/software-engineering/` (planned) |

Do not commit `tools/` or `.venv`. Store data, weights, and training outputs in ignored
`data/` directories or shared storage. Record download links, checksums, and related
experiments in the repository. Avoid personal absolute simulator paths.

Shared baseline:

- [Interface source](https://github.com/tawnkramer/gym-donkeycar): `a1f4ca6961e17a6929c1f2e883358a1edafb479d`
- Simulator: `v25.10.06`; download the [macOS package](https://github.com/tawnkramer/gym-donkeycar/releases/download/v25.10.06/DonkeySimMac.zip) or [Windows package](https://github.com/tawnkramer/gym-donkeycar/releases/download/v25.10.06/DonkeySimWin.zip).
- macOS archive SHA-256: `edc36077126257320d433a7d07266c1feae289245687c4b9139bac353e31dce0`

Record version changes through PRs. Do not independently track the latest `master`.

## Installation

Run from the repository root. Skip creation or cloning if the files already exist:

```bash
mkdir tools
git clone https://github.com/tawnkramer/gym-donkeycar.git tools/gym-donkeycar
git -C tools/gym-donkeycar checkout a1f4ca6961e17a6929c1f2e883358a1edafb479d
```

Extract the simulator into `tools/donkey-sim/`. On Windows, keep the executable,
data folders, and DLLs in their original relative locations.
Create `.venv` as described in the [root README](../README.md), then install both projects:

```bash
# macOS
.venv/bin/python -m pip install -e ./projl/proj -e ./tools/gym-donkeycar
.venv/bin/python -c "import mydezero, gymnasium, gym_donkeycar; print('import OK')"
```

```powershell
# Windows PowerShell; activation is optional
.\.venv\Scripts\python.exe -m pip install -e ./projl/proj -e ./tools/gym-donkeycar
.\.venv\Scripts\python.exe -c "import mydezero, gymnasium, gym_donkeycar; print('import OK')"
```

Record shared Python and dependency versions after compatibility testing.
If macOS cannot launch the simulator because execute permission is missing:

```bash
chmod u+x tools/donkey-sim/DonkeySimMac/donkey_sim.app/Contents/MacOS/donkey_sim
open tools/donkey-sim/DonkeySimMac/donkey_sim.app
```

## Acceptance and Experiment Records

Verify on at least one Mac and one Windows machine: launch the simulator, connect
from Python, receive images, control steering and throttle, and close cleanly.
The Gymnasium API returns `(obs, info)` from `reset()` and
`(obs, reward, terminated, truncated, info)` from `step()`.

For each experiment, record the code commit, dependency versions, simulator version,
track, random seed, evaluation metrics, and inference time.
