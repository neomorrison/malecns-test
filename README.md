# maleCNS fly brain → human body

A spiking simulation of the **maleCNS connectome** (the complete central nervous
system of an adult male *Drosophila*, Janelia FlyEM / Cambridge, v1.0) drives an
**anatomically proportioned humanoid** in MuJoCo. The body has *learned* to
walk, run, turn and jump; the fly brain decides *what* to do: chase fruit it
sees, jump over logs rolling at it, and stop to eat when it tastes sugar.

The locomotion controller is trained sim-to-real style, so the same controller
could in principle run a physical body with these kinematics (see
[Hardware](#running-it-on-a-physical-body)).

```
 world ──senses──▶ identified fly neurons ──▶ 141,615-neuron spiking brain ──▶ descending neurons
                   LC10 (visual target)                                         DNa01/DNa02/DNg13 → turn
                   LPLC2/LC4 (looming)                                          Giant Fiber (DNp01) → jump
                   LB3 sugar receptors                                          P9/oDN1/BDN2 → walk / run
                                                                                MN9 → eat
                         ──▶ "nerve cord": motor-program arbitration ──▶ learned locomotion policy ──▶ 23 joint servos
```

## What is (and is not) the fly brain

A fly connectome cannot move human legs: its motor neurons are wired to six fly
legs, wings and a proboscis. So the split is:

| Layer | What it is | Source |
|---|---|---|
| Decisions: where to turn, when to jump, when to eat | Leaky integrate-and-fire simulation of every traced maleCNS brain neuron (141,615 cells, 86 M synapses) | the connectome; nothing about these behaviours is programmed |
| Sensory interface | Senses become spike rates on identified neurons (LC10 target trackers, LPLC2/LC4 loom detectors, LB3 sugar receptors) | literature + verified in the model (`results/brain_probe.md`) |
| Readout ("nerve cord") | Fixed decoder from named descending neurons to speed / turn / jump / feed | known function of each cell type |
| Hunger | Tonic drive of forward-walking command neurons + gain of the pursuit channel | **not** in the connectome (neuromodulatory); added explicitly |
| Walking, running, jumping, balance | Neural network policy trained with PPO + a rhythm generator | reinforcement learning, not the fly |

What the connectome model does on its own (from `scripts/probe_brain.py`):

* **Looming → escape jump.** Driving LPLC2/LC4 on one side fires the Giant Fiber at ~400 Hz
  and the *opposite* side's steering neurons (turn away).
* **Visual target → pursuit.** LC10 neurons (the male's courtship-chase pathway) excite the
  steering neurons of the *same* side: the body turns toward what it sees.
* **Sugar → feeding.** LB3 labellar taste neurons (identified by their direct synapses onto the
  sugar neurons Bract and Usnea, Shiu et al. 2022) drive the proboscis motor neuron MN9.
* **Odour did not work.** Food-odour receptor input produces no reliable steering or walking
  command in this model, so the agent finds food by sight, not smell.

### Changes to the raw wiring (and why)

Point-neuron models collapse every synapse onto the cell body. That is wrong for a few
well-known contact types and, with maleCNS's dense graph, produced seizure-like runaway
activity. Documented in `malecns_body/connectome.py`:

* per-synapse weight rescaled from Shiu et al.'s FlyWire value (0.275 mV) by 391/692 to match
  maleCNS's higher synapse count per neuron;
* Kenyon cell → Kenyon cell synapses removed (0.7%; mostly axo-axonic in the mushroom body);
* excitatory antennal-lobe projection/local-neuron loops removed (1.1%; largely dendro-dendritic
  within glomeruli);
* the ventral nerve cord is not simulated: it runs fly legs the humanoid does not have, and its
  ascending feedback made the brain drift into self-sustained states.

## The body

`malecns_body/assets/humanoid.xml`: 1.74 m, 72 kg male built from de Leva (1996) segment masses
and lengths; 23 actuated joints with human ranges of motion (3-DoF spine, hips, shoulders; knees,
elbows; 2-DoF ankles); PD servos with torque limits in the range of human joint strength; pelvis
IMU, foot force sensors, joint torque sensors; legs collide with each other.

## Teaching it to move

`scripts/train_locomotion.py`: PPO (JAX) on 256 parallel bodies with domain randomisation. One
policy takes commands (speed, turn rate, jump) and outputs corrections on top of a clock-driven
rhythm generator. A curriculum raises the top speed from walking toward running and switches on
random shoves and jumps once the gait is stable. Watch it learn with
`scripts/training_timelapse.py` and `scripts/plot_training.py`.

## Running it on a physical body

The locomotion controller is built to transfer:

* **The policy only sees on-board sensors**: IMU gravity direction + gyro, joint encoders, its
  previous action and a 5-frame history. Ground truth (velocity, height, contact forces, physics
  parameters) is only given to the critic during training and is not needed at run time.
* **Domain randomisation**: per-body mass (±15%), centre of mass, floor friction (0.4–1.3), servo
  gains (±20–30%), motor strength (−20%/+10%), joint damping and armature, 0–16 ms action latency,
  sensor noise, random shoves.
* **Hardware limits in the loop**: torque limits, joint-speed limits, penalties on power, torque,
  jerky actions and hard foot impacts.
* **Framework-free runtime**: `malecns_body/deploy.py` runs the exported `policy.npz` with numpy
  only and defines a `RobotInterface` (`read_imu`, `read_joints`, `command_joints`). Actuator
  gains, torque and speed limits are in `malecns_body/body.py` (`ACTUATOR_SPEC`).
* **Sim-to-sim check**: `scripts/eval_locomotion.py --sim2sim` runs the controller in a differently
  configured simulator (1 ms step, RK4, other friction / mass / gains).

## Quick start

```bash
pip install -r requirements.txt
python scripts/probe_brain.py                       # downloads maleCNS (~570 MB) and probes circuits
python scripts/train_locomotion.py --minutes 240    # teach the body (CPU, ~10k steps/s on 4 cores)
MUJOCO_GL=osmesa python scripts/run_demo.py --policy models/locomotion_policy.npz --video results/demo.mp4
```

## Status

Work in progress on branch `claude/nice-ride-egny8l`: locomotion training is ongoing (walking and
turning work; running and jumping are being learned), and a learned get-up (fall recovery)
controller is planned.

## References

maleCNS v1.0 (Janelia FlyEM & Cambridge, 2025) · Shiu et al. 2024 *Nature* 634:210 (whole-brain LIF
model) · Shiu et al. 2022 *eLife* (sugar feeding circuit) · von Reyn et al. 2014, 2017; Ache et al.
2019 (Giant Fiber, LPLC2) · Rayshubskiy et al. 2020; Yang et al. 2023 (steering DNs) · Bidaye et al.
2014, 2020; Sapkal et al. 2024 (walking DNs) · Ribeiro et al. 2018; Hindmarsh Sten et al. 2021 (LC10
pursuit) · Iscen et al. 2018 (policies modulating trajectory generators) · de Leva 1996 (anthropometry).
