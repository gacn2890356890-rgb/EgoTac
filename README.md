# EgoTouch Research Survey

## Project Scope
This repository tracks an ongoing literature and experimental survey for EgoTouch, an early-stage research project on egocentric video, tactile sensing, hand-object interaction, and data collection for embodied AI / robot learning.

The current goal is to understand how prior work designs and deploys real-world experiments, including camera setup, tactile sensing hardware, RGB/RGB-D capture, simulation, synchronization, and data collection protocols.

---

## Latest Survey · September 19, 2026

[ **实验部署图文调研：从装置到采集协议** ](survey/experimental_setup.md)

基于《EgoTac (1).pptx》的24项研究工作，补充EgoTactile裸手/手套弱配对案例；包含16张原文截图、相机与触觉比较、同步/标定细节、真实实验和仿真的区分，以及3种候选试采方案。

[论文与证据索引](survey/paper_inventory.md) · [论文索引CSV](survey/paper_inventory.csv) · [本周进展](progress/2026-09-week3.md)

## Current Research Topics

### 1. Experimental Deployment in Prior Work
- Hand configuration: left / right / both
- Gloves or finger-worn devices
- Manual manipulation vs trajectory imitation / replay
- Task and object setup
- Trial design and repetition count
- Ground-truth acquisition
- Calibration and synchronization

[Survey notes](survey/experimental_setup.md)

### 2. Egocentric Camera Setup
- Camera model
- Head-mounted / glasses-mounted / other mounting strategies
- RGB vs RGB-D
- Resolution / FPS / FOV
- Hand visibility and occlusion
- Motion blur
- Sensor synchronization

[Survey notes](survey/camera_setup.md)

### 3. Intel RealSense
- Existing RealSense-based egocentric or manipulation systems
- Depth quality for close-range hand-object interaction
- Mounting feasibility
- Calibration
- RGB-depth synchronization

[Survey notes](survey/realsense.md)

### 4. Finger-Sleeve / Wearable Tactile Sensing
- Finger-worn tactile sensors
- Pressure / force / contact / slip sensing
- Sensor placement
- Sampling rate
- Calibration
- Video synchronization
- Impact on natural manipulation

[Survey notes](survey/tactile_sensing.md)

### 5. Simulation-to-Real / Simulation-to-Data
- Simulation platforms
- Synthetic hand-object interaction
- Contact / force / tactile simulation
- Domain randomization
- Sim-to-real transfer
- Potential use for reducing real-world data collection

[Survey notes](survey/simulation.md)

---

# Progress Log

## Week of Sep. 14–20, 2026

### Completed
- Reviewed the first batch of literature on egocentric tactile sensing and hand-object interaction.
- Reviewed a second batch focusing on experimental methods and related sensing pipelines.
- Identified the need to study experimental deployment details rather than only high-level methods.
- Started organizing prior work by camera, wearable sensing, hand configuration, and experimental protocol.
- Added the first source-grounded experimental deployment survey with 16 figure excerpts, paper-specific protocols, camera/tactile tables, and candidate pilot setups.

### New Project Updates
- Finger-sleeve sensing hardware has been obtained.
- Intel RealSense is available for potential RGB-D experiments.
- Future work may involve offline / in-person experimental setup and data collection.

### Current Tasks
- Survey experimental deployment details in prior papers.
- Investigate finger-sleeve sensing devices and calibration.
- Investigate camera models and mounting strategies used for egocentric video capture.
- Review RealSense-based hand-object interaction setups.
- Revisit simulation-related papers and extract implementation details.
- Compare RGB-only, RGB-D, and tactile-video synchronized setups.

### Expected Deliverables
- Experimental setup comparison table
- Camera and mounting survey
- Finger-sleeve sensing survey
- RealSense feasibility summary
- Simulation-to-real comparison
- Initial candidate experimental setup

[Weekly record](progress/2026-09-week3.md)

---

# Next Milestone
The first survey now outlines three candidate setups. Next: confirm the available finger-sleeve and RealSense models, discuss the candidates, and run a small synchronization/calibration pilot. Candidate setups:

1. Lightweight RGB-only setup
2. RGB-D / RealSense setup
3. Synchronized egocentric video + finger-sleeve tactile sensing setup

