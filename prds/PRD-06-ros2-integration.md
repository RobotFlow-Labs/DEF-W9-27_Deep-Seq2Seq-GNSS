# PRD-06: ROS2 Integration for ANIMA Runtime

> Module: deep-seq2seq-gnss | Priority: P1
> Depends on: PRD-05
> Status: ⬜ Not started

## Objective
Wrap the detector in a ROS2 node with stable topics/messages for GNSS anti-spoofing inference.

## Context (from paper)
GNSS spoofing detection is safety-critical and should integrate into robotics/autonomy runtime stacks.

## Acceptance Criteria
- [ ] ROS2 node subscribes GNSS sequence windows and publishes spoof probabilities.
- [ ] Configurable window length and threshold.
- [ ] Launch file available.
- [ ] Minimal integration test with mocked messages.

## Files to Create
| File | Purpose | Paper Ref | Est. Lines |
|------|---------|-----------|-----------|
| `ros2/anima_gnss_spoof_node.py` | ROS2 node | I | ~220 |
| `ros2/launch/anima_gnss_spoof.launch.py` | launch file | — | ~70 |
| `anima_module.yaml` | module manifest | — | ~80 |
