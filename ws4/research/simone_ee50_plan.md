# Proposed plan from Simone (teammate), received 2026-10-17 11:28 (transcribed verbatim from chat)

XLeRobot Bimanual Adaptation, Short Plan

1. Dataset preparation
   - Collect public XLeRobot, dual-SO-101 and bimanual UMI datasets.
   - Convert robot demonstrations to EE50 using forward kinematics.
   - Convert UMI demonstrations to BFL's EE50 conventions.
2. Validate EE50 to XLeRobot
   - Set up XLeRobot in simulation.
   - Implement constrained IK for both 5-DoF arms.
   - Evaluate reachability, pose errors and collisions.
3. Embodiment fine-tuning
   - Initialize from FLUX 3 Action's pretrained EE50 checkpoint.
   - Train using mixed UMI + XLeRobot data.
   - Fine-tune the backbone with LoRA, adapting the EE50 head as needed.
4. Task-specific fine-tuning
   - Collect 50-100 demonstrations of bimanual obstacle removal and target retrieval.
   - Apply a second LoRA fine-tuning stage.
5. Evaluation and deployment
   - Test in simulation, then on the physical XLeRobot.
   - Measure task success, IK failures and inference latency.
   - Compare against native 12D joint-space adaptation if feasible.

Critical decision: Validate IK before investing in EE50 fine-tuning. If 6D predictions are too difficult
to execute reliably on XLeRobot's 5-DoF arms, switch to the native 12D action head.
