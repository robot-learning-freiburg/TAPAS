from tapas_gmm.env.environment import BaseEnvironment


def disturbe_at_step_no(
    env: BaseEnvironment, current_step: int, disturbe_at_step: int | None
):
    if current_step == disturbe_at_step:
        env.reset_joint_pose()
