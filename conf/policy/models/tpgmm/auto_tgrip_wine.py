from tapas_gmm.policy.models.tpgmm import (
    AutoTPGMMConfig,
    CascadeConfig,
    DemoSegmentationConfig,
    FittingStage,
    FrameSelectionConfig,
    InitStrategy,
    ModelType,
    ReconstructionStrategy,
    TPGMMConfig,
)

tpgmm_config = TPGMMConfig(
    n_components=13,
    model_type=ModelType.HMM,
    use_riemann=True,
    add_time_component=True,
    add_action_component=False,
    # action_as_orientation=True,
    # action_with_magnitude=True,
    position_only=False,
    add_gripper_action=True,
    reg_shrink=1e-2,
    reg_diag=5e-4,
    reg_diag_gripper=1e-1,
    reg_em_finish_shrink=1e-2,
    reg_em_finish_diag=5e-4,
    reg_em_finish_diag_gripper=1e-1,
    fix_first_component=False,
    fix_last_component=False,
    heal_time_variance=False,
)

frame_selection_config = FrameSelectionConfig(
    init_strategy=InitStrategy.TIME_BASED,
    fitting_actions=(FittingStage.INIT,),
    rel_score_threshold=0.49,
    use_bic=False,
)

demos_segmentation_config = DemoSegmentationConfig(
    gripper_based=False,
    distance_based=False,
    velocity_based=True,
    repeat_final_step=0,
    repeat_first_step=0,
    components_prop_to_len=True,
)

cascade_config = CascadeConfig(
    kl_keep_time_dim=True,
    kl_keep_rotation_dim=False,
)

auto_tpgmm_config = AutoTPGMMConfig(
    tpgmm=tpgmm_config,
    frame_selection=frame_selection_config,
    demos_segmentation=demos_segmentation_config,
    cascade=cascade_config,
)
