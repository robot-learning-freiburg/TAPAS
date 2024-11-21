from tapas_gmm.policy.models.tpgmm import (
    TPGMM,
    AutoTPGMM,
    AutoTPGMMConfig,
    FittingStage,
    FrameSelectionConfig,
    InitStrategy,
    ModelType,
    ReconstructionStrategy,
    TPGMMConfig,
    _xdx_to_tangent,
)

tpgmm_config = TPGMMConfig(
    n_components=16,
    model_type=ModelType.HMM,
    use_riemann=True,
    add_time_component=False,
    add_action_component=True,
    position_only=False,
    reg_shrink=1e-3,
    reg_diag=1e-5,
    fix_first_component=True,
    fix_last_component=True,
    fixed_first_component_n_steps=2,
    heal_time_variance=True,
)

frame_selection_config = FrameSelectionConfig(
    init_strategy=InitStrategy.TIME_BASED,
    fitting_actions=[FittingStage.INIT],
    rel_score_threshold=0.75,
)

auto_tpgmm_config = AutoTPGMMConfig(
    tpgmm=tpgmm_config,
    frame_selection=frame_selection_config,
)
