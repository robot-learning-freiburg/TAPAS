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
    add_time_component=True,
    add_action_component=True,
    position_only=False,
    reg_shrink=1e-3,
    reg_diag=1e-5,
    fix_first_component=False,
    fix_last_component=False,
)
