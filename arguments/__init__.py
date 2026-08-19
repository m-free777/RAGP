from argparse import ArgumentParser, Namespace
import sys
import os


class GroupParams:
    pass


class ParamGroup:
    def __init__(self, parser: ArgumentParser, name: str, fill_none=False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group


class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.sky_seg = False
        self.load_normal = False
        self.load_depth = False
        self.use_dynamic_mask = False
        self.dynamic_mask_dir = ""
        # Rebuttal ablations: soft = original prior, hard = thresholded prior,
        # zero = all-zero prior (extreme prior failure).
        self.dynamic_prior_mode = "soft"
        self.dynamic_prior_threshold = 0.5
        self.eval = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        if hasattr(g, "dynamic_mask_dir") and g.dynamic_mask_dir:
            g.dynamic_mask_dir = os.path.abspath(g.dynamic_mask_dir)
        return g


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.normal_loss = False
        self.sparse_loss = False
        self.flatten_loss = False
        self.depth_loss = False
        self.depth2normal_loss = False
        self.lambda_l1_normal = 0.01
        self.lambda_cos_normal = 0.01
        self.lambda_flatten = 100.0
        self.lambda_dssim = 0.2
        self.lambda_sparse = 0.001
        self.lambda_depth = 0.1
        self.lambda_depth2normal = 0.05
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.min_opacity_prune = 0.005
        self.random_background = False

        self.dataset = 'waymo'
        self.propagation_interval = 20
        self.depth_error_min_threshold = 0.05
        self.depth_error_max_threshold = 0.15
        self.propagated_iteration_begin = 1000
        self.propagated_iteration_after = 12000
        self.propagation_refresh_iters = ""
        self.patch_size = 7
        self.pair_path = ''

        # Adaptive patch wrapper options (no gaussianpro kernel changes required)
        self.enable_adaptive_patch = False
        self.adaptive_patch_candidates = "3,7,11"
        self.adaptive_patch_hidden = 32
        self.adaptive_patch_lr = 0.001
        self.lambda_adaptive_patch = 0.05
        self.adaptive_patch_margin = 0.05
        self.adaptive_patch_label_dynamic_max = 0.5
        self.adaptive_patch_densify_dynamic_max = 0.6
        self.adaptive_patch_selector_start_iter = 1500
        self.adaptive_patch_conf_threshold = 0.35
        self.adaptive_patch_selector_ramp_iters = 1200
        self.adaptive_patch_selector_margin = 0.03
        self.adaptive_patch_teacher_conf_floor = 0.12
        self.adaptive_patch_teacher_temperature = 0.35
        self.adaptive_patch_teacher_label_conf_min = 0.12
        self.adaptive_patch_distill_weight = 0.02
        self.adaptive_patch_min_gate_threshold = 0.08
        self.adaptive_patch_max_patch_extra_conf = 0.04
        self.adaptive_patch_max_patch_min_margin = 0.03
        self.adaptive_patch_large_patch_teacher_penalty = 0.08
        self.adaptive_patch_large_patch_collapse_weight = 0.10
        self.adaptive_patch_large_patch_overuse_margin = 0.03
        self.adaptive_patch_extreme_patch_extra_conf = 0.02
        self.adaptive_patch_extreme_patch_min_margin = 0.01
        self.adaptive_patch_extreme_patch_collapse_weight = 0.04
        self.adaptive_patch_extreme_patch_overuse_margin = 0.03
        self.adaptive_patch_extreme_patch_center_margin = 0.03
        self.adaptive_patch_teacher_geo_weight = 1.0
        self.adaptive_patch_teacher_error_weight = 1.30
        self.adaptive_patch_teacher_scale_weight = 0.22
        self.adaptive_patch_teacher_dynamic_weight = 0.45
        self.adaptive_patch_teacher_dynamic_boundary_weight = 0.0
        self.adaptive_patch_teacher_normal_weight = 0.18
        self.adaptive_patch_gain_threshold = 0.02
        self.adaptive_patch_gain_low_texture_relax = 0.01
        self.adaptive_patch_gain_edge_penalty = 0.02
        self.adaptive_patch_gain_dynamic_penalty = 0.03
        # Prevent selector takeover from ignoring negative region gain on weak-teacher pixels.
        self.adaptive_patch_weak_teacher_gain_floor = 0.0
        # Minimum number of geometrically consistent source views for depth-propagation densification.
        # 1 is useful for sparse outdoor street views; 2 is cleaner but can make densify empty.
        self.adaptive_patch_densify_min_geo_support = 2
        self.densify_dynamic_dilation = 1
        self.adaptive_patch_extreme_patch_normal_threshold = 0.68
        self.adaptive_patch_extreme_patch_normal_center_margin = 0.03
        self.adaptive_patch_center_patch_teacher_bonus = 0.02
        self.adaptive_patch_center_patch_support_weight = 0.04
        self.adaptive_patch_center_patch_underuse_margin = 0.02
        self.adaptive_patch_small_patch_teacher_penalty = 0.03
        self.adaptive_patch_log_interval = 200
        self.adaptive_patch_use_kernel = True
        self.adaptive_patch_kernel_static_threshold = 0.25
        self.adaptive_patch_kernel_border = 8

        # Existing defaults moved from getattr so they can be tuned from CLI
        self.dynamic_prob_hard_threshold = 0.85
        self.src_static_prob_threshold = 0.25
        self.crossview_min_sources = 1
        self.source_depth_max = 299.0
        self.dynamic_prior_start_iter = 1000

        # Reliability-guided static optimization. These parameters turn an external
        # dynamic prior into a soft static reliability field used for late gradient
        # gating, reliable static supervision, and dynamic-boundary-aware densify.
        self.enable_reliability_field = False
        # continuous = current soft reliability; hard_mask = binary foreground removal.
        self.reliability_mode = "continuous"
        self.dynamic_mask_dilation = 1
        self.reliability_dynamic_weight = 1.0
        self.reliability_photo_weight = 0.0
        self.reliability_photo_tau = 0.10
        self.reliability_floor = 0.05
        self.dynamic_grad_start_iter = 8000
        self.dynamic_grad_floor = 0.25
        self.lambda_reliable_static = 0.0
        self.reliable_static_start_iter = 8000
        self.reliability_densify_threshold = 0.0

        # Weak late opacity suppression for dynamic/reflection/high-residual pixels.
        self.lambda_unreliable_opacity = 0.0
        self.unreliable_opacity_start_iter = 9000
        self.unreliable_opacity_ramp_iters = 4000
        self.unreliable_dynamic_threshold = 0.50
        self.unreliable_reliability_threshold = 0.35
        self.unreliable_photo_threshold = 0.18
        self.unreliable_opacity_dilation = 5
        self.unreliable_opacity_min_weight = 0.05

        # Metric-aligned static supervision. When enabled, the main photometric
        # loss is computed on the same pure static render used by final render.py
        # and training_report, instead of on a transient/composite image.
        self.metric_static_loss = False
        self.metric_static_loss_start_iter = 3000
        self.metric_static_full_weight = 0.30
        self.metric_static_full_weight_final = 0.50
        self.metric_static_full_weight_ramp_start = 12000
        self.metric_static_full_weight_ramp_iters = 4000
        self.metric_static_ssim_weight_threshold = 0.15

        # Optional late static takeover. Defaults are inert, so early 2k/7k
        # metrics stay close to the original command unless explicitly enabled.
        self.enable_static_takeover = False
        self.lambda_static_takeover = 0.0
        self.static_takeover_start_iter = 10000
        self.static_takeover_ramp_iters = 2000
        self.static_takeover_min_pixels = 100

        self.transient_downscale = 16
        self.transient_init_alpha = 0.02
        self.transient_warmup_iter = 3000
        self.transient_gate_floor = 0.05
        self.lambda_transient_sparse = 0.01
        self.lambda_transient_static = 0.05
        self.lambda_transient_tv = 0.001
        self.lambda_transient_rgb_tv = 0.001
        self.transient_lr = 0.01

        super().__init__(parser, "Optimization Parameters")


def get_combined_args(parser: ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    except FileNotFoundError:
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k, v in vars(args_cmdline).items():
        if v is not None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
