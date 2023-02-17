from toy_problem_library import Curvature, Method, main, DynamicModel
from vmp.losses import custom_loss_func, standard_loss_func

if __name__ == "__main__":
    # freeze_support()

    robustness = False
    prefix = "runs/final-toy/performance-comparison/"

    jobs = [
        # dict(method=Metho d.LSTM, curvature=Curvature.CURVATURE, hidden_dim=64, epochs=400, prefix=prefix),
        # dict(method=Method.LSTM, curvature=Curvature.CURVATURE, hidden_dim=32, epochs=500, prefix=prefix),
        dict(
            method=Method.LSTM,
            curvature=Curvature.CURVATURE,
            hidden_dim=16,
            epochs=350,
            prefix=prefix,
            save_all=True,
        ),
        # dict(method=Method.LSTM, curvature=Curvature.CURVATURE, hidden_dim=8, epochs=1500, prefix=prefix),
        # dict(method=Method.LSTM, curvature=Curvature.CURVATURE, hidden_dim=4, epochs=400, prefix=prefix),
    ]

    lstm_jobs = [dict(d, method=Method.LSTM, loss_func="custom") for d in jobs] + [
        dict(d, method=Method.LSTM, loss_func="disp") for d in jobs
    ]
    ctrv_jobs = [
        dict(
            method=Method.CTRV, curvature=Curvature.CURVATURE, prefix=prefix, epochs=1
        ),
        dict(
            method=Method.CTRA, curvature=Curvature.CURVATURE, prefix=prefix, epochs=1
        ),
    ]
    jobs = [dict(d, method=Method.PIMP) for d in jobs]
    jobs = [dict(d, control_outputs=60) for d in jobs]  # + \
    # [dict(d, control_outputs=30) for d in jobs] #+ \
    # [dict(d, control_outputs=20) for d in jobs] #+ \
    # [dict(d, control_outputs=30) for d in jobs] + \
    # [dict(d, control_outputs=60) for d in jobs]
    jobs = [dict(d, curriculum=False) for d in jobs] + \
       [dict(d, curriculum=True, eps_per_input=2) for d in jobs] + \
       [dict(d, curriculum=True, eps_per_input=1) for d in jobs] #+ \
    # [dict(d, curriculum=True, eps_per_input=3) for d in jobs] + \
    jobs = [
        dict(d, residual=False) for d in jobs
    ]  # + [dict(d,residual=True) for d in jobs]
    jobs = [dict(d, model=DynamicModel.BICYCLE) for d in jobs]
    jobs = [
        dict(d, loss_func="custom") for d in jobs
    ]

    jobs = ctrv_jobs + lstm_jobs + jobs

    gpus = [f"cuda:{i}" for i in range(3)]
    procs_per_gpu = 4
    runner_specs = ["cuda:0"] * 2 + ["cuda:1"] * 4 + ["cuda:2"] * 4

    main(prefix, jobs, gpus, procs_per_gpu, robustness, runner_specs=runner_specs)
