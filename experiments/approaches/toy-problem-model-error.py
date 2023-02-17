from toy_problem_library import Curvature, Method, main
import numpy as np
import random

if __name__ == "__main__":
    #freeze_support()

    robustness=False
    prefix = "runs/final-toy/model-error/"
    
    jobs = [
        #dict(method=Method.LSTM, curvature=Curvature.CURVATURE, hidden_dim=64, epochs=240, prefix=prefix),
        #dict(method=Method.LSTM, curvature=Curvature.CURVATURE, hidden_dim=32, epochs=200, prefix=prefix),
        dict(method=Method.LSTM, curvature=Curvature.CURVATURE, hidden_dim=16, epochs=200, prefix=prefix),
        #dict(method=Method.LSTM, curvature=Curvature.CURVATURE, hidden_dim=8, epochs=1500, prefix=prefix),
        #dict(method=Method.LSTM, curvature=Curvature.CURVATURE, hidden_dim=4, epochs=1500, prefix=prefix),
    ]
    
    jobs = [dict(d, method=Method.PIMP, curriculum=True, eps_per_input=1, control_outputs=60) for d in jobs]
    jobs = sum([[dict(d, wheelbase=wb) for d in jobs] for wb in np.arange(0.0802, 1.5002, 0.01)], [])
    jobs = [dict(d,residual=False) for d in jobs]
    
    jobs = jobs
    
    gpus = [f'cuda:{i}' for i in range(3)]
    procs_per_gpu = 4
    runner_specs = ['cuda:0']*4 + ['cuda:1']*10 + ['cuda:2']*10
    random.shuffle(runner_specs)
    #print(jobs)
    main(prefix, jobs, gpus, procs_per_gpu, robustness, runner_specs=runner_specs)

