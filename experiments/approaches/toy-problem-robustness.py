from toy_problem_library import Curvature, Method, main
import random

if __name__ == "__main__":

    robustness=True
    prefix = "runs/final-toy/robustness-longer/"
    
    jobs = [
        #dict(method=Method.LSTM, curvature=Curvature.CURVATURE, hidden_dim=64, epochs=240, prefix=prefix),
        #dict(method=Method.LSTM, curvature=Curvature.CURVATURE, hidden_dim=32, epochs=1500, prefix=prefix),
        dict(method=Method.LSTM, curvature=Curvature.CURVATURE, hidden_dim=16, epochs=350, prefix=prefix, robustness=True),
        #dict(method=Method.LSTM, curvature=Curvature.CURVATURE, hidden_dim=8, epochs=1500, prefix=prefix),
        #dict(method=Method.LSTM, curvature=Curvature.CURVATURE, hidden_dim=4, epochs=240, prefix=prefix),
    ]
    
    lstm_jobs = [dict(d, method=Method.LSTM) for d in jobs]
    jobs = [dict(d, method=Method.PIMP) for d in jobs]
    jobs = [dict(d, control_outputs=60) for d in jobs] #+ \
           #[dict(d, control_outputs=10) for d in jobs] + \
           #[dict(d, control_outputs=20) for d in jobs] + \
           #[dict(d, control_outputs=30) for d in jobs] + \
           #[dict(d, control_outputs=60) for d in jobs]
    jobs = [dict(d, curriculum=False) for d in jobs] + \
           [dict(d, curriculum=True, eps_per_input=2) for d in jobs] + \
           [dict(d, curriculum=True, eps_per_input=1) for d in jobs] #+ \
           #[dict(d, curriculum=True, eps_per_input=1) for d in jobs] + \
    jobs = [dict(d,residual=False) for d in jobs] #+ [dict(d,residual=True) for d in jobs]
    
    jobs = lstm_jobs + jobs
    
    gpus = [f'cuda:{i}' for i in range(3)]
    procs_per_gpu = 0
    runner_specs = ['cuda:0']*0 + ['cuda:1']*6 + ['cuda:2']*6
    random.shuffle(runner_specs)
    print(runner_specs)
    
    main(prefix, jobs, gpus, procs_per_gpu, robustness, runner_specs=runner_specs)