import wandb
run = wandb.init(project='maicon_all')
model = run.use_artifact('run_ir:latest')
directory = model.download()
model = run.use_artifact('run_thermal:latest')
directory = model.download()