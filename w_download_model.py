import wandb
project = 'maicon_all'
entity = 'ryanbae'
run = wandb.init(project=project)
model = run.use_artifact(f'{entity}/{project}/run_ir:latest', type='model')
directory = model.download()
model = run.use_artifact(f'{entity}/{project}/run_thermal:latest', type='model')
directory = model.download()
