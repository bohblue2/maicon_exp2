import wandb

wandb.init(project='maicon_all')

name = 'run_ir'
epoch = 9
root_dir = '../exp4'
model_artifact = wandb.Artifact(name,type='model',metadata={'epoch':epoch})
model_artifact.add_file(f'{root_dir}/weights/best.pt')
model_artifact.add_file(f'{root_dir}/weights/best.pt')
model_artifact.add_file(f'{root_dir}/weights/best.pt')
wandb.log_artifact(model_artifact)

name = 'run_thermal'
epoch = 9
root_dir = '../exp5'
model_artifact = wandb.Artifact(name,type='model',metadata={'epoch':epoch})
model_artifact.add_file(f'{root_dir}/weights/best.pt')
wandb.log_artifact(model_artifact)