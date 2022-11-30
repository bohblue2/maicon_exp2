import wandb
project = 'maicon_all'
entity = 'ryanbae'
run = wandb.init(project=project)
model = run.use_artifact(f'{entity}/{project}/run_ir:v0', type='model')
directory = model.download(root='yolov5-pip/weights/ir')
model = run.use_artifact(f'{entity}/{project}/run_thermal:v0', type='model')
directory = model.download(root='yolov5-pip/weights/thermal')
