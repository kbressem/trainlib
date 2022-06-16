import argparse

import monai
from kidney.report import ReportGenerator
from kidney.train import SegmentationTrainer
from kidney.utils import load_config

parser = argparse.ArgumentParser(description="Train a segmentation model.")
parser.add_argument("--config", type=str, required=True, help="path to the config file")
args = parser.parse_args()
config_fn = args.config


config = load_config(config_fn)
monai.utils.set_determinism(seed=config.seed)

# if os.path.exists(config.data.cache_dir):
#    print("Clearing cache from previous session")
#    shutil.rmtree(config.data.cache_dir) # rm previous cache DS

print(
    f"""
    Running supervised segmentation training
    Run ID:     {config.run_id}
    Debug:      {config.debug}
    Out dir:    {config.out_dir}
    model dir:  {config.model_dir}
    log dir:    {config.log_dir}
    images:     {config.data.image_cols}
    labels:     {config.data.label_cols}
    data_dir    {config.data.data_dir}
    """
)

# create supervised trainer for segmentation task
trainer = SegmentationTrainer(
    progress_bar=True,
    early_stopping=True,
    metrics=["MeanDice", "HausdorffDistance", "SurfaceDistance"],
    save_latest_metrics=True,
    config=config,
)

# add lr scheduler to trainer
trainer.fit_one_cycle()

# let's train
trainer.run()

# finish script with final evaluation of the best model
trainer.evaluate()

# generate a markdown document with segmentation results
report_generator = ReportGenerator(config.run_id, config.out_dir, config.log_dir)

report_generator.generate_report()
