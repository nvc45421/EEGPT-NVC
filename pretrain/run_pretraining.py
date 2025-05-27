# Training in 256Hz data and 4s
import time
import torch
from pytorch_lightning import loggers as pl_loggers

from engine_pretraining import *
from configs import *
torch.set_float32_matmul_precision("medium")

seed_torch(7)


# init model


model = LitEEGPT(get_config(**(MODELS_CONFIGS[tag])), 
                 USE_LOSS_A =(variant != "A"),
                 USE_LN     =(variant != "B"),
                 USE_SKIP   =(variant != "C"))
lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
callbacks = [lr_monitor]

# Init loggers
tb_logger = pl_loggers.TensorBoardLogger('./logs/', name=f"EEGPT_{tag}_{variant}_tb")
csv_logger = pl_loggers.CSVLogger('./logs/', name=f"EEGPT_{tag}_{variant}_csv")


trainer = pl.Trainer(
    strategy='auto',
    devices=devices,
    max_epochs=max_epochs,
    callbacks=callbacks,
    logger=[tb_logger, csv_logger]
)


# ⏱ Time the training
start_time = time.time()
trainer.fit(model, train_loader, valid_loader)
end_time = time.time()

# Compute time in hours
elapsed_hours = (end_time - start_time) / 3600

# ✅ Log to CSVLogger
csv_logger.log_metrics({'total_pretraining_time_hours': elapsed_hours})
print(f"Total pretraining time: {elapsed_hours:.2f} hours")