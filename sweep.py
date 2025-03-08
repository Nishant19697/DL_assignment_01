from model import trainer
import wandb

def main():
    wandb.init()
    wandb.run.name = f"epoch{wandb.config.epochs}_batch{wandb.config.batch_size}_optim{wandb.config.optimizer}_lr{wandb.config.learning_rate}_init{wandb.config.weight_init}_layers{wandb.config.num_layers}_hidden{wandb.config.hidden_size}_activation{wandb.config.activation}"
    trainer(args=wandb.config, logging=True)

if __name__ == "__main__":
    sweep_configuration = {
        "method": "bayes",
        "name": "sweep_trial01",
        "metric": {"goal": "maximize", "name": "val_accuracy"},
        "parameters": {
            "dataset": {"values":["fashion_mnist"]},
            "epochs": {"values": [5, 10]},
            "beta": {"values": [0.5]},
            "beta1": {"values": [0.9]},
            "beta2": {"values": [0.999]},
            "momentum": {"values": [0.5]},
            "epsilon": {"values": [1e-8]},
            "num_layers": {"values": [3, 4, 5]},
            "hidden_size": {"values": [64, 128, 256]},
            "weight_decay": {"values": [0, 0.0005, 0.5]},
            "learning_rate": {"values": [1e-2, 1e-3, 1e-4]},
            "optimizer": {"values": ["sgd", "momentum", "rmsprop", "adam", "nadam"]},
            "batch_size": {"values": [64]},
            "weight_init": {"values": ["random", "Xavier", "kaiming"]},
            "activation": {"values": ["sigmoid", "tanh", "ReLU"]},
        },
    }

    sweep_id = wandb.sweep(sweep=sweep_configuration, project="sweep_trial01", entity="nishant19697-indian-institute-of-technology-madras")
    wandb.agent(sweep_id, function=main, count=50)