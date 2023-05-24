from torch.cuda import is_available, get_device_name
from torch.nn.utils import clip_grad_value_
from torch import device, save, no_grad
from torch.utils.data import DataLoader
from torch.nn import Module, MSELoss
from torch import tensor, float32
from torch.optim import Adam
from pandas import DataFrame
from os.path import join
from sys import stdout
from os import getcwd
from tqdm import tqdm

class Trainer:
    def __init__(self, model:Module, dl_train:DataLoader, dl_val:DataLoader, dl_test:DataLoader):
        self.device = device("cuda" if is_available() else "cpu")
        print(f"The device that will be used in training is {get_device_name(self.device)}")

        self.model = model.to(self.device)
        self.model_name = model.name

        self.optimizer = Adam(self.model.parameters(), lr=0.001)
        self.criterion = MSELoss()

        assert self.criterion is not None, "Please define a loss function"
        assert self.optimizer is not None, "Please define an optimizer"
        assert self.model_name is not None, "Please define a model name"

        self.train = dl_train
        self.val = dl_val
        self.test = dl_test

    def train_epoch(self, dl:DataLoader):
        # Put the model in training mode
        self.model.train().float()

    # Store each step's accuracy and loss for this epoch
        epoch_metrics = {
            "loss": [],
        }

        # Create a progress bar using TQDM
        stdout.flush()
        with tqdm(total=len(dl), desc=f'Training') as pbar:
            # Iterate over the training dataset
            for inputs, truths in dl:
                # Zero the gradients from the previous step
                self.optimizer.zero_grad()

                # Move the inputs and truths to the target device
                inputs = tensor(inputs, device=self.device, dtype=float32)
                inputs.required_grad = True  # Fix for older PyTorch versions
                truths = tensor(truths, device=self.device, dtype=float32)

                # Run model on the inputs
                output = self.model(inputs)

                # Perform backpropagation
                loss = self.criterion(output, truths)
                loss.backward()
                clip_grad_value_(self.model.parameters(), 0.1)
                self.optimizer.step()

                # Store the metrics of this step
                step_metrics = {
                    'loss': loss.item(),
                }

                # Update the progress bar
                pbar.set_postfix(**step_metrics)
                pbar.update(1)

                # Add to epoch's metrics
                for k,v in step_metrics.items():
                    epoch_metrics[k].append(v)

        stdout.flush()

        # Return metrics
        return epoch_metrics
    
    def val_epoch(self, dl:DataLoader):
        # Put the model in evaluation mode
        self.model.eval().float()

        # Store the total loss and accuracy over the epoch
        amount = 0
        total_loss = 0

        # Create a progress bar using TQDM
        stdout.flush()
        with no_grad(), tqdm(dl, desc=f'Validation') as pbar:
            # Iterate over the validation dataloader
            for inputs, truths in dl:
                 # Move the inputs and truths to the target device
                inputs = tensor(inputs, device=self.device, dtype=float32 )
                inputs.required_grad = True  # Fix for older PyTorch versions
                truths = tensor(truths, device=self.device, dtype=float32)

                # Run model on the inputs
                output = self.model(inputs)
                loss = self.criterion(output, truths)

                # Store the metrics of this step
                step_metrics = {
                    'loss': loss.item(),
                }

                # Update the progress bar
                pbar.set_postfix(**step_metrics)
                pbar.update(1)

                amount += 1
                total_loss += step_metrics["loss"]

        stdout.flush()

        # Print mean of metrics
        total_loss /= amount
        print(f'Validation loss is {total_loss/amount}')

        # Return mean loss and accuracy
        return {
            "loss": [total_loss],
        }
    
    
    def save_model(self, DIR:str=getcwd()):
        """Save the model"""
        store_path = join(DIR, self.model_name)
        
        save(self.model.state_dict(), store_path)

    def fit(self, epochs: int, batch_size:int):
        # Initialize Dataloaders for the `train` and `val` splits of the dataset. 
        # A Dataloader loads a batch of samples from the each dataset split and concatenates these samples into a batch.
        dl_train = self.train
        dl_val = self.val

        # Store metrics of the training process (plot this to gain insight)
        df_train = DataFrame()
        df_val = DataFrame()

        # Train the model for the provided amount of epochs
        for epoch in range(1, epochs+1):
            print(f'Epoch {epoch}')
            metrics_train = self.train_epoch(dl_train)
            df_train = df_train.append(DataFrame({'epoch': [epoch for _ in range(len(metrics_train["loss"]))], **metrics_train}), ignore_index=True)

            metrics_val = self.val_epoch(dl_val)
            df_val = df_val.append(DataFrame({'epoch': [epoch], **metrics_val}), ignore_index=True)

        df_train.to_csv(f'savefolderpytorch\\train_{self.model_name}.csv')
        df_val.to_csv(f'savefolderpytorch\\val_{self.model_name}.csv')
        # Return a dataframe that logs the training process. This can be exported to a CSV or plotted directly.
    
    def load_model(self, model_name:str, DIR:str=getcwd()):
        """Load the model"""
        store_path = join(DIR, model_name)
        
        self.model.load_state_dict(store_path)
        