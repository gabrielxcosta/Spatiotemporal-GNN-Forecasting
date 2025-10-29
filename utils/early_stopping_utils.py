class EarlyStopping:
    """
    Implements early stopping for model training with a minimum improvement tolerance 
    and patience counter.

    This class monitors the validation loss during training and stops the training 
    process if the loss does not improve after a specified number of epochs 
    (`patience`). A minimal improvement threshold (`min_delta`) is also used to 
    prevent stopping for insignificant changes.

    Attributes:
        patience (int): Number of epochs to wait for improvement before stopping.
        min_delta (float): Minimum required change in validation loss to qualify as an improvement.
        best_loss (float): The best (lowest) validation loss observed so far.
        counter (int): Number of consecutive epochs without improvement.
        should_stop (bool): Flag indicating whether training should stop.
        best_state (dict): A copy of the model's best state dictionary (weights) when the best loss was recorded.
    """

    def __init__(self, patience=20, min_delta=1e-4):
        """
        Initializes the EarlyStopping object.

        Args:
            patience (int, optional): Number of epochs with no improvement after which training will be stopped. 
                                      Defaults to 20.
            min_delta (float, optional): Minimum change in validation loss to qualify as an improvement. 
                                         Defaults to 1e-4.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False
        self.best_state = None

    def step(self, val_loss, model):
        """
        Performs a check at the end of each epoch to determine if training should stop.

        Args:
            val_loss (float): The current epoch’s validation loss.
            model (torch.nn.Module): The model being trained, whose state will be saved if improvement is detected.

        Returns:
            bool: True if training should stop (no improvement for 'patience' epochs), False otherwise.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop

