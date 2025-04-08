import matplotlib.pyplot as plt
import time
import torch
import IPython
from IPython.core.display_functions import clear_output


def train_model(model, loss_fn, optimizer, x_train, y_train, x_test, y_test, save_path):
    """
    Train the provided model and save it whenever the loss is lower than the previous best.

    Args:
        model (torch.nn.Module): The model to train.
        loss_fn (callable): The loss function to use.
        optimizer (torch.optim.Optimizer): The optimizer for training.
        x_train (torch.Tensor): Training input data.
        y_train (torch.Tensor): Training target data.
        x_test (torch.Tensor): Test input data.
        y_test (torch.Tensor): Test target data.
        save_path (str): Path to save the model when a new best loss is achieved.
    """
    best_loss = float('inf')
    start_time = time.time()
    times, losses = [], []
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 5))

    while True:
        # Forward pass
        outputs = model(x_train)
        loss = loss_fn(outputs, y_train)
        test_loss = loss_fn(model(x_test), y_test)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Record time and loss
        elapsed_time = time.time() - start_time
        times.append(elapsed_time)
        losses.append(loss.item())

        # Save the model if the loss is lower than the best loss
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), save_path)
            print(f"New best loss: {best_loss:.4f}. Model saved to {save_path}.")

        # Update the graph
        ax.clear()
        ax.plot(times, losses, label="Training Loss")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Loss")
        ax.set_title("Loss vs Time")
        ax.legend()
        ax.grid()
        plt.pause(0.01)  # Pause briefly to update the plot

        print(f"Current Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}")