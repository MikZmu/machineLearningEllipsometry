import os

import matplotlib.pyplot as plt
import time
import torch
import IPython
from IPython.core.display_functions import clear_output
#from orca.orca_state import device


def train_model(model, loss_fn, optimizer, x_train, y_train, x_test, y_test, save_path, batch_size=32):
    os.environ['TERM'] = 'xterm'
    best_loss = float('inf')
    start_time = time.time()
    times, losses = [], []
    """plt.ion()
    fig, ax = plt.subplots(figsize=(10, 5))"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = model.to(device)
    if batch_size != 0:
        dataset = torch.utils.data.TensorDataset(x_train, y_train)
        test_dataset = torch.utils.data.TensorDataset(x_test, y_test)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    else:
        x_train, y_train = x_train.to(device), y_train.to(device)
        x_test, y_test = x_test.to(device), y_test.to(device)

    def r2_loss(y_pred, y_true):
        ss_total = torch.sum((y_true - torch.mean(y_true)) ** 2)
        ss_residual = torch.sum((y_true - y_pred) ** 2)
        r2 = 1 - (ss_residual / ss_total)
        return 1 - r2  # Loss is 1 - R²

    if batch_size != 0:
        while True:
            model.train()
            for batch_x, batch_y in dataloader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                # Forward pass
                outputs = model(batch_x)
                r2 = r2_loss(outputs, batch_y)
                loss = loss_fn(outputs, batch_y)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()

            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(model.state_dict(), save_path)
                print(f"New best loss: {best_loss:.8f}. Model saved to {save_path}.")
            print(f"Current Loss: {loss.item():.8f}")
            print(f"Current R2 Loss: {r2:.8f}")

    else:
        while True:
            # Forward pass
            outputs = model(x_train)
            loss = loss_fn(outputs, y_train)

            test_pred = 0
            with torch.no_grad():
                test_pred = model(x_test)
                test_loss = loss_fn(test_pred, y_test)
                r2_test_loss = r2_loss(test_pred, y_test)


            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #if scheduler:
             #   scheduler.step(loss)

            # Record time and loss
            elapsed_time = time.time() - start_time
            times.append(elapsed_time)
            losses.append(loss.item())
            os.system('clear')
            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(model.state_dict(), save_path)
                print(f"New best loss: {best_loss:.4f}. Model saved to {save_path}.")

            """ax.clear()
            ax.plot(times, losses, label="Training Loss")
            ax.set_xlabel("Time (seconds)")
            ax.set_ylabel("Loss")
            ax.set_title("Loss vs Time")
            ax.legend()
            ax.grid()
            plt.pause(0.01)  # Pause briefly to update the plot"""

            print(f"Current Loss: {loss.item():.8f}, Test Loss: {test_loss.item():.8f}")
            print(f"Current R2 Loss: {r2_loss(outputs, y_train).item():.8f}, Test R2 Loss: {r2_test_loss:.8f}")


def train_conv_mlp(model, loss_fn, optimizer, x_train, y_train, x_val, y_val, save_path, patience=10, max_epochs=100):
    best_val_loss = float('inf')
    patience_counter = 0

    def r2_loss(y_pred, y_true):

        ss_total = torch.sum((y_true - torch.mean(y_true)) ** 2)
        ss_residual = torch.sum((y_true - y_pred) ** 2)
        r2 = 1 - (ss_residual / ss_total)
        return 1 - r2  # Loss is 1 - R²
    r2 = 9999999999999999

    for epoch in range(max_epochs):
        # Training phase
        model.train()
        optimizer.zero_grad()
        y_pred = model(x_train)
        train_loss = loss_fn(y_pred, y_train)
        train_loss.backward()
        optimizer.step()

        # Validation phase
        model.eval()
        with torch.no_grad():
            y_val_pred = model(x_val)
            val_loss = loss_fn(y_val_pred, y_val)
            r2 = r2_loss(y_val_pred, y_val)

        print(f"Epoch {epoch+1}/{max_epochs}, Train Loss: {train_loss.item():.16f}, Val Loss: {val_loss.item():.16f}, R2 Loss: {r2:.16f}")

        # Check for improvement
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"Model saved with Val Loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print("Training complete.")