import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import pandas as pd

import torch
import torch.nn.functional as F

def smoothen_metric(metric, window_size=40):
    return pd.Series(metric).rolling(window_size, min_periods=1).mean()


def plot_metric(steps, metric_values):
    plt.plot(
        np.linspace(0, steps, len(metric_values)), 
        metric_values)
    

def plot_metrics(metrics, steps, model, xlim=None):
    training_loss = smoothen_metric([m["training_loss"] for m in metrics])
    
    plt.title("Training Loss")
    plt.xlim(xlim)
    plot_metric(steps, training_loss)
    plt.show()
    
    print(f"Training loss: {training_loss.iloc[-1]}")

def train_model(model, criterion, optimizer, train_loader, test_loader, epochs=3, print_every=40):
    steps = 0
    metrics = []

    total_steps = epochs * len(iter(train_loader))
    
    for e in range(epochs):
        for data in iter(train_loader):
            images = data['image'].type(torch.FloatTensor)
            key_pts = data['keypoints'].view(data['keypoints'].size(0), -1).type(torch.FloatTensor)
            
            optimizer.zero_grad()

            output = model(images)
            loss = criterion(output, key_pts)
            loss.backward()
            optimizer.step()
            
            metrics.append({
                "training_loss": loss.item()
            })

            if steps % print_every == 0:
                clear_output(wait=True)

                plot_metrics(metrics, steps, model, xlim=(0, total_steps))

            steps += 1
