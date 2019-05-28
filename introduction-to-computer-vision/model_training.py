import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import pandas as pd

import torch

def smoothen_metric(metric, window_size=40):
    return pd.Series(metric).rolling(window_size, min_periods=1).mean()


def evaluate_metric_on_dataset(model, loader, evaluate_metric):
    ground_truth, predictions = map(
        torch.cat, 
        zip(*[
            [ground_truth, model(images)] 
            for images, ground_truth in loader]))
    
    return evaluate_metric(ground_truth, predictions)


def eval_accuracy(ground_truth, predictions):
    return float(
        (ground_truth == predictions.argmax(dim=1))
        .type(torch.FloatTensor)
        .mean())


def plot_metric(steps, metric_values):
    plt.plot(
        np.linspace(0, steps, len(metric_values)), 
        metric_values)
    

def plot_metrics(metrics, steps, model, xlim=None):
    training_loss = smoothen_metric([m["training_loss"] for m in metrics])
    training_acc = smoothen_metric([m["training_accuracy"] for m in metrics])
    testing_acc = smoothen_metric([m["testing_accuracy"] for m in metrics], window_size=80)
    
    plt.title("Loss")
    plt.xlim(xlim)
    plot_metric(steps, training_loss)
    plt.legend(["Training Loss"])
    plt.show()
    
    plt.title("Accuracy")
    plt.xlim(xlim)
    plot_metric(steps, training_acc)
    plt.title("Training Loss")
    plot_metric(steps, testing_acc)
    plt.legend(["Training Accuracy", "Testing Accuracy"])
    plt.show()
    
    print(f"Training loss: {training_loss.iloc[-1]}")
    print(f"Training Accuracy: {training_acc.iloc[-1]}")
    print(f"Testing Accuracy: {testing_acc.iloc[-1]}")

def train_model(model, criterion, optimizer, train_loader, test_loader, epochs=3, print_every=40):
    steps = 0
    metrics = []

    total_steps = epochs * len(iter(train_loader))
    
    for e in range(epochs):
        for images, labels in iter(train_loader):
            optimizer.zero_grad()

            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            metrics.append({
                "training_loss": loss.item(),
                "training_accuracy": eval_accuracy(labels, output),
                "testing_accuracy": (
                    evaluate_metric_on_dataset(model, test_loader, eval_accuracy)
                    if steps % print_every == 0
                    else metrics[-1]["testing_accuracy"])
            })

            if steps % print_every == 0:
                clear_output(wait=True)

                plot_metrics(metrics, steps, model, xlim=(0, total_steps))

            steps += 1
