import plotly.graph_objects as go
import polars as pl
import numpy as np

def plot_predictions(df: pl.DataFrame, predicted_labels: np.array, predicted_proba: np.array):
    """
    Create a plot of price movement with buy and sell predictions.
    
    Parameters:
    df (polars.DataFrame): Dataset containing 'time' and 'price' columns
    predicted_labels (numpy.array): Array of predicted labels (0, 1, or 2)
    predicted_proba (numpy.array): Array of predicted probabilities (shape: N x 3)
    
    Returns:
    None (displays the plot)
    """
    # Ensure the dataframe has 'time' and 'price' columns
    if 'time' not in df.columns or 'price_last' not in df.columns:
        raise ValueError("DataFrame must contain 'time' and 'price' columns")
    
    # Convert Polars DataFrame to numpy arrays for easier indexing
    df = df.sort('time')
    time_array = df['time'].to_numpy()
    price_array = df['price_last'].to_numpy()
    
    # Create the main price line trace
    trace_price = go.Scatter(
        x=time_array,
        y=price_array,
        mode='lines',
        name='Price'
    )
    
    # Create buy (label 1) and sell (label 2) traces
    buy_indices = np.where(predicted_labels == 1)[0]
    sell_indices = np.where(predicted_labels == 2)[0]
    
    trace_buy = go.Scatter(
        x=time_array[buy_indices],
        y=price_array[buy_indices],
        mode='markers',
        name='Buy',
        marker=dict(color='blue', size=10),
        text=[f"Buy probability: {predicted_proba[i, 1]:.2f}" for i in buy_indices],
        hoverinfo='text+x+y'
    )
    
    trace_sell = go.Scatter(
        x=time_array[sell_indices],
        y=price_array[sell_indices],
        mode='markers',
        name='Sell',
        marker=dict(color='red', size=10),
        text=[f"Sell probability: {predicted_proba[i, 2]:.2f}" for i in sell_indices],
        hoverinfo='text+x+y'
    )
    
    # Combine all traces
    data = [trace_price, trace_buy, trace_sell]
    
    # Create the layout
    layout = go.Layout(
        title='Price Movement with Buy/Sell Predictions',
        xaxis=dict(title='Time'),
        yaxis=dict(title='Price'),
        hovermode='closest'
    )
    
    # Create and show the figure
    fig = go.Figure(data=data, layout=layout)
    fig.show()

