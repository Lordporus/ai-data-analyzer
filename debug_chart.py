
import plotly.graph_objects as go
import plotly.io as pio

try:
    print("Creating figure...")
    fig = go.Figure(go.Scatter(x=[1, 2, 3], y=[4, 1, 2]))
    print("Writing image...")
    fig.write_image("debug_chart.png")
    print("Image written successfully.")
except Exception as e:
    print(f"Error: {e}")
