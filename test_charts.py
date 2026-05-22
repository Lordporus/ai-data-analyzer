
import plotly.graph_objects as go
import plotly.io as pio
import sys

print("Starting chart test...")
try:
    fig = go.Figure(data=go.Bar(y=[2, 3, 1]))
    print("Figure created. Attempting write_image...")
    pio.write_image(fig, "test_chart.png")
    print("Chart saved successfully.")
except Exception as e:
    print(f"Chart generation failed: {e}")
except ImportError:
    print("Kaleido not installed.")
print("Done.")
