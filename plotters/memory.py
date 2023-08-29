import plotly.graph_objects as go
from plotly.subplots import make_subplots
from databuilders.pd import PandasMemoryDatabuilder

from plotters.base import BasePlotter


class MemoryPlotter(BasePlotter):
    def __init__(self, databuilder: PandasMemoryDatabuilder | None = None):
        super().__init__(
            databuilder=databuilder,
            databuilder_class=PandasMemoryDatabuilder,
        )

    def get_plot(self) -> go.Figure:
        data = self.databuilder.get_data()
        plot_data = [[
            go.Candlestick(
                x=data['open_time'],
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                increasing_line_color='lightgray',
                decreasing_line_color='darkgray',
                name=self.databuilder.memory_bank.symbol,
            ),
        ]]
        for key in data.columns:
            if key.startswith('sma') or key.startswith('ema'):
                plot_data[0].append(
                    go.Scatter(
                        x=data['open_time'],
                        y=data[key],
                        name=key.upper(),
                    ),
                )
            elif key.startswith('fso') or key.startswith('sso'):
                plot_data.append([
                    go.Bar(
                        x=data['open_time'],
                        y=data[key],
                        name=key.upper(),
                    ),
                ])
            elif key.startswith('rsi'):
                plot_data.append([
                    go.Bar(
                        x=data['open_time'],
                        y=data[key],
                        name=key.upper(),
                    ),
                ])
            elif key.startswith('macd') and not key.startswith('macd_hist'):
                plot_data.append([
                    go.Scatter(
                        x=data['open_time'],
                        y=data[key],
                        name=key.upper(),
                    ),
                ])
            elif key.startswith('macd_hist'):
                plot_data.append([
                    go.Bar(
                        x=data['open_time'],
                        y=data[key],
                        name=key.upper(),
                    ),
                ])
            elif key.startswith('obv'):
                plot_data.append([
                    go.Scatter(
                        x=data['open_time'],
                        y=data[key],
                        name=key.upper(),
                    ),
                ])
        fig = make_subplots(
            rows=len(plot_data), cols=1,
            shared_xaxes=True, vertical_spacing=0.01,
        )
        for row_idx, plots in enumerate(plot_data):
            for plot in plots:
                fig.add_trace(plot, row=row_idx + 1, col=1)
        fig.update_xaxes(rangeslider_visible=False)
        fig.update_yaxes(fixedrange=False)
        return fig
