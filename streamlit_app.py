import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State
import plotly.express as px
import pandas as pd

# Sample data for dummy chart
df = pd.DataFrame({
    "x": range(10),
    "y": [i ** 1.5 for i in range(10)]
})
fig = px.line(df, x="x", y="y", title="Dummy Chart")
fig.update_layout(template="plotly_dark", paper_bgcolor="black", plot_bgcolor="black")

# Define a news card component (dummy content)
def news_card(headline, snippet, sentiment, link):
    return dbc.Card(
        dbc.CardBody([
            html.H5(headline, className="card-title", style={"color": "white"}),
            html.P(snippet, className="card-text", style={"color": "lightgray"}),
            dbc.Button("→", color="secondary", size="sm", href=link, target="_blank"),
            html.Div(f"Sentiment: {sentiment}", style={"color": "lightgreen", "marginTop": "10px"})
        ]),
        style={"marginBottom": "10px", "backgroundColor": "#222222", "border": "none"}
    )

# Sample news cards list
news_cards = [
    news_card("Market Rally", "Stocks surged amid strong earnings reports.", "Positive", "https://example.com/story1"),
    news_card("Economic Update", "Inflation numbers show signs of stabilizing.", "Neutral", "https://example.com/story2"),
    news_card("Tech News", "A major tech firm announces a new product line.", "Positive", "https://example.com/story3"),
]

# Chatbot dummy messages container (for now static)
chat_history = html.Div(id="chat-history", children=[
    html.Div("Bot: Welcome to Trading AI. How can I help you today?", style={"color": "lightblue", "padding": "5px"})
], style={"height": "100%", "overflowY": "auto", "padding": "10px", "backgroundColor": "#333333", "borderRadius": "5px"})

# Chat input area
chat_input = dbc.InputGroup(
    [
        dbc.Input(id="chat-input", placeholder="Ask a question...", type="text", style={"backgroundColor": "#444", "color": "white"}),
        dbc.Button("Send", id="send-btn", color="primary")
    ],
    className="mt-2"
)

# Create the main layout
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "Trading AI Dashboard"

app.layout = dbc.Container(fluid=True, style={"padding": "0", "backgroundColor": "black", "height": "100vh"}, children=[
    dbc.Row([
        # Left Section: Merged first two columns with dummy chart
        dbc.Col(
            dcc.Graph(figure=fig, style={"height": "100vh"}),
            width=8, style={"padding": "0"}
        ),
        # Right Section: Divided into two vertical parts
        dbc.Col([
            # Top: News cards area (top two rows merged) with hidden y-overflow if necessary
            html.Div(
                news_cards,
                id="news-container",
                style={
                    "height": "65vh",
                    "overflowY": "auto",
                    "padding": "10px",
                    "backgroundColor": "#111111",
                    "borderBottom": "1px solid #444"
                }
            ),
            # Bottom: Chatbot UI
            html.Div([
                chat_history,
                chat_input
            ],
            id="chat-container",
            style={
                "height": "35vh",
                "padding": "10px",
                "backgroundColor": "#111111"
            })
        ], width=4, style={"padding": "0", "display": "flex", "flexDirection": "column", "height": "100vh"})
    ], style={"height": "100vh", "margin": "0"})
])

# Dummy callback to clear input after send (extend functionality later)
@app.callback(
    Output("chat-input", "value"),
    Input("send-btn", "n_clicks"),
    State("chat-input", "value"),
    prevent_initial_call=True
)
def handle_send(n_clicks, message):
    # Here you would normally process the message and update the chat history
    return ""

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, dev_tools_ui=False, dev_tools_hot_reload=False)