import dash
from dash import Dash, dcc, html, Input, Output
from dash.dependencies import Input, Output, State
from transformers import BertTokenizer, BertModel
from transformers import AutoModel, AutoTokenizer
from sklearn.svm import SVC
import torch
import numpy as np
import pickle


external_stylesheets = ['static/styles.css']

model_name = 'bert-base-uncased'
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)



def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


with open('Model Training/BERT_SVM_model_restructured.pkl', 'rb') as f:
    final_model = pickle.load(f)


def extract_features(text):
    # Tokenize the text and add special tokens for BERT
    input_ids = torch.tensor(tokenizer.encode(text, add_special_tokens=True)).unsqueeze(0)
    # Encode the input tokens with BERT and get the hidden states
    outputs = model(input_ids)
    hidden_states = outputs.last_hidden_state
    # Apply max pooling to the hidden states to get a fixed-length representation of the text
    pooled = torch.max(hidden_states, 1)[0]
    return pooled.detach().numpy()



app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div([
    html.Center(html.H1(children="Stance Detection using BERT and Bi-LSTM", className="header")),
    html.Br(),
    html.P(children="Post your tweet here:", className="h4-content"),
    dcc.Textarea(id='input-tweet', placeholder='Enter tweet here...', className="textarea", style={'width': '100%', 'height': 100}),
    html.Br(),
    html.P(children="Select the target:", className="h4-content"),
    dcc.Dropdown(
        id='target-dropdown',
        options=[
            {'label': 'Atheism', 'value': 'atheism'},
            {'label': 'Legalization of abortion', 'value': 'legalization of abortion'},
            {'label': 'Feminist Movement', 'value': 'feminist movement'},
            {'label': 'Climate change is a real concern', 'value': 'climate change is a real concern'},
            {'label': 'Gun Laws', 'value': 'gun laws'},
            {'label': 'Racism', 'value': 'racism'},
            {'label': 'Immigration', 'value': 'immigration'},
        ],
        value='atheism', className="dropdown"
    ),
    html.Br(),
    html.Br(),
    html.Center(html.Button('Submit', id='submit-button', n_clicks=0, className="button")),
    html.Br(),
    html.Center(html.Div(id='output-container', className="bigger")),
    html.Br(),
    html.Br(),
    html.Center(html.H4(children="Powered by G9: Stance Detection", className="header"))
])
@app.callback(
    Output(component_id='output-container', component_property='children'),
    Input(component_id='submit-button', component_property='n_clicks'),
    State(component_id='input-tweet', component_property='value'),
    State(component_id='target-dropdown', component_property='value')
)
def update_output(n_clicks, input_tweet, target):
    if n_clicks > 0:
        
        inputstring = ""

        if str(target) == "atheism":
            inputstring = str(input_tweet) + " is related to " + str(target) + " which is defined as the doctrine or belief that there is no God"
        elif str(target) == "legalization of abortion":
            inputstring = str(input_tweet) + " is related to " + str(target) + " which is defined the act of making lawful  termination of pregnancy"
        elif str(target) == "feminist movement":
            inputstring = str(input_tweet) + " is related to " + str(target) + " which is defined as a supporter of feminism a change of position that does not entail a change of location"
        elif str(target) == "climate change is a real concern":
            inputstring = str(input_tweet) + " is related to " + str(target) + " which is defined as the weather in some location averaged over some long period of time an event that occurs when something passes from one state or phase to another have the quality of being; (copula, used with an adjective or a predicate noun) a metric unit of length equal to one ten billionth of a meter (or 0.0001 micron); used to specify wavelengths of electromagnetic radiation any rational or irrational number something that interests you because it is important or affects you"
        elif str(target) == "gun laws":
            inputstring = str(input_tweet) + " is related to " + str(target) + " which is defined as a weapon that discharges a missile at high velocity (especially from a metal tube or barrel) the first of three divisions of the Hebrew Scriptures comprising the first five books of the Hebrew Bible considered as a unit"
        elif str(target) == "racism":
            inputstring = str(input_tweet) + " is related to " + str(target) + " which is defined as the prejudice that members of one race are intrinsically superior to members of other races"
        elif str(target) == "immigration":
            inputstring = str(input_tweet) + " is related to " + str(target) + " which is defined as migration into a place (especially migration to a country of which you are not a native in order to settle there)"

        # inputstring = str(input_tweet)+" is related to "+str(target)
        features = extract_features(inputstring)
        stance = final_model.predict(features)
        print(stance)
        stance_text = "Sorry, some issue with predicting output!!!"


        if int(stance[0]) == 0:
            stance_text = html.Span("Stance: AGAINST", style={'color': 'red'})
        elif int(stance[0]) == 1:
            stance_text = html.Span("Stance: FAVOR", style={'color': 'green'})
        elif int(stance[0]) == 2:
            stance_text = html.Span("Stance: NEITHER", style={'color': 'orange'})
        
        # if int(np.argmax(stance[0])) == 0:
        #     stance_text = html.Span("Stance: AGAINST", style={'color': 'red'})
        # elif int(np.argmax(stance[0])) == 1:
        #     stance_text = html.Span("Stance: FAVOR", style={'color': 'green'})
        # elif int(np.argmax(stance[0])) == 2:
        #     stance_text = html.Span("Stance: NEITHER", style={'color': 'orange'})
        return html.Center(stance_text)

if __name__ == '__main__':
    app.run_server(debug=True)