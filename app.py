import requests
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
import numpy
import seaborn as sns

sns.set_theme()

# create a difference transform of the dataset
def difference(dataset):
	diff = list()
	for i in range(1, len(dataset)):
		value = dataset[i] - dataset[i - 1]
		diff.append(value)
	return numpy.array(diff)

def get_player_info(username):
    payload = {'game': 'aoe2de', 'leaderboard_id': 3, 'search': username}

    # 1 vs 1
    URL = 'https://aoe2.net/api/leaderboard'
    res = requests.get(url = URL, params = payload)
    data_1v1 = res.json()

    # TM
    payload['leaderboard_id'] = 4
    res = requests.get(url = URL, params = payload)
    data_TM = res.json()


    return data_1v1, data_TM

# player match history
def get_player_match_history(id):
    payload = {'game': 'aoe2de', 'profile_id': id, 'count': 1000}
    URL = 'https://aoe2.net/api/player/matches'
    res = requests.get(URL, params = payload)
    data = res.json()
    df = pd.DataFrame(pd.DataFrame(data))
    return df

# player rating history
def get_player_rating_history(id, leaderboard_id):
    payload = {'game': 'aoe2de', 'leaderboard_id': leaderboard_id, 'profile_id': id, 'count': 10000}
    URL = 'https://aoe2.net/api/player/ratinghistory'
    res = requests.get(URL, params = payload)
    data = res.json()
    df = pd.DataFrame(pd.DataFrame(data))
    return df

# fit and evaluate an AR model
def train_model(df, N, desired_elo):
    X = df['rating'].values
    X = X[::-1]
    data = X
    X = difference(X)
    window_size = [1, 2, 5]
    model = AutoReg(X, lags = window_size)
    model_fit = model.fit()
    last_ob = data[len(data) - 1]
    # make prediction
    predictions = model_fit.predict(start=len(data), end=len(data))
    # transform prediction
    yhat = predictions + last_ob

    df_data = pd.DataFrame()
    df_data['Actual Rating'] = data
    predicted_data = data

    # num of max games to predict
    count_games = 0
    for _ in range(N):

        last_ob = predicted_data[len(predicted_data) - 1]

        # make prediction
        predictions = model_fit.predict(start=len(predicted_data), end=len(predicted_data), dynamic = False)

        # transform prediction
        yhat = predictions + last_ob

        predicted_data = np.append(predicted_data, int(yhat) )

        if int(yhat) >= desired_elo:
            break

        count_games += 1

    return count_games, predicted_data

st.title('AoE2: DE - How many games to be a pro?')

username = st.text_input('Insert your username of AoE2:DE', 'Ivanrs297')

profile_info_1v1, profile_info_TM = get_player_info(username)

if len(profile_info_TM['leaderboard']) > 0:

    profile_id = profile_info_1v1['leaderboard'][0]['profile_id']
    name = profile_info_1v1['leaderboard'][0]['name']
    elo_1v1 = profile_info_1v1['leaderboard'][0]['rating']
    elo_TM = profile_info_TM['leaderboard'][0]['rating']

    st.header(f'Hi {name}!')
    st.subheader(f'Your ELO 1 vs 1 is **{elo_1v1}**')

    df_rating_1v1 = get_player_rating_history(profile_id, 3)
    df_rating_TM = get_player_rating_history(profile_id, 4)

    st.write("""#### 1 vs 1 -  Game History""")
    rating_1v1 = pd.DataFrame()
    rating_1v1['1 vs 1 Rating'] = df_rating_1v1['rating'].values[::-1]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rating_1v1)
    plt.ylabel('ELO')
    plt.xlabel('# Game')
    st.pyplot(fig)

    st.subheader(f'Your ELO TM is **{elo_TM}**')
    st.write("""#### Team Match -  Game History""")
    rating_TM = pd.DataFrame()
    rating_TM['TM Rating'] = df_rating_TM['rating'].values[::-1]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rating_TM)
    plt.ylabel('ELO')
    plt.xlabel('# Game')
    st.pyplot(fig)


    st.write("""
        # What is your desired ELO?
    """)

    desired_elo = st.slider('Select desired ELO', 900, 2800, 2000)


    N = 1000

    # 1 vs 1 Results
    st.write("""
        ## 1 vs 1
        """)
    count_1v1, pred_1v1 = train_model(df_rating_1v1, N, desired_elo)
    if count_1v1 >= N:
        st.write("You need more than 1000 games, noob.")
    else:
        st.write("Total games to be a pro: ", count_1v1)

    # TM Results
    st.write("""
    ## Team Match
    """)
        
    count_TM, pred_TM = train_model(df_rating_TM, N, desired_elo)
    if count_TM >= N:
        st.write("You need more than 1000 games, noob.")
    else:
        st.write("Total games to be a pro: ", count_TM)


    st.write("""
    ## Expected Win Rate
    According to your past games.
    """)
        
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(pred_TM, label ="TM")
    ax.plot(pred_1v1, label ="1vs1")
    plt.ylabel('ELO')
    plt.xlabel('# Game')
    ax.legend()
    st.pyplot(fig)

else:
    st.write("""
        ## Error. User not found.
    """)


st.write("Created by IvanR - Using API from [AoE2.net](https://aoe2.net/)")

