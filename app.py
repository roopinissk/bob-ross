import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Read the CSV data
bobross = pd.read_csv('bob_ross_paintings.csv')


def color_freq():

    #color mapping
    color_map = {
        'Black_Gesso': '#000000',
        'Bright_Red': '#DB0000',
        'Burnt_Umber': '#8A3324',
        'Cadmium_Yellow': '#FFEC00',
        'Dark_Sienna': '#5F2E1F',
        'Indian_Red': '#CD5C5C',
        'Indian_Yellow': '#FFB800',
        'Liquid_Black': '#000000',
        'Liquid_Clear': '#FFFFFF',
        'Midnight_Black': '#000000',
        'Phthalo_Blue': '#0C0040',
        'Phthalo_Green': '#102E3C',
        'Prussian_Blue': '#021E44',
        'Sap_Green': '#0A3410',
        'Titanium_White': '#FFFFFF',
        'Van_Dyke_Brown': '#221B15',
        'Yellow_Ochre': '#C79B00',
        'Alizarin_Crimson': '#4E1500'
    }

    # Counting the frequency of each color
    color_columns = list(color_map.keys())
    color_usage = bobross[color_columns].sum().sort_values(ascending=False)

    # bar plot
    fig = plt.figure(figsize=(15, 8), facecolor='#E6F3FF')  # Light blue background
    ax = plt.axes()
    ax.set_facecolor('#E6F3FF')  # Light blue background for the plot area

    bars = plt.bar(color_usage.index, color_usage.values, color=[color_map[c] for c in color_usage.index])

    #value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height}',
                ha='center', va='bottom', fontweight='bold')
    plt.xlabel('Colors', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Paintings', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')

    #gridlines
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    st.title("Frequency of Colors in Bob Ross Paintings")
    st.pyplot(fig)
    return fig


def color_heatmap():


    #a. color palette similarities
    # Converting season and episode to numeric types
    bobross['season'] = pd.to_numeric(bobross['season'])
    bobross['episode'] = pd.to_numeric(bobross['episode'])

    #list of all color columns
    color_column = ['Black_Gesso', 'Bright_Red', 'Burnt_Umber', 'Cadmium_Yellow', 'Dark_Sienna', 'Indian_Red', 'Indian_Yellow', 'Liquid_Black', 'Liquid_Clear', 'Midnight_Black', 'Phthalo_Blue', 'Phthalo_Green', 'Prussian_Blue', 'Sap_Green', 'Titanium_White', 'Van_Dyke_Brown', 'Yellow_Ochre', 'Alizarin_Crimson']

    # a. Color Palette Similarities

    # Calculating the correlation matrix for color usage
    color_correlation = bobross[color_column].corr()

    # Plotting a heatmap of color correlations
    fig=plt.figure(figsize=(10, 8))
    sns.heatmap(color_correlation, annot=False, cmap='coolwarm', center=0)
    plt.title('Color Usage Correlation Heatmap')
    plt.tight_layout()
    plt.show()

    st.title("Color Usage Correlation Heatmap")
    st.pyplot(fig)
    # 2. Temporal Analysis of Colors

    
    #b. Temporal Analysis of Colors

    #average number of colors used per season
    colors_per_season = bobross.groupby('season')[color_column].mean()

    #Plotting the trend of color usage over seasons
    fig=plt.figure(figsize=(12, 8))
    sns.heatmap(colors_per_season.T, cmap='YlOrRd')
    plt.title('Color Usage Trends Across Seasons')
    plt.xlabel('Season')
    plt.ylabel('Color')
    plt.show()  

    st.title("Color Usage Trends Across Seasons")
    st.pyplot(fig)

def season_colors():
    # Season against colors - kde
    fig = plt.figure(figsize=(15,7))
    season_color = sns.violinplot(data=bobross, y='num_colors', x='season', palette= 'pastel', linewidth= 0.5)
    season_color.set_xlabel('Season')
    season_color.set_ylabel('Number of colors used')
    st.title('Season vs Number of Colors: Violin plot (a)')
    st.pyplot(fig)
    

def season_colors_cut():
    # Season against colors- cutting
    fig = plt.figure(figsize=(15,7))
    season_color = sns.violinplot(data=bobross, y='num_colors', x='season', palette= 'pastel', linewidth= 0.5, cut=0)

    #Axis labels
    season_color.set_xlabel('Season')
    season_color.set_ylabel('Number of colors used')

    st.title('Violin Plot (b)')
    st.pyplot(fig)

    
    #making box plot with 1 to 11; 11 to 22; 21 to 31
    season1_10 = bobross[:131][['season','num_colors']]
    season11_20 = bobross[131:261][['season','num_colors']]
    season21_31 = bobross[261:][['season','num_colors']]

    # 1 to 11
    fig2 = plt.figure(figsize=(15,7))
    OneToEleven = sns.boxplot(season1_10, x='season', y= 'num_colors', palette= 'pastel')

    #labels
    OneToEleven.set_xlabel('Season')
    OneToEleven.set_ylabel('Number of colors used')

    st.title('Season vs number of colors - Box plot')
    st.title('Seasons 1 to 11')
    st.pyplot(fig2)

    #11 to 21
    fig3 = plt.figure(figsize=(15,7))
    ElevenToTwentyone = sns.boxplot(season11_20, x='season', y= 'num_colors', palette= 'pastel')

    #labels
    ElevenToTwentyone.set_xlabel('Season')
    ElevenToTwentyone.set_ylabel('Number of colors used')
    ElevenToTwentyone.set_title('Season vs Number of Colors')
    

    st.title('Seasons 11 to 21')
    st.pyplot(fig3)

    #21 to 31
    fig4 = plt.figure(figsize=(15,7))
    TwentyOneToThirtyOne = sns.boxplot(season21_31, x='season', y= 'num_colors', palette= 'pastel')

    #labels
    TwentyOneToThirtyOne.set_xlabel('Season')
    TwentyOneToThirtyOne.set_ylabel('Number of colors used')
    TwentyOneToThirtyOne.set_title('Season vs Number of Colors')
    

    st.title('Seasons 21 to 31')
    st.pyplot(fig4)

## Q1. "What are the top ten most commonly used colors pallets does Bob use in his paintings?  (total of 403 paintings)
def top_10():

    num_paintings = len(bobross)
    print(f'Total Number of paintings: {num_paintings}')

    # Example: Count of unique colors
    color_counts = bobross['colors'].explode().value_counts()

    # Limit to top N color combinations
    top_n = 10  #
    color_counts = color_counts.head(top_n)

    # Plot
    fig = plt.figure(figsize=(10, 8))  # Increase the figure size for better readability
    sns.barplot(x=color_counts.values, y=color_counts.index)
    plt.xlabel('Count')
    plt.ylabel('Color')
    
    st.title('Top 10 Most Frequent Colors')    
    st.pyplot(fig)


def jacard():

    episodes = bobross['episode'].unique()

    # Create the unique identifier for each season and episode combination
    bobross['season_episode'] = 'S' + bobross['season'].astype(str) + '-E' + bobross['episode'].astype(str)
    # Get list of unique season_episode
    episodes = bobross['season_episode'].unique()

    # Initialize a dictionary to store sets of colors for each season_episode
    episode_colors = {episode: set(bobross[bobross['season_episode'] == episode]['colors'].explode()) for episode in episodes}

    # Initialize a DataFrame to store Jaccard Similarity values
    jaccard_df = pd.DataFrame(index=episodes, columns=episodes)

    # Compute Jaccard Similarity for each pair of season_episode
    for episode1 in episodes:
        for episode2 in episodes:
            intersection = len(episode_colors[episode1].intersection(episode_colors[episode2]))
            union = len(episode_colors[episode1].union(episode_colors[episode2]))
            jaccard_similarity = intersection / union if union != 0 else 0
            jaccard_df.loc[episode1, episode2] = jaccard_similarity

    # Convert to numeric values
    jaccard_df = jaccard_df.apply(pd.to_numeric)

    # Display the Jaccard Similarity DataFrame
   
    # Plot the Jaccard Similarity matrix using heatmap without annotations; use Seaborn for visualization
    fig= plt.figure(figsize=(15, 12))
    sns.heatmap(jaccard_df, cmap='viridis', annot=False, cbar=True)
    plt.xlabel('Episodes')
    plt.ylabel('Episodes')
    st.title('Jaccard Similarity between Episodes')
    st.pyplot(fig)
# Here we can see the earlier seasons had similar type of colors as compared to later ones. Especially the first 6 seasons(and episodes) for example have unique palletes that were not used in later shows

    # appears to be only 0.0 or 1.0 no gradient of values.

def color_usage():
    #

    # This below example is another shot at this question, while not as complelling as above, but another way to see this perhaps and you can see similar partition of coolors
    color_usage = bobross.groupby('season')['colors'].apply(lambda x: x.explode().value_counts()).unstack().fillna(0)
    fig = plt.figure()
    sns.heatmap(color_usage, cmap='BuPu')
    st.title('Color Usage Across Seasons')
    st.pyplot(fig)

def word_cloud():
    # How did he describe his paintings in the titles?  Here we do full dataset but you could also group of top ten pallete colors, or first six seasons

    # Extract the titles
    titles = bobross['painting_title']

    from collections import Counter
    import re

    # Tokenize the titles
    tokens = [re.findall(r'\b\w+\b', title.lower()) for title in titles]
    flat_tokens = [item for sublist in tokens for item in sublist]

    # Count the frequency of each word
    word_counts = Counter(flat_tokens)

    # Display the most common words
    print(word_counts.most_common(10))

    from wordcloud import WordCloud
    import matplotlib.pyplot as plt

    # Generate the word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_counts)

    # Plot the word cloud
    fig = plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.title('Word Cloud of All Bob Ross Painting Titles')
    st.pyplot(fig)

def vis_sid():
    import plotly.express as px

    #Creating a Hover plot that displays show details when the cursor is hovered over it
    #Create a scatter plot with hover information
    figv1 = px.scatter(bobross,
                    x='season',
                    y='episode',
                    color = 'season', #added later to differentiate colors by seasons
                    hover_name='painting_title',
                    hover_data={'episode': True, 'season': True,},
                    title='Bob Ross Paintings Details')

    # Update layout
    figv1.update_layout(xaxis_title='Season', yaxis_title='Episode')

    st.title("Season vs Episode")
    st.plotly_chart(figv1)

    #Creating a 3D plot that shows the numbers of colors used per episode per season
    import plotly.graph_objects as go

    bobross['num_colors'] = pd.to_numeric(bobross['num_colors'])

    # Create a 3D scatter plot
    figv2 = go.Figure(data=[go.Scatter3d(
        x = bobross['episode'],
        y = bobross['season'],
        z = bobross['num_colors'],
        mode = 'markers',
        marker = dict(size=5, color=bobross['num_colors'], colorscale='Viridis', colorbar_title='No. of Colors Used'),
        text = bobross['painting_title'],
        hoverinfo = 'text'
    )])

    # Update layout
    figv2.update_layout(scene=dict(
                        xaxis_title = 'Episode',
                        yaxis_title = 'Season',
                        zaxis_title = 'Number of Colors'),
                    title= '3D Plot of Numbers of Colors used in an Episode')
    
    st.title("3D Plot of Numbers of Colors used in an Episode")
    st.plotly_chart(figv2)


    # Create the 2D scatter plot
    figv3 = go.Figure(data=[go.Scatter(
        x = bobross['season'],
        y = bobross['episode'],
        mode='markers',
        marker=dict(size=10, color=bobross['num_colors'], colorscale='Viridis', colorbar_title='No. of Colors Used'),
        text = bobross['painting_title'],
        hoverinfo='text'
    )])

    # Update layout
    figv3.update_layout(
        xaxis_title='Season',
        yaxis_title='Episode',
        title='2D Plot of Numbers of Colors used in an Episode'
    )
    st.title("2D Plot of Numbers of Colors used in an Episode")
    st.plotly_chart(figv3)



    
if __name__=="__main__":

    st.title("Analysis of Bob Ross Paintings")
    st.write("Bob Ross, known for his TV series 'The Joy of Painting' (1983-1994), inspired many to learn his painting technique.For this project our team selected a publicly available dataset containing the metadata for all paintings featured in this popular TV show ‘The Joy of Painting”. This curated dataset was pulled from the GitHub repository jwilber/BobRossPaintings which describes the upstream web-scraping of all the paintings featured in TwoInchBrush.com resulting in the creation of a csv metadata file that we will be using for this project.")

    top_10()
    jacard()
    color_usage()
    word_cloud()

    c_freq = color_freq()
    color_heatmap()
    season_colors()
    season_colors_cut()
    vis_sid()