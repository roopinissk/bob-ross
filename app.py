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


    # Convert season and episode to numeric types
    bobross['season'] = pd.to_numeric(bobross['season'])
    bobross['episode'] = pd.to_numeric(bobross['episode'])

    # Create a new column for the total number of colors used
    bobross['total_colors'] = bobross['num_colors']

    # Create a list of all color columns
    color_columns = ['Black_Gesso', 'Bright_Red', 'Burnt_Umber', 'Cadmium_Yellow', 'Dark_Sienna',
                    'Indian_Red', 'Indian_Yellow', 'Liquid_Black', 'Liquid_Clear', 'Midnight_Black',
                    'Phthalo_Blue', 'Phthalo_Green', 'Prussian_Blue', 'Sap_Green', 'Titanium_White',
                    'Van_Dyke_Brown', 'Yellow_Ochre', 'Alizarin_Crimson']

    # 1. Color Palette Similarities

    # Calculating the correlation matrix for color usage
    color_correlation = bobross[color_columns].corr()

    # Plotting a heatmap of color correlations
    plt.figure(figsize=(12, 10))
    sns.heatmap(color_correlation, annot=False, cmap='coolwarm', center=0)
    plt.title('Color Usage Correlation Heatmap')
    plt.tight_layout()
    plt.show()

    # 2. Temporal Analysis of Colors

    # Calculating the average number of colors used per season
    colors_per_season = bobross.groupby('season')[color_columns].mean()

    # Plotting the trend of color usage over seasons
    fig=plt.figure(figsize=(12, 6))
    sns.heatmap(colors_per_season.T, cmap='YlOrRd', cbar_kws={'label': 'Average Usage'})
    plt.xlabel('Season')
    plt.ylabel('Color')
    plt.tight_layout()

    st.title("Color Usage Trends Across Seasons")
    st.pyplot(fig)

def season_colors():
    # Season against colors - kde
    fig = plt.figure(figsize=(15,7))
    season_color = sns.violinplot(data=bobross, y='num_colors', x='season', palette= 'pastel', linewidth= 0.5)
    season_color.set_xlabel('season')
    season_color.set_ylabel('num of colors used')
    st.title('Season vs Number of Colors')
    st.pyplot(fig)
    

def season_colors_cut():
    # Season against colors- cutting
    fig = plt.figure(figsize=(15,7))
    season_color = sns.violinplot(data=bobross, y='num_colors', x='season', palette= 'pastel', linewidth= 0.5, cut=0)

    #Axis labels
    season_color.set_xlabel('season')
    season_color.set_ylabel('num of colors used')

    st.title('Season vs Number of Colors')
    st.pyplot(fig)


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


if __name__=="__main__":

    top_10()
    jacard()
    color_usage()
    word_cloud()

    c_freq = color_freq()
    color_heatmap()
    season_colors()
    season_colors_cut()