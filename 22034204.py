# -*- coding: utf-8 -*-
"""
Created on Fri May  5 07:30:57 2023

@author: HAMZA
"""

# Import the necessary libraries
import numpy as np               # Library for numerical computations
import pandas as pd              # Library for data manipulation and analysis
import matplotlib.pyplot as plt        # Library for creating visualizations
import seaborn as sns                  # Library for statistical visualizations
from scipy.stats import kurtosis       # Function to calculate kurtosis
pd.set_option('display.precision', 2)  # Set the display precision of pandas 
                                       # dataframes

# Import additional libraries
import matplotlib.patches as mpatches   # Library for creating patches (
                                        # e.g., legends)
import matplotlib.gridspec              # Library for creating grid layouts

""" Defining Functions to be used in a Program """

def load_bank_data(file_path):
    """
    Load bank marketing data from a file.
    
    Args:
        file_path (str): The path to the data file.
    
    Returns:
        pandas.DataFrame: The loaded bank marketing data.
    """
    
    # Load the data from the file
    df = pd.read_csv(file_path, delimiter=';')
    
    # Display the first 3 rows of the data
    print(df.head(n=3))
    
    # Display the column names
    print()
    print('Display the column names: ')
    print("\n".join(df.columns.to_list()))
    
    # Display the summary of the data frame
    print()
    print('Summary of the DataFrame:')
    df.info()
    
    # Display the number of records and columns in the data frame
    print()
    print("Number of records in the data frame: {}".format(df.shape[0]))
    print("Number of columns in the data frame: {}".format(df.shape[1]))
    
    # Display the number of null values in each column
    print()
    print('Number of null values in each column: ')
    print(df.isnull().sum())
    
    # Return the loaded data frame
    return df

def categorize_features(df):
    """
    Categorizes the features of a pandas dataframe into numerical and 
    categorical.

    Parameters:
    df (pandas.DataFrame): The dataframe to categorize the features of.

    Returns:
    numerical_features (list): A list of the numerical features in the 
    dataframe.
    categorical_features (list): A list of the categorical features in 
    the dataframe.
    """
    
    # Create empty lists to store numerical and categorical features
    numerical_features = []
    categorical_features = []

     # Iterate through each column in the dataframe and check if it is 
     # numerical or categorical
    for col in df.columns:
        # Check if the column contains numerical values
        if df[col].dtypes in ('int64','float64'): 
            # If it does, add it to the list of numerical features
            numerical_features.append(col)          
        else:
            # If it doesn't, add it to the list of categorical features
            categorical_features.append(col)   
            
    # Return the lists of numerical and categorical features
    return numerical_features, categorical_features   

def gs_visualize_job_and_subscription(dataframe, ax1, ax2):
    """
    This function takes a dataframe and creates two subplots: 
    - A bar chart to visualize the job distribution of clients in the bank 
    dataset.
    - A countplot to show the subscription status by job type.

    Parameters:
    dataframe (pandas dataframe): the input dataframe

    Returns:
    None
    """
    
    # Plot 1: Job Distribution
    job_counts = dataframe['job'].value_counts()
    sns.barplot(x=job_counts.values, y=job_counts.index, palette="hls", ax=ax1)
    ax1.set_title("Distribution of job types", fontsize=20, weight='bold')
    ax1.set_xlabel("Number of clients", weight='bold', fontsize=14)
    ax1.set_ylabel("Job type", weight='bold', fontsize=14)

    # Add text labels for the percentage of each job type
    for i, v in enumerate(job_counts):
        ax1.text(v, i+0.25, f'{v} ({round(v/len(dataframe)*100, 2)}%)', \
                 fontweight='bold', fontsize=10)
    
    # Adjust the limits of the x-axis
    xlim_max = job_counts.max() + 2000
    ax1.set_xlim(right=xlim_max)

    # Plot 2: Subscription by Job Type
    sns.countplot(y='job', hue='y', data=dataframe, order=job_counts.index,\
                  palette="hls", ax=ax2)
    ax2.set_title('Subscription by Job Type', fontsize=20, weight='bold')
    ax2.set_xlabel('Number of Clients', weight='bold', fontsize=14)
    ax2.set_ylabel('')  # Remove y-axis label for better alignment
    # Turn off yticks
    ax2.set_yticklabels([])

    
    # Add percentage labels to the bars
    for i, v in enumerate(job_counts):
        subscribed_count = dataframe[(dataframe['job'] == job_counts.index[i])\
                                     & (dataframe['y'] == 'yes')].shape[0]
        subscribed_percent = round(subscribed_count / v * 100, 1)
        ax2.text(v-20, i + 0.3, f'{subscribed_count} ({subscribed_percent}%)'\
         , color=sns.color_palette("hls")[3], fontsize=10, fontweight='bold')

        not_subscribed_count = dataframe[(dataframe['job'] == \
                                          job_counts.index[i]) & \
                                         (dataframe['y'] == 'no')].shape[0]
        not_subscribed_percent = round(not_subscribed_count / v * 100, 1)
        ax2.text(v-20, i - 0.1, \
                 f'{not_subscribed_count} ({not_subscribed_percent}%)',\
                 color=sns.color_palette("hls")[0], fontsize=10, \
                     fontweight='bold')

    # Adjust the limits of the x-axis
    xlim_max = job_counts.max() + 1800
    ax2.set_xlim(right=xlim_max)

    ax2.legend(loc='lower right', title='Subscription\n     status')

    # Adjust the spacing between the subplots
    plt.tight_layout()
    
    # Adjust the spacing between the subplots
    plt.subplots_adjust(wspace=0)
    
    return

def gs_visualize_education_distribution(dataframe, ax):
    """
    This function takes a dataframe and creates a pie chart to visualize
    the education distribution of clients in the bank dataset.

    Parameters:
    dataframe (pandas dataframe): the input dataframe
    ax (matplotlib axes object): the subplot axes object to plot the chart on

    Returns:
    None
    """
    
    # Calculate the percentage of each education level in the dataframe
    edu_pct = dataframe['education'].value_counts(normalize=True)*100

    # Create a pie chart to visualize the education distribution
    patches, texts, autotexts = ax.pie(x=edu_pct, autopct='%1.1f%%', \
                               explode=[0.1,0,0,0], startangle=90, \
                                   pctdistance=0.7,
        colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99'])
    
    # Set the font size and color of the text labels
    for autotext in autotexts:
        autotext.set_color('#41393E')
        autotext.set_fontsize(12)
        autotext.set_weight('bold')
    
    # Create a legend for the education types
    ax.legend(edu_pct.index, loc='upper left', bbox_to_anchor=(-0.3, 1),\
              title='Education Levels')

    # Set the title of the chart
    ax.set_title("Distribution of\neducation level", fontsize=20, \
                 weight='bold', y=0.95)    
    
    # Adjust the chart layout and display it
    plt.tight_layout()
    
    return

def gs_plot_subscription_by_education(edu_df, ax):
    """
    Creates a nested pie chart to show the subscription status of clients
    grouped by education level.

    Parameters:
    edu_df (pandas.DataFrame): The dataframe containing education and 
    subscription status information.
    ax (matplotlib.axes.Axes): The subplot axes object to plot the chart on.

    Returns:
    None
    """
        
    # Group the data by education and subscription status, and 
    # count the number of clients in each group
    edu_sub_count = edu_df.groupby(['education', 'y']).size()\
                                    .reset_index(name='count')

    # Pivot the data to have subscription status as columns
    edu_sub_pivot = edu_sub_count.pivot(index='education', \
                                        columns='y', values='count')

    # Get the unique education levels
    education_levels = edu_sub_count['education'].unique()

    # Initialize lists to store the sizes and labels for the 
    # outer and inner pie charts
    outer_sizes = []
    inner_sizes = []
    outer_labels = []

    # Calculate the sizes and labels for the outer and inner pie charts
    for edu in education_levels:
        subscribed_count = edu_sub_pivot.loc[edu, 'yes']
        not_subscribed_count = edu_sub_pivot.loc[edu, 'no']
        total_count = subscribed_count + not_subscribed_count

        outer_sizes.append(total_count)
        inner_sizes.extend([not_subscribed_count, subscribed_count])
        outer_labels.append(edu)

    # Calculate the inner explode values
    inner_explode = [0.05] * 2 * len(education_levels)

    # Create the outer and inner pie charts
    outer_colors = ['#99ff99', '#ff9999', '#66b3ff', '#ffcc99']\
                    [:len(education_levels)]
    inner_colors = ['#ff6666', '#ffcc66'] * len(education_levels)

    wedges_outer, _ = ax.pie(outer_sizes, labels=None, startangle=90,\
                             colors=outer_colors, pctdistance=0.89, \
                             wedgeprops=dict(width=0.5, edgecolor='grey'))
    wedges_inner, _, autotexts_inner = ax.pie(inner_sizes, labels=None,\
                              autopct='%1.1f%%', startangle=90, \
                              colors=inner_colors, pctdistance=0.7,\
                              radius=0.75, explode=inner_explode, \
                              wedgeprops=dict(width=0.5, edgecolor='grey'))

    # Set the text color for the inner pie chart labels
    for text in autotexts_inner:
        text.set_color('black')
        text.set_fontsize(10)
        text.set_weight('bold')

    # Add a legend describing the yes no color
    handles = [plt.Rectangle((0,0),1,1, color='#ff6666'), \
               plt.Rectangle((0,0),1,1, color='#ffcc66')]
    labels = ['No', 'Yes']
    ax.legend(handles, labels, title='Subscription\n    Status'\
              , loc=[-0.1, 0.75])

    # Set the rotation of the inner autopct labels
    plt.setp(autotexts_inner, rotation=90)

    # Add a title to the plot
    ax.set_title('Subscription Status by\nEducation Level',\
                 fontsize=20, weight='bold', y=0.95)
        
    return

def gs_create_housing_pie_chart(house_df, ax):
    """
    Create a 3D pie chart showing the percentage of clients with 
    or without a housing loan.

    Parameters:
    bank_df (pandas.DataFrame): DataFrame containing bank marketing 
    campaign data.
    ax (matplotlib.axes.Axes): Axes object to plot the chart on.

    Returns:
    None
    """
    
    # Count the number of occurrences of each value in the 'housing' column
    value_counts = house_df['housing'].value_counts()

    # Calculate the percentage of occurrences for each unique value
    # in the 'housing' column
    housing_pct = value_counts * 100 / len(house_df)
    housing_pct = housing_pct[::-1]

    # Create the 3D pie chart with the specified properties
    patches, _, autotexts = ax.pie(x=housing_pct, autopct='%1.1f%%',\
                                   startangle=90, labels=None,
                                   pctdistance=0.5, 
                                   colors=['#D79B70', '#F2D6C0'],
                                   wedgeprops={'linewidth': 1,\
                                               'edgecolor': 'black', 
                                               'alpha': 0.7},
                                   explode=[0.1, 0], shadow=True)

    # Set properties for autopct labels
    for autotext in autotexts:
        autotext.set_fontsize(12)
        autotext.set_weight('bold')

    # Set the aspect ratio to be equal so that pie is drawn as a circle
    ax.axis('equal')

    # Create legends for each pie slice
    handles = [plt.Rectangle((0, 0), 1, 1, color='#D79B70'),\
               plt.Rectangle((0, 0), 1, 1, color='#F2D6C0')]
    labels = housing_pct.index
    ax.legend(handles, labels, title='Housing Loan', \
              loc='upper left', bbox_to_anchor=(-0.1, 1))

    # Display the 3D pie chart
    ax.set_title('Pie Chart of\nHousing Loan', fontsize=20, \
                 weight='bold', y=0.95)
    
    return


def gs_housing_subscription_pie_chart(house_df, ax):
    """
    This function takes a pandas dataframe containing bank 
    marketing data as input, and creates a pie chart
    that shows the distribution of subscription status 
    (yes/no) by housing status (yes/no). The chart has an
    outer ring that shows the total number of clients in 
    each housing category, and an inner ring that shows the
    distribution of subscription status for each housing category.

    Parameters:
        dataframe (pandas.DataFrame): A pandas dataframe 
        containing bank marketing data.
        ax (matplotlib.axes.Axes): The subplot axes on 
        which to plot the pie chart.

    Returns:
        None
    """
    
    # Group the data by housing and subscription status, 
    # and count the number of clients in each group
    housing_sub_count = house_df.groupby(['housing', 'y']).\
                                size().reset_index(name='count')
    
    # Pivot the data to have subscription status as columns
    housing_sub_pivot = housing_sub_count.pivot(index='housing',\
                                        columns='y', values='count')
    
    # Get the unique housing categories
    housing_categories = housing_sub_count['housing'].unique()
    
    # Initialize lists to store the sizes and labels for the outer and
    # inner pie charts
    outer_sizes = []
    inner_sizes = []
    outer_labels = []
    
    # Calculate the sizes and labels for the outer and inner pie charts
    for housing in housing_categories:
        subscribed_count = housing_sub_pivot.loc[housing, 'yes']
        not_subscribed_count = housing_sub_pivot.loc[housing, 'no']
        total_count = subscribed_count + not_subscribed_count
        
        outer_sizes.append(total_count)
        inner_sizes.extend([not_subscribed_count, subscribed_count])
        outer_labels.append(housing)
    
    # Calculate the inner explode values
    inner_explode = [0.05] * 2 * len(housing_categories)
    
    # Create the outer pie chart
    outer_colors = ['#D79B70', '#F2D6C0'] * len(housing_categories)
    
    wedges_outer, _= ax.pie(outer_sizes, labels=None, startangle=90,
                    colors=outer_colors, pctdistance=0.89,
                    wedgeprops={'linewidth': 1, 'edgecolor': 'black',\
                                'alpha': 0.7, 'antialiased': True,\
                                'linestyle': 'solid'},
                    explode=[0.1, 0], shadow=True)
    
    # Set the properties of the outer pie chart labels
#     plt.setp(autotexts_outer, size=10, rotation=90, color='black',\
#                                                     weight='bold')
#     plt.setp(texts_outer, size=12, weight='bold')
    
    # Create the inner pie charts
    inner_pie_radius = 0.35
    inner_colors = ['#D79B70', '#F2D6C0']
    
    # Calculate the center positions for the inner pie charts
    centers_x = [-0.5, 0.5]
    centers_y = [0, 0]
    
    for i, (housing, center_x, center_y) in enumerate(zip(housing_categories,\
                                                          centers_x, 
                                                          centers_y)):
        inner_wedges, _, autotexts_inner = \
        ax.pie(inner_sizes[i * 2: (i + 1) * 2],labels=None, autopct='%1.1f%%', 
               startangle=90, colors=inner_colors, pctdistance=0.7, 
               radius=inner_pie_radius, 
               explode=inner_explode[i * 2: (i + 1) * 2], 
               wedgeprops=dict(width=0.5, edgecolor='grey'), 
               center=(center_x, center_y))

        # Set the properties of the inner autopct labels
        for autotext in autotexts_inner:
            autotext.set_color('black')
            autotext.set_fontsize(10)

    # Add a legend describing the subscription status colors
    handles_inner = [plt.Rectangle((0, 0), 1, 1, color='#D79B70'), \
                     plt.Rectangle((0, 0), 1, 1, color='#F2D6C0')]
    labels_inner = ['No', 'Yes']
    plt.legend(handles_inner, labels_inner, loc=[0.85, 0.8],\
               title='Subsription Status')

    # Set the aspect ratio to be equal so that pie is drawn as a circle
    ax.axis('equal')
    
    # Add a title to the plot
    ax.set_title('Subscription Status\nby Housing Loan',\
                 fontsize=20, weight='bold', y=0.95)

    # Adjust the chart layout
    plt.tight_layout()
    
    return

def gs_analyze_age(age_df, ax1, ax2):
    """
    This function takes a Pandas DataFrame containing bank customer
    information, and two matplotlib axes objects. It plots the age
    distribution of the customers in the DataFrame, including a 
    boxplot and histogram with KDE. It also calculates and displays
    the IQR, median, skewness, kurtosis, and mean of the 'age' column.
    
    Parameters:
    age_df (pandas.DataFrame): The DataFrame containing bank 
    customer information.
    ax1 (matplotlib.axes.Axes): The first subplot for the boxplot.
    ax2 (matplotlib.axes.Axes): The second subplot for the histogram.
    
    Returns:
    None
    """

    # Calculate the interquartile range (IQR) of the 'age' column
    iqr = np.percentile(age_df['age'], 75) -\
                                np.percentile(age_df['age'], 25)
    
    # Calculate the skewness and kurtosis of the 'age' column
    skewness = age_df['age'].skew()
    kurtosis = age_df['age'].kurtosis()

    # Plot the boxplot of 'age' with optional whisker length
    # (whis) of 1.5
    sns.boxplot(data=age_df, x='age', whis=1.5, color='#3182bd',\
                ax=ax1)
    ax1.set_xlabel('')
    ax1.set_xticklabels([])
    
     # Add text box to box plot
    iqr_text = f'IQR: {int(iqr)}\nMedian: {int(bank_df["age"].median())}'
    ax1.text(0.75, 0.85, iqr_text, transform=ax1.transAxes,
                 verticalalignment='top', bbox=dict(boxstyle='round'\
                        , facecolor='wheat', alpha=0.5), fontsize=12)
      
    # Plot the histogram of 'age' with KDE (Kernel Density Estimation)
    sns.histplot(data=age_df, x='age', kde=True, color='#3182bd', ax=ax2)
    ax2.set_ylabel('Count', fontsize=14, weight='bold')
    
     # Add text box to histogram plot
    stats_text = f'Skewness: {skewness:.2f}\nKurtosis: {kurtosis:.2f}' +\
                                f'\nMean: {int(bank_df["age"].mean())}'
    ax2.text(0.75, 0.75, stats_text, transform=ax2.transAxes,
                 verticalalignment='top', bbox=dict(boxstyle='round', \
                            facecolor='wheat', alpha=0.5), fontsize=12)

    ax2.set_xlabel('Age', fontsize=14, weight='bold')
    
    # Add a title to the plot
    ax1.set_title('Age Distribution', fontsize=20, weight='bold', y=1)
    
    # Adjust the spacing between the subplots
    plt.tight_layout()
    
    return

def gs_plot_age_distribution_by_subscription(age_df, ax0, ax1):
    """
    This function takes a dataframe and two axes as input and 
    creates a boxplot and histogram to compare the age 
    distribution of customers who have subscribed to a term 
    deposit and those who have not subscribed.

    Args:
    - age_df: Pandas DataFrame containing the bank 
    marketing data
    - ax0: Axes object for the boxplot
    - ax1: Axes object for the histogram

    Returns:
    - None (displays a visualization)
    """

    # Create separate dataframes for subscribed and 
    # non-subscribed customers
    subscribed_df = age_df[age_df['y'] == 'yes']
    not_subscribed_df = age_df[age_df['y'] == 'no']

    # Plot the boxplot of 'age' for non-subscribed customers
    not_subscribed_boxplot = sns.boxplot(data=not_subscribed_df, 
                                         x='age', whis=1.5, 
                                         color='#ff0000', ax=ax0,
                                         fliersize=5, linewidth=1.5, 
                                         saturation=0.75, 
                                         boxprops=dict(alpha=.7), 
                                         medianprops=dict(alpha=.7), 
                                         capprops=dict(alpha=.7), 
                                         whiskerprops=dict(alpha=.7), 
                                         flierprops=dict(marker='d', \
                                            markerfacecolor='#ff0000', \
                                                markersize=5, alpha=.7))

    # Plot the boxplot of 'age' for subscribed customers
    subscribed_boxplot = sns.boxplot(data=subscribed_df, x='age', 
                                     whis=1.5, color='#ffa500', 
                                     ax=ax0, fliersize=5, 
                                     linewidth=1.5, saturation=0.75, 
                                     boxprops=dict(alpha=.7), 
                                     medianprops=dict(alpha=.7), 
                                     capprops=dict(alpha=.7), 
                                     whiskerprops=dict(alpha=.7), 
                                     flierprops=dict(marker='o', 
                                                     markerfacecolor='#ffa500', 
                                                     markersize=5, alpha=.7))

    ax0.set_xlabel('')
    ax0.set_xticklabels([])

    # Create a legend for the boxplot
    not_subscribed_patch = mpatches.Patch(color='#ff0000', 
                                          label='Not Subscribed')
    subscribed_patch = mpatches.Patch(color='#ffa500', 
                                      label='Subscribed')
    ax0.legend(handles=[not_subscribed_patch, subscribed_patch],\
                                               loc='upper right')

    # Plot the histogram of 'age' for non-subscribed customers
    sns.histplot(data=not_subscribed_df, x='age', kde=True, \
                 color='#ff0000', ax=ax1, alpha=0.7, \
                 label='Not Subscribed')

    # Plot the histogram of 'age' for subscribed customers
    sns.histplot(data=subscribed_df, x='age', kde=True,
                 color='#ffa500', ax=ax1, alpha=0.7, 
                 label='Subscribed')
    ax1.set_ylabel('')
    ax1.set_yticklabels([])
    
    ax0.set_title('Age Distribution by Subscription Status', \
                  fontsize=20, weight='bold', y=1)

    ax1.set_xlabel('Age', fontsize=14, weight='bold')

    # Add a legend to the plots
    ax1.legend()

    # Add text box to histogram plot
    not_subscribed_stats_text = f'Not Subscribed\nSkewness: ' +\
    f'{not_subscribed_df["age"].skew():.2f}\nKurtosis: '+\
    f'{not_subscribed_df["age"].kurtosis():.2f}\nMean: '+\
    f'{not_subscribed_df["age"].mean():.0f}'
    subscribed_stats_text = f'Subscribed\nSkewness: '+\
    f'{subscribed_df["age"].skew():.2f}\nKurtosis: '+\
    f'{subscribed_df["age"].kurtosis():.2f}\nMean: '+\
    f'{subscribed_df["age"].mean():.0f}'

    ax1.text(0.65, 0.55, not_subscribed_stats_text, 
             transform=ax1.transAxes,
             verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='wheat',
                       alpha=0.5), fontsize=12)

    ax1.text(0.83, 0.55, subscribed_stats_text, 
             transform=ax1.transAxes,
             verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='wheat',
                       alpha=0.5), fontsize=12)

    # Add text box to box plot
    not_subscribed_iqr = int(not_subscribed_df["age"].quantile(0.75) - \
                             not_subscribed_df["age"].quantile(0.25))
    not_subscribed_median = int(not_subscribed_df["age"].median())
    subscribed_iqr = int(subscribed_df["age"].quantile(0.75) - \
                         subscribed_df["age"].quantile(0.25))
    subscribed_median = int(subscribed_df["age"].median())

    not_subscribed_iqr_text = f'Not Subscribed\nIQR: {not_subscribed_iqr}'+\
    f'\nMedian: {not_subscribed_median}'
    subscribed_iqr_text = f'Subscribed \nIQR: {subscribed_iqr}\nMedian: '+\
        f'{subscribed_median}'
    
    not_subscribed_boxplot.text(0.78, 0.36, not_subscribed_iqr_text, 
                                transform=ax0.transAxes,
                                verticalalignment='top', 
                                bbox=dict(boxstyle='round', 
                                          facecolor='wheat', 
                                          alpha=0.5), fontsize=12)

    subscribed_boxplot.text(0.5, 0.36, subscribed_iqr_text, 
                            transform=ax0.transAxes,
                            verticalalignment='top', 
                            bbox=dict(boxstyle='round', 
                                      facecolor='wheat', 
                                      alpha=0.5), fontsize=12)

    # Adjust the spacing between the subplots
    plt.tight_layout()
    
    return

def gs_analyze_balance(df, ax0, ax1):
    """
    Perform a descriptive analysis of the 'balance' 
    column in a given DataFrame.

    Parameters:
    df (pandas.DataFrame): DataFrame to analyze.

    Returns:
    None.
    """

    # Plot the boxplot of the 'balance' column with 
    # whiskers extending to 1.5 times the interquartile range
    sns.boxplot(data=df, x='balance', whis=1.5, color='#b5838d',\
                ax=ax0)
    median = df['balance'].median()
    q1, q3 = np.percentile(df['balance'], [25, 75])
    iqr = q3 - q1
    
    # Define the lower and upper thresholds for outliers
    bal_outlier_low_thrs = q1 - (1.5 * iqr)
    bal_outlier_high_thrs = q3 + (1.5 * iqr)

    # Find outliers based on the thresholds
    bal_outliers_df = df[(df.balance < bal_outlier_low_thrs) | \
                         (df.balance > bal_outlier_high_thrs)]

    # Add text to the plot
    textstr = f"Median = {int(median)}\nIQR = {iqr:.0f}\n"+\
    f"Outliers # = {bal_outliers_df.shape[0]}"
    ax0.text(0.75, 0.39, textstr, transform=ax0.transAxes, fontsize=12,
        verticalalignment='top', bbox=dict(facecolor='wheat', alpha=0.5))
    
    ax0.set_xlabel('')
    ax0.set_xticklabels([])

    # Plot the histogram of the 'balance' column with a kernel 
    # density estimate
    sns.histplot(data=df, x='balance', kde=True, color='#b5838d', \
                 ax=ax1)
    ax1.set_ylabel('Count', fontsize=14, weight='bold')

    # Add text to the plot
    mean = df['balance'].mean()
    skewness = df['balance'].skew()
    kurtosis = df['balance'].kurtosis()
    textstr = f"Mean = {int(mean)}\nSkewness = {skewness:.2f}\n"+\
    f"Kurtosis = {kurtosis:.2f}"
    ax1.text(0.75, 0.75, textstr, transform=ax1.transAxes, fontsize=12,
        verticalalignment='top', bbox=dict(facecolor='wheat', alpha=0.5))
    
    # Set the title
    ax0.set_title('Balance Distribution', fontsize=20, weight='bold', y=1)
    
    # set the x lable
    ax1.set_xlabel('Balance', fontsize=14, weight='bold')    
    
    # Adjust the spacing between the subplots
    plt.tight_layout()
    
    return

def gs_plot_balance_distribution_by_subscription(balance_df, ax0, ax1):
    """
    This function takes a dataframe and two axes as input and 
    creates a boxplot and histogram to compare the balance 
    distribution of customers who have subscribed to a term 
    deposit and those who have not subscribed.

    Args:
    - balance_df: Pandas DataFrame containing the bank 
    marketing data
    - ax0: Axes object for the boxplot
    - ax1: Axes object for the histogram

    Returns:
    - None (displays a visualization)
    """

    # Create separate dataframes for subscribed and non-
    # subscribed customers
    subscribed_df = balance_df[balance_df['y'] == 'yes']
    not_subscribed_df = balance_df[balance_df['y'] == 'no']

    # Plot the boxplot of 'age' for non-subscribed customers
    not_subscribed_boxplot = sns.boxplot(data=not_subscribed_df, 
                                         x='balance', whis=1.5, 
                                         color='grey', ax=ax0,
                                         fliersize=5, linewidth=1.5, 
                                         saturation=0.75, 
                                         boxprops=dict(alpha=.7), 
                                         medianprops=dict(alpha=.7), 
                                         capprops=dict(alpha=.7), 
                                         whiskerprops=dict(alpha=.7), 
                                         flierprops=dict(marker='d', \
                                                     markerfacecolor='grey', \
                                                     markersize=5, alpha=.7))

    # Plot the boxplot of 'age' for subscribed customers
    subscribed_boxplot = sns.boxplot(data=subscribed_df, 
                                     x='balance', whis=1.5, 
                                     color='brown', ax=ax0, 
                                     fliersize=5, linewidth=1.5, 
                                     saturation=0.75, 
                                     boxprops=dict(alpha=.7), 
                                     medianprops=dict(alpha=.7), 
                                     capprops=dict(alpha=.7), 
                                     whiskerprops=dict(alpha=.7), 
                                     flierprops=dict(marker='o', 
                                                     markerfacecolor='brown', 
                                                     markersize=5, alpha=.7))

    # remove x labels and x ticks labels
    ax0.set_xlabel('')
    ax0.set_xticklabels([])

    # Create a legend for the boxplot
    not_subscribed_patch = mpatches.Patch(color='grey', 
                                          label='Not Subscribed')
    subscribed_patch = mpatches.Patch(color='brown', 
                                      label='Subscribed')
    ax0.legend(handles=[not_subscribed_patch, subscribed_patch], 
               loc='upper right')

    # Plot the histogram of 'age' for non-subscribed customers
    sns.histplot(data=not_subscribed_df, x='balance', kde=True, 
                 color='grey', ax=ax1, alpha=0.7, 
                 label='Not Subscribed')

    # Plot the histogram of 'age' for subscribed customers
    sns.histplot(data=subscribed_df, x='balance', kde=True, 
                 color='brown', ax=ax1, alpha=0.7, 
                 label='Subscribed')
    
    # remove y labels and y ticks labels
    ax1.set_ylabel('')
    ax1.set_yticklabels([])

    # Set plot title
    ax0.set_title('Balance Distribution by Subscription Status', \
                  fontsize=20, weight='bold', y=1)

    # Set xlabel
    ax1.set_xlabel('Balance', fontsize=14, weight='bold')

    # Add a legend to the plots
    ax1.legend()

    # Add text box to histogram plot
    not_subscribed_stats_text = f'Not Subscribed\nSkewness: '+\
    f'{not_subscribed_df["balance"].skew():.2f}\nKurtosis: '+\
    f'{not_subscribed_df["balance"].kurtosis():.2f}\nMean: '+\
    f'{not_subscribed_df["balance"].mean():.0f}'
    subscribed_stats_text = f'Subscribed\nSkewness: '+\
    f'{subscribed_df["balance"].skew():.2f}\nKurtosis: '+\
    f'{subscribed_df["balance"].kurtosis():.2f}\nMean: '+\
    f'{subscribed_df["balance"].mean():.0f}'

    ax1.text(0.4, 0.6, not_subscribed_stats_text, 
             transform=ax1.transAxes,
             verticalalignment='top', 
             bbox=dict(boxstyle='round', 
                       facecolor='wheat', 
                       alpha=0.5), fontsize=12)

    ax1.text(0.6, 0.6, subscribed_stats_text, 
             transform=ax1.transAxes,
             verticalalignment='top', 
             bbox=dict(boxstyle='round', 
                       facecolor='wheat', 
                       alpha=0.5), fontsize=12)

    # Add text box to box plot
    not_subscribed_iqr = int(not_subscribed_df["balance"].quantile(0.75) \
                             - not_subscribed_df["balance"].quantile(0.25))
    not_subscribed_median = int(not_subscribed_df["balance"].median())
    not_subscribed_outliers = \
    len(not_subscribed_df[(not_subscribed_df["balance"] < \
                           not_subscribed_df["balance"].quantile(0.25) - \
                           1.5*(not_subscribed_df["balance"].quantile(0.75) \
                            - not_subscribed_df["balance"].quantile(0.25))) \
                          | (not_subscribed_df["balance"] > \
                             not_subscribed_df["balance"].quantile(0.75) +\
                             1.5*(not_subscribed_df["balance"].quantile(0.75) \
                              - not_subscribed_df["balance"].quantile(0.25)))])
    
    not_subscribed_iqr_text = f'Not Subscribed\nIQR: {not_subscribed_iqr}\
    , Median: {not_subscribed_median}\nOutliers #: {not_subscribed_outliers}'

    subscribed_iqr = int(subscribed_df["balance"].quantile(0.75) \
                         - subscribed_df["balance"].quantile(0.25))
    subscribed_median = int(subscribed_df["balance"].median())
    subscribed_outliers = \
    len(subscribed_df[(subscribed_df["balance"] < \
                       subscribed_df["balance"].quantile(0.25) - \
                       1.5*(subscribed_df["balance"].quantile(0.75) - \
                            subscribed_df["balance"].quantile(0.25)))\
                      | (subscribed_df["balance"] > \
                         subscribed_df["balance"].quantile(0.75) + \
                         1.5*(subscribed_df["balance"].quantile(0.75) \
                              - subscribed_df["balance"].quantile(0.25)))])
    
    subscribed_iqr_text = f'Subscribed\nIQR: {subscribed_iqr}, \
    Median: {subscribed_median}\nOutliers #: {subscribed_outliers}'

    not_subscribed_boxplot.text(0.35, 0.38, not_subscribed_iqr_text, 
                                transform=ax0.transAxes,
                                verticalalignment='top', 
                                bbox=dict(boxstyle='round', 
                                          facecolor='wheat', 
                                          alpha=0.5), fontsize=12)

    subscribed_boxplot.text(0.63, 0.38, subscribed_iqr_text, 
                            transform=ax0.transAxes,
                            verticalalignment='top', 
                            bbox=dict(boxstyle='round', 
                                      facecolor='wheat', 
                                      alpha=0.5), fontsize=12)

    # Adjust the spacing between the subplots
    plt.tight_layout()
    
    return

def gs_plot_heatmaps(duration_df, ax1, ax2):
    """
    Plots two heatmaps side-by-side for subscribed customers 
    of a bank. The first heatmap shows job type vs balance vs age
    (in decades), and the second heatmap shows job type vs 
    duration vs age (in decades).

    Args:
        duration_df (pd.DataFrame): A pandas DataFrame 
        containing information about customers of a bank.
        ax1 (matplotlib.axes.Axes): The subplot axes for the first heatmap.
        ax2 (matplotlib.axes.Axes): The subplot axes for the second heatmap.

    Returns:
        None
    """
    
    # Filter the subscribed customers
    subscribed_customers = duration_df[duration_df['y'] == 'yes']
    
    # Convert age to decades
    subscribed_customers['age_decades'] = \
    (subscribed_customers['age'] // 10) * 10
    
    # Create a pivot table of job type vs balance vs age_decades
    pivot_table1 = subscribed_customers.pivot_table(values='balance',
                                                    index='age_decades', 
                                                    columns='job')
    
    # Create a pivot table of job type vs duration vs age_decades
    pivot_table2 = subscribed_customers.pivot_table(values='duration', 
                                                    index='age_decades', 
                                                    columns='job')
    
    # Plot the first heatmap
    sns.heatmap(pivot_table1, cmap='viridis', 
                cbar_kws={'label': 'Balance'}, ax=ax1)
    ax1.set_xlabel('Job Type', fontsize=12, weight='bold')
    ax1.set_ylabel('Age (in Decades)', fontsize=12, weight='bold')
    ax1.tick_params(axis='x', rotation=90, labelsize=10)
    ax1.tick_params(axis='y', labelsize=10)
    # Increase font size of color bar labels
    ax1.collections[0].colorbar.ax.tick_params(labelsize=10)  
    # Update color bar label
    ax1.collections[0].colorbar.ax.set_ylabel('Balance', 
                                              fontsize=12, 
                                              weight='bold')  
    
    # Plot the second heatmap
    sns.heatmap(pivot_table2, cmap='viridis', cbar_kws=\
                {'label': 'Duration (seconds)'}, ax=ax2)
    ax2.set_xlabel('Job Type', fontsize=12, weight='bold')
    ax2.set_ylabel('Age (in Decades)', fontsize=12, weight='bold')
    ax2.tick_params(axis='x', rotation=90, labelsize=10)
    ax2.tick_params(axis='y', labelsize=10)
    # Increase font size of color bar labels
    ax2.collections[0].colorbar.ax.tick_params(labelsize=10) 
    # Update color bar label
    ax2.collections[0].colorbar.ax.set_ylabel('Duration (seconds)', \
                                              fontsize=12, weight='bold') 
    
    # Add a single title for both heat maps
    ax1.set_title('Job Type vs Balance/Call Duration vs Age (in Decades) \
for Subscribed Customers', fontsize=20, weight='bold', x=1.25, y=1.05)

    # Adjust the spacing between the subplots
    plt.tight_layout()
    
    return

""" Main Program """

# Calling the function load_bank_data and store the return df in bank_df
bank_df = load_bank_data('train.csv')

# Calling the function categorize_features and store the return features 
# in bank_df
numerical_features, categorical_features = categorize_features(bank_df)

# Print the numerical features of the dataframe
print('Numerical Features: \n', numerical_features)
print()
# Print the categorical features of the dataframe
print('Categorical Features: \n', categorical_features)

# Print the number of unique job types
print("Number of unique job types:", bank_df['job'].nunique())

# Print the unique job types
print("\n".join(bank_df['job'].unique()))

# Print the number of null values in the 'job' column
print("Number of null values in 'job' column:", bank_df['job'].isnull().sum())

# Print the count of each job type
print("\nCount of each job type:")
print(bank_df['job'].value_counts())

# Print the percentage of each job type
print("\nPercentage of each job type:")
print(bank_df['job'].value_counts(normalize=True)*100)

# Print the number of unique education levels
print("Number of unique education levels:", bank_df['education'].nunique())

# Print the unique education levels
print("\n".join(bank_df['education'].unique()))

# Print the number of null values in the 'education' column
print()
print("Number of null values in 'education' column:", bank_df['education']\
      .isnull().sum())

# Print the count of each education level
print("\nCount of each education level:")
print(bank_df['education'].value_counts())

# Print the percentage of each education level
print("\nPercentage of each education level:")
edu_pct = bank_df['education'].value_counts(normalize=True)*100
print(edu_pct)

# Count the number of unique values in the 'housing' column
num_unique = bank_df['housing'].nunique()

# Print out the unique values in the 'housing' column
print("\n".join(bank_df['housing'].unique()))

# Count the number of missing values in the 'housing' column
num_missing = bank_df['housing'].isnull().sum()

# Count the number of occurrences of each value in the 'housing' column
value_counts = bank_df['housing'].value_counts()

# Calculate the percentage of occurrences for each unique value in the 
# 'housing' column
housing_pct = value_counts * 100 / len(bank_df)
print(housing_pct)
housing_pct = housing_pct[::-1]
print(housing_pct)

# Compute descriptive statistics of the 'age' column
age_stats = bank_df['age'].describe()
print(age_stats)

# Calculate percentiles of the 'age' column
percentile_10 = np.percentile(bank_df['age'], 10)
percentile_50 = np.percentile(bank_df['age'], 50)
percentile_90 = np.percentile(bank_df['age'], 90)
print('10th percentile: ', percentile_10)
print('50th percentile: ', percentile_50)
print('90th percentile: ', percentile_90)

# Calculate the interquartile range (IQR) of the 'age' column
iqr = np.percentile(bank_df['age'], 75) - np.percentile(bank_df['age'], 25)
print('iqr: ', iqr)

# Calculate the span of ages (difference between the maximum and minimum 
# values)
age_span = bank_df['age'].max() - bank_df['age'].min()
print('Age Span: ', age_span)

# Calculate the skewness and kurtosis of the 'age' column
skewness = bank_df['age'].skew()
kurtosis = bank_df['age'].kurtosis()
print('skew: ', skewness)
print('kurtosis: ', kurtosis)

# Calculate the interquartile range (IQR) of the 'age' column
iqr = np.percentile(bank_df['age'], 75) - np.percentile(bank_df['age'], 25)

# Calculate the lower and upper thresholds for outliers
outlier_low_thrs = np.percentile(bank_df['age'], 25) - (1.5 * iqr)
outlier_high_thrs = np.percentile(bank_df['age'], 75) + (1.5 * iqr)

# Select the rows where age is below the lower threshold or above the 
# upper threshold
age_outliers_df = bank_df[(bank_df['age'] < outlier_low_thrs) | \
                          (bank_df['age'] > outlier_high_thrs)]

# Compute descriptive statistics of the age outliers
age_outliers_stats = age_outliers_df['age'].describe()

# Calculate and print descriptive statistics
print("Descriptive Statistics:")
print(bank_df['balance'].describe())

# Calculate and print selected percentiles
print("\nSelected Percentiles:")
print("10th percentile: ", np.percentile(bank_df['balance'],10))
print("50th percentile: ", np.percentile(bank_df['balance'],50))
print("90th percentile: ", np.percentile(bank_df['balance'],90))
print("Interquartile Range (IQR): ", np.percentile(bank_df['balance'],75) - \
      np.percentile(bank_df['balance'],25))
print("Range: ", bank_df['balance'].max() - bank_df['balance'].min())
print("Skewness: ", bank_df['balance'].skew())
print("Kurtosis: ", bank_df['balance'].kurtosis())

# Calculate the interquartile range (IQR)
q1, q3 = np.percentile(bank_df['balance'], [25, 75])
iqr = q3 - q1

# Define the lower and upper thresholds for outliers
bal_outlier_low_thrs = q1 - (1.5 * iqr)
bal_outlier_high_thrs = q3 + (1.5 * iqr)

# Find outliers based on the thresholds
bal_outliers_df = bank_df[(bank_df.balance < bal_outlier_low_thrs) | \
                          (bank_df.balance > bal_outlier_high_thrs)]

# Print the outliers
print("Balance Outliers:")
print(bal_outliers_df.head(n=3))
print("\nOutlier Statistics:")
print(bal_outliers_df['balance'].describe())

# Create a GridSpec with 10 rows and 4 columns
fig = plt.figure(figsize=(20, 32))
gs = fig.add_gridspec(34, 12, hspace=0, wspace=0, 
                      height_ratios=[0.1, 1.6, 0.1, 0.4, 1, 1, \
                                     0.5, 0.4, 0.1, 0.4, 1, 1, 0.7,\
                                     0.1, 0.4, 1, 1, 0.4, 0.1, 0.4, \
                                     1, 1, 0.4, 0.1, 0.6, 1, 1, 0.7, \
                                     0.9, 0.1, 0.05, 1, 0.05, 0.1]
                     , width_ratios=[0.05, 0.4, 1, 1, 0.1, 0.025, \
                                     0.025, 0.1, 1, 1, 0.4, 0.05])


# Define the GridSpec locations for the title and description
title_ax = fig.add_subplot(gs[1, 2:10])
description_ax = fig.add_subplot(gs[31, 2:10])

sns.set_style('whitegrid')
sns.set_theme(style='white', rc={"grid.linewidth": 0.5, 
                                 "axes.grid": True, 
                                 "grid.color": "grey"})

# Define the GridSpec locations for each plot
plot1_ax = fig.add_subplot(gs[4:7, 2:6])
plot2_ax = fig.add_subplot(gs[4:7, 6:10:])

# Calling the job subscription function
gs_visualize_job_and_subscription(bank_df, plot1_ax, plot2_ax)

plot3_ax = fig.add_subplot(gs[10:13, 2])

# Calling the education distribution function
gs_visualize_education_distribution(bank_df, plot3_ax)

plot4_ax = fig.add_subplot(gs[10:13, 3])

# Calling the education subscription function
gs_plot_subscription_by_education(bank_df, plot4_ax)

plot5_ax = fig.add_subplot(gs[10:13, 8])

# Calling the housing distribution function
gs_create_housing_pie_chart(bank_df, plot5_ax)

plot6_ax = fig.add_subplot(gs[10:13, 9])

# Calling the housing subscription function
gs_housing_subscription_pie_chart(bank_df, plot6_ax)

plot7_ax = fig.add_subplot(gs[15, 2:6])
plot8_ax = fig.add_subplot(gs[16, 2:6])

# Calling the age distribution function
gs_analyze_age(bank_df, plot7_ax, plot8_ax)

plot9_ax = fig.add_subplot(gs[15, 6:10])
plot10_ax = fig.add_subplot(gs[16, 6:10])

# Calling the age subscription function
gs_plot_age_distribution_by_subscription(bank_df, plot9_ax, plot10_ax)

plot11_ax = fig.add_subplot(gs[20, 2:6])
plot12_ax = fig.add_subplot(gs[21, 2:6])

# Calling the balance distribution function
gs_analyze_balance(bank_df, plot11_ax, plot12_ax)

plot13_ax = fig.add_subplot(gs[20, 6:10])
plot14_ax = fig.add_subplot(gs[21, 6:10])

# Calling the balance subscription function
gs_plot_balance_distribution_by_subscription(bank_df, plot13_ax, plot14_ax)

plot13_ax = fig.add_subplot(gs[25:28, 2:4])
plot14_ax = fig.add_subplot(gs[25:28, 8:10])

# Calling the heatmap of call duration and balance function
gs_plot_heatmaps(bank_df, plot13_ax, plot14_ax)

# Defining the border line plots 
hborder_1 = fig.add_subplot(gs[0, :])
hborder_2 = fig.add_subplot(gs[2, :])
hborder_3 = fig.add_subplot(gs[8, :])
hborder_4 = fig.add_subplot(gs[13, :])
hborder_5 = fig.add_subplot(gs[18, :])
hborder_6 = fig.add_subplot(gs[23, :])
hborder_7 = fig.add_subplot(gs[29, :])
hborder_8 = fig.add_subplot(gs[33, :])

vborder_1 = fig.add_subplot(gs[1:33, 0])
vborder_2 = fig.add_subplot(gs[1:33, 11])
vborder_3 = fig.add_subplot(gs[9:13, 5:7])

# Set light grey color
light_grey = (0.9, 0.9, 0.9)

# Hide labels and edge lines for borders
for ax in [hborder_1, hborder_2, hborder_3, hborder_4, hborder_5, \
           hborder_6, hborder_7, hborder_8, vborder_1, vborder_2, vborder_3]:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Fill with light grey background color
    ax.set_facecolor(light_grey)


# Hide tick labels for title and description axes
title_ax.axis('off')
description_ax.axis('off')

# Set the title and description text
title_ax.set_title('Customer Analytics for Bank\nCreated by: \
Muhammad Ameer Hamza (ID# 22034204)', \
                   fontsize=26, fontweight='bold', y=0.65, fontname='Arial')

# Set the title text
title_text = '''
The dataset contains information on a Portuguese bank's direct marketing \
campaigns via phone calls to customers. It includes demographic data,\n \
contact history, loan and credit default information, and account balance. \
The campaigns aimed to promote term deposit subscriptions, with\n data \
indicating subscription outcomes. The infographic analyzes feature \
correlations with term deposit subscriptions.\nLink to dataset: \
https://www.kaggle.com/datasets/prakharrathi25/banking-dataset-marketing\
-targets
'''
title_ax.text(0.5, 0.35, title_text, ha='center', va='center', 
              fontsize='18', fontweight='bold', fontstyle='italic', 
              bbox=dict(boxstyle='round,pad=0', facecolor=(0.8, 0.9, 1.0), 
                        edgecolor=(0.5, 0.5, 0.5), linewidth=1.5))

# Set the description text
description_text = '''
The data visualization analysis reveals several trends and patterns. \
The majority of subscribers have job types classified as blue-collar,\
\nmanagement, and technicians, with secondary or tertiary education. \
Many of them do not have house loans. The age distribution of subscribers\
\nranges from 30 to 50 years, and their account balances are typically \
up to 2000 euros. Additionally, the call duration required to persuade\n\
customers for a term deposit mostly falls within the range of up to 600 \
seconds.
'''
description_ax.text(0.5, 0.5, description_text, ha='center', 
                    va='center', fontsize='18', fontweight='bold', 
                    fontstyle='italic', 
                    bbox=dict(boxstyle='round,pad=0', 
                              facecolor=(0.8, 0.9, 1.0), 
                              edgecolor=(0.5, 0.5, 0.5), linewidth=1.5))
# Save the fig with 300 dpi
plt.savefig('22034204.png', dpi=300)

# Show the plot
plt.show()


        































