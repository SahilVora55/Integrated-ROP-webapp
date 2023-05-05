import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import base64
import time
import os
import matplotlib.pyplot as plt
import plotly.express as px
from io import BytesIO
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import pickle
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Define function for plotting graph.
def plot_graphs(df, x_cols, y_cols, graph_type, plot_size):
    fig = px.line(df, x=x_cols, y=y_cols) if graph_type == "line" else px.scatter(df, x=x_cols, y=y_cols)
    if graph_type == "line-scatter":
        fig.add_trace(px.line(df, x=x_cols, y=y_cols).data[0])
    fig.update_layout(width=plot_size[0], height=plot_size[1], plot_bgcolor='#FFFFFF', paper_bgcolor='#F2F2F2',
                      xaxis=dict(showgrid=True, gridcolor='#DDDDDD', linecolor='#999999', linewidth=1,
                                 mirror=True, title_font=dict(size=14)),
                      yaxis=dict(showgrid=True, gridcolor='#DDDDDD', linecolor='#999999', linewidth=1,
                                 mirror=True, title_font=dict(size=14)),
                      font=dict(size=12),
                      margin=dict(t=50, b=50, r=50, l=50),
                      hoverlabel=dict(bgcolor='#FFFFFF', font_size=12, font_family="Arial"))

    st.plotly_chart(fig)

    # Download button
    filename = f"{graph_type}.png"
    data = fig.to_image(format="png")
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:file/png;base64,{b64}" download="{filename}">Download Plot</a>'
    st.markdown(href, unsafe_allow_html=True)

def plot_boxplot(df, column, plot_size):
    fig = px.box(df, y=column)
    fig.update_layout(width=plot_size[0], height=plot_size[1], plot_bgcolor='#FFFFFF', paper_bgcolor='#F2F2F2',
                      xaxis=dict(showgrid=False, zeroline=False, linecolor='#999999', linewidth=1,
                                 mirror=True, tickfont=dict(size=12)),
                      yaxis=dict(showgrid=True, gridcolor='#DDDDDD', linecolor='#999999', linewidth=1,
                                 mirror=True, tickfont=dict(size=12)),
                      font=dict(size=12),
                      margin=dict(t=50, b=50, r=50, l=50),
                      hoverlabel=dict(bgcolor='#FFFFFF', font_size=12, font_family="Arial"))

    st.plotly_chart(fig)

    # Download button
    filename = "Boxplot.png"
    data = fig.to_image(format="png")
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:file/png;base64,{b64}" download="{filename}">Download Plot</a>'
    st.markdown(href, unsafe_allow_html=True)

def plot_histogram(df, column, plot_size):
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    fig = px.histogram(df, x=column, nbins=30, marginal="rug", opacity=0.7, color_discrete_sequence=["#FCB711"])
    fig.update_layout(width=plot_size[0], height=plot_size[1], plot_bgcolor='#FFFFFF', paper_bgcolor='#F2F2F2',
                      xaxis=dict(showgrid=True, gridcolor='#DDDDDD', linecolor='#999999', linewidth=1,
                                 mirror=True, title_font=dict(size=14)),
                      yaxis=dict(showgrid=True, gridcolor='#DDDDDD', linecolor='#999999', linewidth=1,
                                 mirror=True, title_font=dict(size=14)),
                      font=dict(size=12),
                      margin=dict(t=50, b=50, r=50, l=50),
                      hoverlabel=dict(bgcolor='#FFFFFF', font_size=12, font_family="Arial"))
    st.plotly_chart(fig)
    
    # Download button
    filename = "Histogram.png"
    data = fig.to_image(format="png")
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:file/png;base64,{b64}" download="{filename}">Download Plot</a>'
    st.markdown(href, unsafe_allow_html=True)

st.set_page_config(page_title='ROP Prediction App', page_icon=':chart_with_upwards_trend:', layout='wide')
# Add some custom CSS to adjust the sidebar width
st.markdown(
    """
    <style>
    .css-1aumxhk {
        max-width: 250px;
    }
    </style>
    """,
    unsafe_allow_html=True,
) 
# initial_sidebar_state="collapsed"

# Define page background color and font styles
PAGE_BG = "#f2f2f2"
HEADER_FONT = ("Verdana", 35, "bold")
SUBHEADER_FONT = ("Verdana", 20, "italic")
TEXT_FONT = ("Verdana", 15)

# Define a list of options
menu_options = ["Home", "Clean the data", "Data Visualization", "Craft an Machine-Learning model","Real-Time Prediction using Continuous approach","Help!"]

# Add a radio button to the sidebar to display the options
selected_option = st.sidebar.radio("Select an option", menu_options)


if selected_option == "Home":
    
    st.header('Rate of Penetration (ROP) prediction web-app')

    # Define header image and text
    header_image = Image.open("header.png")
    st.image(header_image, use_column_width=True)

    st.write('The **rate of penetration (ROP)**, expressed in feet per hour (ft/h), is a critical metric used in the oil and gas industry to evaluate drilling efficiency. **ROP refers to the speed at which the drill bit advances into the formation being drilled**, and it is influenced by various parameters such as the **formation characteristics, drilling parameters**, and **drilling fluid properties**.')
    st.write('Predicting ROP accurately is essential for **optimizing drilling operations, reducing drilling time,** and **minimizing drilling costs**. It can also help in identifying potential drilling hazards, such as **bit wear, formation changes**, or **drilling fluid-related issues**.')

    col1, col2, col3 = st.columns(3)

    with col1:
        img_path = "cleandata.png"

        image = open(img_path, 'rb').read()
        image_base64 = base64.b64encode(image).decode('utf-8')
        image_html = f'<img src="data:image/png;base64,{image_base64}" style="height:150px; padding:35px;" alt="Clean Data Image">'

        st.markdown(image_html, unsafe_allow_html=True)

    with col2:
        img_path = "viz.png"

        image = open(img_path, 'rb').read()
        image_base64 = base64.b64encode(image).decode('utf-8')
        image_html = f'<img src="data:image/png;base64,{image_base64}" style="height:130px; padding:25px;" alt="Visualization Image">'

        st.markdown(image_html, unsafe_allow_html=True)

    with col3:
        img_path = "ml.png"

        image = open(img_path, 'rb').read()
        image_base64 = base64.b64encode(image).decode('utf-8')
        image_html = f'<img src="data:image/png;base64,{image_base64}" style="height:130px; padding:25px;" alt="ML image">'

        st.markdown(image_html, unsafe_allow_html=True)

    st.write('Below you will find a comprehensive introduction for each section.')

    # Set CSS style for the box
    box_style = (
            f"background: linear-gradient(to bottom, #E7F2F8, #D4E7F0);"
            f"border-radius: 10px;"
            f"padding: 10px;"
            f"height: 300px;"
        )

    coly,colz = st.columns(2)
    with coly:
        # Section 1: Clean the Data
        st.write(
            f'<div style="{box_style}">',
            '<h3 style="text-align: left; color: black;">1) Clean the Data</h3>',
            '<p style="text-align: justify; color: black;">The application offers a user-friendly interface that enables the uploading of <strong>CSV</strong> or <strong>Excel</strong> files for data cleaning. For example, if the user is working with <strong>drilling data</strong> and wants to predict the <strong>Rate of Penetration (ROP)</strong>, they can upload the data to the app, remove irrelevant columns, rename columns, and fill in missing data. After cleaning the data, they can download it for further analysis or modeling. The process is simple, efficient, and empowers users to work with accurate and reliable data.</p>',
            '</div>',
            unsafe_allow_html=True,
        )

    st.write('')

    with colz:
        # Section 2: Visualize the Data
        st.write(
            f'<div style="{box_style}">',
            '<h3 style="text-align: left; color: black;">2) Visualize the Data</h3>',
            '<p style="text-align: justify; color: black;">After cleaning the data, users can visualize the data using different types of plots, such as <strong>histograms, scatter plots</strong>, and <strong>line plots</strong>. This allows users to gain insights into the data and identify any patterns or trends that may exist. The application provides various visualization options that can be customized to suit the user\'s needs.</p>',
            '</div>',
            unsafe_allow_html=True,
        )

    st.write('')

    # Set CSS style for the box
    box_style = (
            f"background: linear-gradient(to bottom, #E7F2F8, #B5D7E7);"
            f"border-radius: 10px;"
            f"padding: 10px;"
            f"height: 600px;"
        )
    cols,colt = st.columns(2)
    with cols:
        # Section 3: Craft an Machine-Learning model
        st.write(
            f'<div style="{box_style}">',
            '<h3 style="text-align: left; color: black;">3) Craft a Machine-Learning model</h3>',
            '<p style="text-align: justify; color: black;">Predicting the <strong>Rate of Penetration (ROP)</strong> is an important aspect of drilling operations. Machine Learning (ML) models can be used to predict the ROP based on various input parameters such as Weight on Bit (WOB), Standpipe Pressure (SPP), and Surface Torque (ST). After data cleaning and visualization, the following steps can be taken to predict the ROP using ML:</p>',
            '<ul style="text-align: left; color: black;">',
            '<li>Select a suitable ML model.</li>',
            '<li>Train the ML model using the training set.</li>',
            '<li>Evaluate the performance of the trained model using the testing set.</li>',
            '<li>Adjusting the hyperparameters</li>',
            '<li>Predict the ROP for new input values.</li>',
            '</ul>',
            '</div>',
            unsafe_allow_html=True,
        )

    with colt:
        # Section 4: Real-Time Prediction using Continuous approach
        st.write(
            f'<div style="{box_style}">',
            '<h3 style="text-align: left; color: black;">4) Real-Time Prediction using Continuous approach</h3>',
            '<p style="text-align: justify; color: black;">The <strong>continuous approach</strong> for predicting the <strong>Rate of Penetration (ROP)</strong> in drilling operations using Machine Learning (ML) involves the development of a model that can predict ROP in real-time as the drilling operation progresses. This approach requires a continuous feed of data from various sensors and instruments that are monitoring the drilling process, such as weight on bit, standpipe pressure, surface torque, and other factors that can influence ROP.</p>',
            '<p style="text-align: justify; color: black;">The model is being updated with <strong>each new depth drilled</strong> or after every iteration, and it predicts the <strong>rate of penetration (ROP)</strong> for the <strong>next section</strong>. The ROP prediction is based on various inputs such as the <strong>drilling parameters</strong>, <strong>geological data</strong>, and <strong>historical ROP data</strong>, which are analyzed using machine learning algorithms. <strong>By continuously updating the model, it can adapt to changing drilling conditions and provide more accurate predictions, helping to optimize drilling operations and improve efficiency</strong>.</p>',
            '</div>',
            unsafe_allow_html=True,
        )

    st.write('')
    st.write("The '**Help**' section provides a guide with step-by-step instructions for working with each section.")

if selected_option == "Clean the data":
    # Add a title to the app
    st.header('Data Cleaning')
    # Add a file uploader to the app
    uploaded_file = st.file_uploader('Upload your data file', type=['csv', 'xlsx'])

    # Add a checkbox to skip header row
    skip_header = st.checkbox('Skip header row',True)

    # Display the uploaded data file
    if uploaded_file is not None:
        # Load the data file into a Pandas DataFrame
        if uploaded_file.type == 'application/vnd.ms-excel':
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        else:
            df = pd.read_csv(uploaded_file, header= 0 if skip_header else None)
        
        col10, col11 = st.columns([1,3])
        with col10:
            st.write('**Uploaded Data:**')
        with col11:
            # Display the shape of the data
            st.write(f'Shape of data is : <span style="color:blue">{df.shape}</span>', unsafe_allow_html=True)
        st.dataframe(df, height=220)

        # Select columns to keep
        st.write('**Column Selection:**')
        cols_to_keep = []
        col_chunks = [df.columns[i:i+4] for i in range(0, len(df.columns), 4)]

        # Display columns in chunks of 4
        for cols_chunk in col_chunks:
            cols_column = st.columns(len(cols_chunk))
            for i, col in enumerate(cols_chunk):
                if cols_column[i].checkbox(col):
                    cols_to_keep.append(col)

        # Keep only the selected columns
        if cols_to_keep:
            df = df[cols_to_keep]
            st.write('Dataframe with Selected Columns:')
            st.dataframe(df, height=220)
        else:
            st.write('No columns were selected.')

        st.write('**Rename selected columns:**')
        rename = st.checkbox('Rename columns')
        if rename:
            st.write('**Rename Columns:**')
            col1, col2, col3, col4 = st.columns(4)  # create 4 columns

            # Loop through each column and create a text input in each column
            rename_map = {}
            for i, col in enumerate(df.columns):
                with locals()[f"col{i % 4 + 1}"]:
                    new_col_name = st.text_input(f'Rename "{col}" to:', col)
                    if new_col_name != col:
                        rename_map[col] = new_col_name

            if rename_map:
                df = df.rename(columns=rename_map)
                st.write('Renamed Columns:')
                st.dataframe(df,height=220)
            else:
                st.write('No columns were renamed.')


        # Number of missing value in each column.
        st.write('**Missing value:**')
        if st.checkbox('Show number of missing data in each selected columns.'):
            st.write(df.isnull().sum())

        # Drop or Fill NaN values
        st.write('**Drop or Fill NaN values:**')
        if st.checkbox('Process for missing value Values'):
            # Replace missing values with NaN
            df.replace({'': np.nan, ' ': np.nan, 'NaN': np.nan, 'N/A': np.nan, 'n/a': np.nan, 'na': np.nan}, inplace=True)

            # Remove leading and trailing whitespaces
            df = df.applymap(lambda x: x.strip() if type(x) == str else x)

            # Choose fill method
            fill_methods = st.selectbox('Select a data filling method:', ['No Fill', 'Fill with Mean', 'Fill with Median', 'Fill with Mode', 'Fill with ffill', 'Fill with bfill'])

            if fill_methods == 'No Fill':
                # Drop NaN
                df.dropna(inplace=True)
            elif fill_methods == 'Fill with Mean':
                # Fill NaN with mean value
                df.fillna(df.mean(), inplace=True)
                df.dropna(inplace=True)
            elif fill_methods == 'Fill with Median':
                # Fill NaN with median value
                df.fillna(df.median(), inplace=True)
                df.dropna(inplace=True)
            elif fill_methods == 'Fill with Mode':
                # Fill NaN with mode value
                df.fillna(df.mode().iloc[0], inplace=True)
                df.dropna(inplace=True)
            elif fill_methods == 'Fill with ffill':
                # Fill NaN with forward fill method
                df.fillna(method='ffill', inplace=True)
                df.dropna(inplace=True)
            elif fill_methods == 'Fill with bfill':
                # Fill NaN with backward fill method
                df.fillna(method='bfill', inplace=True)
                df.dropna(inplace=True)


            # Display the cleaned data
            col7,col8 = st.columns([1,3])
            with col7:
                st.write('**Cleaned Data:**')
            with col8:
                st.write(f'Shape of cleaned data is : <span style="color:blue">{df.shape}</span>', unsafe_allow_html=True)
            st.dataframe(df,height=220)
            st.write('<span style="color:green">Successfully processed NaN values', unsafe_allow_html=True)


        col15, col16 = st.columns([2,4])

        with col15:
            # Add a download button to download the cleaned data
            cleaned_data = df.to_csv(index=False)
            if st.download_button(
                label='Download Cleaned Data',
                data=cleaned_data,
                file_name='cleaned_data.csv',
                mime='text/csv',
                use_container_width = True):
                with col16:
                    st.write('<span style="color:green">Downloaded successfully.', unsafe_allow_html=True)
        
        # Save the dataframe in session state
        st.session_state['clean_df'] = df



if selected_option == "Data Visualization":

    st.title('Data Visualization')

    st.write("Choose data source")
    data_option = st.radio("Select data source:", ("Upload new data", "Use existing cleaned data"))

     # Load data based on option selected
    if data_option == "Upload new data":
        uploaded_file = st.file_uploader("Choose a file", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            # Save the dataframe in session state
            st.session_state['clean_df'] = df
            df = st.session_state['clean_df']
        else:
            st.warning('Please upload a file!') # Display warning if no file is uploaded
            st.stop()
    else:
        # Use session state data
        if "clean_df" in st.session_state:
            df = st.session_state['clean_df']
        else:
            st.warning('Please upload a file first!') # Display warning if no file is uploaded and session state is empty

    if "clean_df" in st.session_state:
        # Display data
        if st.checkbox('**Display data**'):
            st.dataframe(df, height=220)
        col1, col_extra, col2 = st.columns([6,1,9])
        with col1:
            graph_type = st.selectbox("**Select the type of graph**", ["line", "scatter", "line-scatter", "boxplot", "histplot","Heatmap (correlation metrix)"])
            if graph_type in ["boxplot", "histplot"]:
                cols = st.selectbox("Select the column to plot", df.columns)
            if graph_type in ["line", "scatter"]:
                col3, col4 = st.columns(2)
                with col3:
                    # Select X-axis column
                    x_cols = st.selectbox("Select the X-axis column(s)", df.columns)
                with col4:
                    # Select Y-axis column(s)
                    y_cols = st.multiselect("Select the Y-axis column(s)", df.columns)
            if graph_type in ["line-scatter"]:
                col7, col8 = st.columns(2)
                with col7:
                    # Select X-axis column
                    x_cols = st.selectbox("Select the X-axis column(s)", df.columns)
                with col8:
                    # Select Y-axis column(s)
                    default_y_cols = [df.columns[1]]
                    y_cols = st.multiselect("Select the Y-axis column(s)", df.columns, default=default_y_cols)

            if graph_type in ["Heatmap (correlation metrix)"]:            
                # Compute correlation matrix
                corr_matrix = df.corr()

                # Get correlation values for 'Rate of Penetration'
                ROP = st.selectbox('Please select the column that contains the values for the **rate of penetration**.',df.columns)
                corr_values = corr_matrix[ROP]

                # Select columns with correlation greater than 0.25
                suggested_cols = corr_values[(corr_values > 0.25)|(corr_values < -0.25)].drop(ROP)

                # Display the suggested columns
                st.markdown('<p style="color: green"><b>Suggested columns which has good correlation with Rate of Penetration:</b></p>', unsafe_allow_html=True)
                st.write(suggested_cols)



            col5,col6 = st.columns(2)
            with col5:
                plot_size_x = st.slider("Select the width of plot", 100, 1000, 700)
            with col6:
                plot_size_y = st.slider("Select the height of plot", 100, 1000, 450)
            plot_size = (plot_size_x, plot_size_y)

            Graph = st.checkbox("**Display Graph**",True)


        with col2:
            st.write('<h3>Graph:</h3>', unsafe_allow_html=True)
            if Graph:
                if graph_type in ["line", "scatter", "line-scatter"]:
                    plot_graphs(df, x_cols, y_cols, graph_type, plot_size)
                elif graph_type == "boxplot":
                    plot_boxplot(df, cols, plot_size)
                elif graph_type == "histplot":
                    plot_histogram(df, cols, plot_size)
                else:
                    # Display heatmap
                    fig, ax = plt.subplots()
                    sns.heatmap(df.corr(), cmap='YlGnBu', annot=True, ax=ax)
                    st.pyplot(fig)

                    # Download button
                    filename = "Heatmap.png"
                    canvas = FigureCanvas(fig)
                    png_output = BytesIO()
                    canvas.print_png(png_output)
                    data = png_output.getvalue()
                    b64 = base64.b64encode(data).decode()
                    href = f'<a href="data:file/png;base64,{b64}" download="{filename}">Download Plot</a>'
                    st.markdown(href, unsafe_allow_html=True)


        # add vertical line between the columns
        st.markdown("""<style> .stHorizontal { display: none; } </style>""", unsafe_allow_html=True)
        col1.markdown("""<hr style='height: 2px; background-color: #8c8c8c;'></hr>""", unsafe_allow_html=True)



if selected_option == "Craft an Machine-Learning model":

    # Define the app
    def app():
        # Set the page title and description
        st.title('ROP Prediction App')
        st.markdown('This app predicts the Rate of Penetration (ROP) using real-time drilling data.')
        
        st.write("-----")
        st.write("")
        st.write("")

        if 'df_pred' not in st.session_state:
            st.session_state['df_pred'] = pd.DataFrame()

        if 'model' not in st.session_state:
            st.session_state['model'] = None

        # Upload the drilling data
        st.subheader('Make a Machine Learning model to predict the Rate of Penetration (ROP).')
        
        st.write("Choose data source")
        data_option = st.radio("Select data source:", ("Upload new data", "Use existing cleaned data"))

        # Load data based on option selected
        if data_option == "Upload new data":
            uploaded_file = st.file_uploader("Choose a file", type="csv")
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                # Save the dataframe in session state
                st.session_state['clean_df'] = df
                df = st.session_state['clean_df']
            else:
                st.warning('Please upload a file!') # Display warning if no file is uploaded
                st.stop()
        else:
            # Use session state data
            if "clean_df" in st.session_state:
                df = st.session_state['clean_df']
            else:
                st.warning('Please upload a file first!') # Display warning if no file is uploaded and session state is empty

        if "clean_df" in st.session_state:

            # Display the drilling data
            st.write('Drilling Data:')
            st.dataframe(df, height=200)

            # Select the prediction model
            st.subheader('Selecting Prediction Model and Features')
            model_name = st.selectbox('Select the prediction model', ['Random Forest Regression', 'Gradient Boosting Regression', 'XGBoost Regression', 'Decision Tree Regression','K-Nearest Neighbors','Artificial Neural Network'])

            col1, col2 = st.columns(2)

            with col1:
                # Select the Rate of Penetration column
                target_column = st.selectbox('Select the Rate of Penetration column', list(df.columns), key='target_column')
            with col2:
                # Select the input features for the ROP prediction model
                selected_features = st.multiselect('Select the input features', list(df.drop('Rate of Penetration m/h',axis=1).columns))

            # Set the model parameters based on the selected model
            st.write(f'<h3 style="font-size:16px;">Adjust model parameters for {model_name}</h3>', unsafe_allow_html=True)

            if model_name == 'Random Forest Regression':
                # Create three columns with equal width
                col1, col2, col3 = st.columns(3)
                # Add sliders to each column
                with col1:
                    n_estimators = st.slider('Number of Trees', min_value=10, max_value=500, value=100, step=10)
                with col2:
                    max_depth = st.slider('Max Depth', min_value=1, max_value=20, value=10)
                with col3:
                    min_samples_split = st.slider('Min Samples Split', min_value=2, max_value=20, value=5)
                # Create a dictionary with the slider values
                model_params = {'n_estimators': n_estimators,
                                'max_depth': max_depth,
                                'min_samples_split': min_samples_split}
                model = RandomForestRegressor(**model_params)

            elif model_name == 'Gradient Boosting Regression':
                # Create three columns with equal width
                col1, col2, col3 = st.columns(3)

                # Add sliders to each column
                with col1:
                    n_estimators = st.slider('Number of Trees', min_value=10, max_value=500, value=100, step=10)

                with col2:
                    max_depth = st.slider('Max Depth', min_value=1, max_value=20, value=10)

                with col3:
                    min_samples_split = st.slider('Min Samples Split', min_value=2, max_value=20, value=5)

                # Create a dictionary with the slider values
                model_params = {'n_estimators': n_estimators,
                                'max_depth': max_depth,
                                'min_samples_split': min_samples_split}

                model = GradientBoostingRegressor(**model_params)
            elif model_name == 'XGBoost Regression':
                # Create three columns with equal width
                col1, col2, col3 = st.columns(3)

                # Add sliders to each column
                with col1:
                    n_estimators = st.slider('Number of Trees', min_value=10, max_value=500, value=100, step=10)

                with col2:
                    max_depth = st.slider('Max Depth', min_value=1, max_value=20, value=10)

                with col3:
                    learning_rate = st.slider('Learning Rate', min_value=0.01, max_value=0.5, value=0.1, step=0.01)

                # Create a dictionary with the slider values
                model_params = {'n_estimators': n_estimators,
                                'max_depth': max_depth,
                                'learning_rate': learning_rate}

                model = XGBRegressor(**model_params)
            elif model_name == 'Decision Tree Regression':
                # Create two columns with equal width
                col1, col2 = st.columns(2)

                # Add sliders to each column
                with col1:
                    max_depth = st.slider('Max Depth', min_value=1, max_value=20, value=10)

                with col2:
                    min_samples_split = st.slider('Min Samples Split', min_value=2, max_value=20, value=5)

                # Create a dictionary with the slider values
                model_params = {'max_depth': max_depth,
                                'min_samples_split': min_samples_split}

                model = DecisionTreeRegressor(**model_params)

            elif model_name == 'K-Nearest Neighbors':
                # Create three columns with equal width
                col1, col2, col3 = st.columns(3)

                # Add sliders to each column
                with col1:
                    n_neighbors = st.slider('Number of Neighbors', min_value=1, max_value=50, value=5)

                with col2:
                    weights = st.selectbox('Weights', options=['uniform', 'distance'])

                with col3:
                    algorithm = st.selectbox('Algorithm', options=['auto', 'ball_tree', 'kd_tree', 'brute'])

                # Create a dictionary with the slider and selectbox values
                model_params = {'n_neighbors': n_neighbors,
                                'weights': weights,
                                'algorithm': algorithm}

                model = KNeighborsRegressor(**model_params)


            elif model_name == 'Artificial Neural Network':
                # Create three columns with equal width
                col1, col2, col3 = st.columns(3)

                # Add sliders to each column
                with col1:
                    hidden_layer_sizes = st.slider('Hidden Layer Sizes', min_value=10, max_value=1000, value=100, step=10)

                with col2:
                    activation = st.selectbox('Activation Function', options=['identity', 'logistic', 'tanh', 'relu'])

                with col3:
                    solver = st.selectbox('Solver', options=['lbfgs', 'sgd', 'adam'])

                # Create a dictionary with the slider and selectbox values
                model_params = {'hidden_layer_sizes': (hidden_layer_sizes,),
                                'activation': activation,
                                'solver': solver}

                # Create the ANN model with a specific max_iter value to increase speed
                model = MLPRegressor(max_iter=1000, **model_params)

    #        else:
    #            # Create two columns with equal width
    #            col1, col2 = st.columns(2)
    #
    #            # Add sliders to each column
    #            with col1:
    #                generations = st.slider('Number of generations', min_value=5, max_value=50, value=10, step=5)
    #
    #            with col2:
    #                population_size = st.slider('Population size', min_value=10, max_value=100, value=50, step=10)
    #
    #            # Create a dictionary with the slider values
    #            model_params = {'generations': generations,
    #                            'population_size': population_size}
    #
    #            model = TPOTRegressor(generations=model_params['generations'], population_size=model_params['population_size'], verbosity=0, random_state=42)


            # Select test size
            st.write('<h3 style="font-size:16px;">Train-test split</h3>', unsafe_allow_html=True)
            test_size = st.slider('Select Test Size', min_value=0.1, max_value=0.5, value=0.2, step=0.01)

            # Make the ROP prediction
            button_html = '<button style="background-color: lightgreen; color: white; font-size: 16px; padding: 0.5em 1em; border-radius: 5px; border: none;">Make Prediction</button>'
            if st.button('Make Prediction',use_container_width=True):
                st.text('Prediction in progress...')  # display message while prediction is happening
                # Check if the target column exists in the input data
                if target_column not in df.columns:
                    st.warning(f'The input data does not have a column named "{target_column}". Please upload valid drilling data.')
                else:
                    # Preprocess the input data
                    X = df[selected_features]
                    y = df[target_column]
                    
                    # Train-test split
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

                    # Fit the model to the data
                    model.fit(X_train, y_train)

                    # Serialize the model using pickle
                    if model is not None:
                        model_bytes = pickle.dumps(model)
                    else:
                        st.warning('please make an ml model')
                        st.stop()

                    # Save the serialized model in session state
                    if model_bytes is not None:
                        st.session_state['model'] = model_bytes
                    else:
                        st.warning('please make an ml model first!')

                    # Predict the ROP
                    y_pred_train = model.predict(X_train)
                    y_pred_test = model.predict(X_test)

                    st.success('Prediction successful!')  # display success message after prediction

                    # Save the model to a file
                    filename = 'model.pkl'
                    with open(filename, 'wb') as file:
                        pickle.dump(model, file)

                    # Load the saved model
                    with open(filename, 'rb') as file:
                        model = pickle.load(file)

                    # Encode the model file to base64
                    with open(filename, 'rb') as f:
                        bytes_data = f.read()
                    b64 = base64.b64encode(bytes_data).decode()

                    # Create a download link for the model file
                    href = f'<a href="data:file/model.pkl;base64,{b64}" download="model.pkl">Download Trained Model (.pkl)</a>'
                    st.markdown(href, unsafe_allow_html=True)

                    # Calculate MAE & R2 Score
                    MAE_train = mean_absolute_error(y_train, y_pred_train)
                    MAE_test = mean_absolute_error(y_test, y_pred_test)
                    R2_train = r2_score(y_train, y_pred_train)
                    R2_test = r2_score(y_test, y_pred_test)

                    st.subheader('Result')
                    col1, col2, col3 = st.columns([1,1,2])
                    with col1:    
                        st.write('for training data\n- R2-score: <span style="color:#007D5C;font-weight:bold">{:.3f}</span>\n- MAE: <span style="color:#007D5C;font-weight:bold">{:.3f}</span>'.format(R2_train, MAE_train), unsafe_allow_html=True)
                    with col2:    
                        st.write('for testing data\n- R2-score: <span style="color:#007D5C;font-weight:bold">{:.3f}</span>\n- MAE: <span style="color:#007D5C;font-weight:bold">{:.3f}</span>'.format(R2_test, MAE_test), unsafe_allow_html=True)
                    with col3:
                        # Display the ROP prediction
                        
                        df_pred1 = pd.DataFrame({'ROP_actual':y_test,'ROP_pred': y_pred_test})
                        X_test['ROP_actual'] = df_pred1['ROP_actual']
                        X_test['ROP_pred'] = df_pred1['ROP_pred']
                        st.session_state['df_pred'] = pd.concat([st.session_state['df_pred'], X_test], axis=0)

                        # Add a download button to download the dataframe as a CSV file
                        csv = st.session_state['df_pred'].to_csv(index=False)
                        b64 = base64.b64encode(csv.encode()).decode()  # some strings
                        href = f'<a href="data:file/csv;base64,{b64}" download="st.session_state[\'df_pred\'].csv">Download predicted ROP data</a>'
                        st.markdown(href, unsafe_allow_html=True)

                        # Display the dataframe in Streamlit
                        st.write('ROP Predicted:')
                        st.dataframe(st.session_state['df_pred'][['ROP_actual','ROP_pred']], height=200,width=400)

        # Add a blank line between the buttons
        st.write("")
        st.write("")
        st.write("")
        st.write("________")

        # Create a file uploader widget
        st.subheader('Calculate ROP using created machine learning model.')
        # model_file = st.file_uploader("Upload a saved ML model (pkl)", type=["pkl"])

        # # If a file has been uploaded, load the model from the file
        # if model_file is not None:
        #     with model_file:
        #         model = pickle.load(model_file)

        #     # Display the loaded model
        #     if 'model' in locals():
        #         st.write("<span style='color:green; font-weight:bold'>Model loaded successfully!</span>", unsafe_allow_html=True)
        #         st.write(model)

        st.write("Choose data source")
        data_option = st.radio("Select the model source:", ("Upload new ML model","Continue with above ML model"))

        # Load data based on option selected
        if data_option == "Upload new ML model":
            model_file = st.file_uploader("Upload a saved ML model (pkl)", type=["pkl"])
            if model_file is not None:
                with model_file:
                    model = pickle.load(model_file)
            else:
                st.warning('Please upload a model!') # Display warning if no file is uploaded
                st.stop()
        else:
            # Use session state data
            if "model" in st.session_state:
                # Load the serialized model from session state
                model_bytes = st.session_state['model']
                # Deserialize the model using pickle
                if model_bytes is not None:
                    model = pickle.loads(model_bytes)
                else:
                    st.warning('Please make an ml model first!')
                    st.stop()
            else:
                st.warning('Please upload a file first!') # Display warning if no file is uploaded and session state is empty
                
        

        if model is not None:
        # Display the loaded model
            if 'model' in locals():
                st.write("<span style='color:green; font-weight:bold'>Model loaded successfully!</span>", unsafe_allow_html=True)
                st.write(model)

            # Get the list of column names
            columns = st.session_state['df_pred'].drop(['ROP_actual','ROP_pred'],axis=1).columns.tolist()

            # Create a row of input boxes using beta_columns
            input_cols = st.columns(min(len(columns), 5))
            input_array = np.zeros(len(columns))
            for i, input_col in enumerate(input_cols):
                input_value = input_col.number_input(label=columns[i], step=0.1, value=0.0, min_value=0.0, max_value=1000000.0, key=columns[i])
                input_array[i] = input_value
                
            # Create additional rows of input boxes if there are more than 5 columns
            if len(columns) > 5:
                for j in range(5, len(columns), 5):
                    input_cols = st.columns(min(len(columns)-j, 5))
                    for i, input_col in enumerate(input_cols):
                        input_value = input_col.number_input(label=columns[j+i], step=0.1, value=0.0, min_value=0.0, max_value=1000000.0, key=columns[j+i])
                        input_array[j+i] = input_value

            # Define colors and font sizes  
            HIGHLIGHT_COLOR = '#22c1c3'
            RESULT_FONT_SIZE = '36px'

            if st.button('Calculate ROP'):
                input_array = input_array.reshape(1, -1)
                rop = model.predict(input_array)
                st.success('Calculated successful!')
                # Format the output message
                result_text = f"Calculated Rate of Penetration (ROP): {rop[0]:.2f} ft/hr"
                result_html = f'<div style="font-size:{RESULT_FONT_SIZE}; color:{HIGHLIGHT_COLOR};">{result_text}</div>'
                st.markdown(result_html, unsafe_allow_html=True)


        # Add a blank line between the buttons
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("__________")

    # Run the app
    if __name__ == '__main__':
        app()



# Define the countdown timer function
def countdown_timer():
    remaining_time = st.empty()
    #bar = st.progress(0)
    for i in range(1, 11):
        remaining_time.markdown(f"<p style='color: green;'>Time remaining for the next batch of data to come: <span style='font-weight: bold;'>{11 - i} second</span></p>", unsafe_allow_html=True)
        #bar.progress(int((i / 15) * 100))
        time.sleep(1)
    remaining_time.markdown("<p style='color: green; font-weight: bold;'>We have new data available for model update and prediction.</p>", unsafe_allow_html=True)
    
if selected_option == "Real-Time Prediction using Continuous approach":
    st.header('Real-Time Rate of Penetration Prediction using Continuous approach')
    #st.set_page_config(page_title='Real-time ROP Prediction App', page_icon=':chart_with_upwards_trend:', layout='wide')
    uploaded_data = st.file_uploader("Please upload a file", type="csv")

    if uploaded_data is not None:
        try:
            df = pd.read_csv(uploaded_data)
            st.dataframe(df,height=220)
        except Exception as e:
            st.write("Error:", e)
        cold, cole, colf = st.columns([8,1,6])
        with cold:
            depth_col = st.selectbox('Select the **depth column** to iterate over', options=df.columns)
        with colf:
            step = st.number_input('Please specify the **step size (in meters)** for splitting the data.', min_value=10, value=50, step=10)
        button_clicked = st.checkbox('Make an machine learning model')

        if button_clicked:
            min_depth = df[depth_col].min()
            max_depth = df[depth_col].max()
            depths = range(int(min_depth), int(max_depth)+step, step)

            # Create a dictionary to store the resulting dataframes
            dfs = {}

            for i in range(len(depths)-1):
                depth_start = depths[i]
                depth_end = depths[i+1]

                #st.write(f"\nData for depth {depth_start} m to {depth_end} m:")
                depth_data = df[(df[depth_col] >= depth_start) & (df[depth_col] < depth_end)]
                #st.write(depth_data)

                # Store the resulting dataframe in the dictionary
                dfs[f"df{i+1}"] = depth_data

            # Concatenate dataframes and display them
            #st.write('\nCombined dataframes:')
            df_comb = {}
            df_comb[1] = dfs['df1']

            
            for i in range(2, len(dfs)+1):
                df_list = [dfs[f"df{j}"] for j in range(1, i+1)]
                df_concat = pd.concat(df_list, axis=0)
                #st.write(df_concat)
                df_comb[i] = df_concat

            # st.write(df_comb[1])
            # st.write(df_comb[2])
            # st.write(df_comb[3])

            # make an ml model to predict ROP continuously.

            # Select the prediction model
            st.subheader('Selecting Prediction Model and Features')
            model_name = st.selectbox('**Select the prediction model**', ['Random Forest Regression', 'Gradient Boosting Regression', 'XGBoost Regression', 'Decision Tree Regression','K-Nearest Neighbors'])
            cola, colb, colc = st.columns([8,1,12])
            with cola:
                ROP = st.selectbox('**Select the target column**',df_comb[1].columns)
            with colc:
                features = st.multiselect('**Select the features**', df_comb[1].drop(ROP, axis=1).columns)

            # Set the model parameters based on the selected model
            st.write(f'<h3 style="font-size:16px;">Adjust model parameters for {model_name}</h3>', unsafe_allow_html=True)

            if model_name == 'Random Forest Regression':
                # Create three columns with equal width
                col1, col2, col3 = st.columns(3)
                # Add sliders to each column
                with col1:
                    n_estimators = st.slider('Number of Trees', min_value=10, max_value=500, value=100, step=10)
                with col2:
                    max_depth = st.slider('Max Depth', min_value=1, max_value=20, value=10)
                with col3:
                    min_samples_split = st.slider('Min Samples Split', min_value=2, max_value=20, value=5)
                # Create a dictionary with the slider values
                model_params = {'n_estimators': n_estimators,
                                'max_depth': max_depth,
                                'min_samples_split': min_samples_split}
                model = RandomForestRegressor(**model_params)

            elif model_name == 'Gradient Boosting Regression':
                # Create three columns with equal width
                col1, col2, col3 = st.columns(3)

                # Add sliders to each column
                with col1:
                    n_estimators = st.slider('Number of Trees', min_value=10, max_value=500, value=100, step=10)

                with col2:
                    max_depth = st.slider('Max Depth', min_value=1, max_value=20, value=10)

                with col3:
                    min_samples_split = st.slider('Min Samples Split', min_value=2, max_value=20, value=5)

                # Create a dictionary with the slider values
                model_params = {'n_estimators': n_estimators,
                                'max_depth': max_depth,
                                'min_samples_split': min_samples_split}

                model = GradientBoostingRegressor(**model_params)
            elif model_name == 'XGBoost Regression':
                # Create three columns with equal width
                col1, col2, col3 = st.columns(3)

                # Add sliders to each column
                with col1:
                    n_estimators = st.slider('Number of Trees', min_value=10, max_value=500, value=100, step=10)

                with col2:
                    max_depth = st.slider('Max Depth', min_value=1, max_value=20, value=10)

                with col3:
                    learning_rate = st.slider('Learning Rate', min_value=0.01, max_value=0.5, value=0.1, step=0.01)

                # Create a dictionary with the slider values
                model_params = {'n_estimators': n_estimators,
                                'max_depth': max_depth,
                                'learning_rate': learning_rate}

                model = XGBRegressor(**model_params)
            elif model_name == 'Decision Tree Regression':
                # Create two columns with equal width
                col1, col2 = st.columns(2)

                # Add sliders to each column
                with col1:
                    max_depth = st.slider('Max Depth', min_value=1, max_value=20, value=10)

                with col2:
                    min_samples_split = st.slider('Min Samples Split', min_value=2, max_value=20, value=5)

                # Create a dictionary with the slider values
                model_params = {'max_depth': max_depth,
                                'min_samples_split': min_samples_split}

                model = DecisionTreeRegressor(**model_params)

            elif model_name == 'K-Nearest Neighbors':
                # Create three columns with equal width
                col1, col2, col3 = st.columns(3)

                # Add sliders to each column
                with col1:
                    n_neighbors = st.slider('Number of Neighbors', min_value=1, max_value=50, value=5)

                with col2:
                    weights = st.selectbox('Weights', options=['uniform', 'distance'])

                with col3:
                    algorithm = st.selectbox('Algorithm', options=['auto', 'ball_tree', 'kd_tree', 'brute'])

                # Create a dictionary with the slider and selectbox values
                model_params = {'n_neighbors': n_neighbors,
                                'weights': weights,
                                'algorithm': algorithm}

                model = KNeighborsRegressor(**model_params)

            # Select the test size.
            test_size = st.slider('Select Test Size', min_value=0.1, max_value=0.5, value=0.2, step=0.01)

            # # Set the initial delay time to 0 seconds
            # delay = 0
            #Create new dataframe to save the results
            df_result = pd.DataFrame({
                'Iteration': [],
                'R2_score_train': [],
                'R2_score_test': [],
                'MAE_train': [],
                'MAE_test': [],
                'R2_score_next_section':[]
            })
            df_ROP = pd.DataFrame(columns=['Measured Depth m','Predicted ROP'])

            predict = st.checkbox('Make Prediction!')
            if predict:
                for i in range(len(df_comb)-2):

                    # Call the countdown timer function
                    countdown_timer()

                    X = df_comb[i+1][features]
                    y = df_comb[i+1][ROP]

                    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = test_size, random_state=42)

                    model.fit(X_train,y_train)
                    # Predict the ROP
                    y_pred_train = model.predict(X_train)
                    y_pred_test = model.predict(X_test)

                    col11, col12, col13 = st.columns([6,4,4])

                    with col11:
                        st.success('Model updated successfully!')  # display success message after prediction
                        st.markdown(f"<p style='color: blue;'><b>Depth:</b> from <b>{df_comb[i+1][depth_col].min()}</b> to <b>{df_comb[i+1][depth_col].max()}</b></p>", unsafe_allow_html=True)

                        # Calculate MAE & R2 Score
                        MAE_train = mean_absolute_error(y_train, y_pred_train)
                        MAE_test = mean_absolute_error(y_test, y_pred_test)
                        R2_train = r2_score(y_train, y_pred_train)
                        R2_test = r2_score(y_test, y_pred_test)

                        st.subheader('Result')
                        col1, col2 = st.columns([1,3])
                        with col1:    
                            st.write('for training data\n- R2-score: <span style="color:#007D5C;font-weight:bold">{:.3f}</span>\n- MAE: <span style="color:#007D5C;font-weight:bold">{:.3f}</span>'.format(R2_train, MAE_train), unsafe_allow_html=True)
                        with col2:    
                            st.write('for testing data\n- R2-score: <span style="color:#007D5C;font-weight:bold">{:.3f}</span>\n- MAE: <span style="color:#007D5C;font-weight:bold">{:.3f}</span>'.format(R2_test, MAE_test), unsafe_allow_html=True)

                        X1 = df_comb[i+2][features]
                        y1 = df_comb[i+2][ROP]

                        next_pred = model.predict(X1)
                        R2_next = r2_score(y1, next_pred)

                        st.write(f"Here's the R2 score for the predicted rate of penetration of the next section,  <span style='font-weight:bold; color:green;'>{np.round(R2_next,3)}</span>", unsafe_allow_html=True)

                        # Saving the results to df_result
                        df_result_temp = pd.DataFrame({
                            'Iteration': [],
                            'R2_score_train': [],
                            'R2_score_test': [],
                            'MAE_train': [],
                            'MAE_test': [],
                            'R2_score_next_section':[]
                        })
                        df_result_temp.loc[0] = [i+1, R2_train, R2_test, MAE_train, MAE_test, R2_next]

                        # Append the new dataframe to the existing one
                        df_result = pd.concat([df_result, df_result_temp], ignore_index=True)

                    with col12:
                        st.subheader('Predicted ROP:')
                        st.markdown(f"<p style='color: blue;'><b>Depth:</b> from <b>{dfs[f'df{i+2}'][depth_col].min()}</b> to <b>{dfs[f'df{i+2}'][depth_col].max()}</b></p>", unsafe_allow_html=True)

                        df_ROP_temp = pd.DataFrame(columns=['Predicted ROP', 'Measured Depth m'])

                        # assign values to the DataFrame columns using the .loc accessor
                        df_ROP_temp.loc[:, 'Predicted ROP'] = next_pred
                        df_ROP_temp.loc[:, 'Measured Depth m'] = df_comb[i+2][depth_col]

                        df_ROP = pd.concat([df_ROP, df_ROP_temp], ignore_index=True)
                        
                        df_ROP = df_ROP.drop_duplicates(['Measured Depth m']).reset_index(drop=True)
                        df_ROP = df_ROP.iloc[len(df_ROP)-len(dfs[f'df{i+2}']):].reset_index(drop=True)

                        st.dataframe(df_ROP,height = 350)
                        # Add download button
                        csv = df_ROP.to_csv(index=False)
                        b64 = base64.b64encode(csv.encode()).decode()
                        href = f'<a href="data:file/csv;base64,{b64}" download="Predicted_ROP.csv">Download</a>'
                        st.markdown(href, unsafe_allow_html=True)

                    with col13:
                        drill_bit_marker = (6, 0, 0)
                        fig, ax = plt.subplots(figsize=(3, 10))
                        if i == 0:
                            ax.plot(dfs[f'df{i+1}'][ROP], dfs[f'df{i+1}'][depth_col], 'r', lw=2)
                            ax.plot(df_ROP['Predicted ROP'], df_ROP['Measured Depth m'], 'g',ls='--' ,lw=2)
                            ax.scatter(df_ROP['Predicted ROP'][0], df_ROP['Measured Depth m'][0],color='blue', marker=drill_bit_marker, s=300)
                            ax.set_title('Predicted ROP',fontsize = 16, fontweight = 'bold', color = 'olive')
                            ax.set_xlabel('ROP (m/h)', fontsize = 12, fontweight = 'bold', color = 'maroon')
                            ax.set_ylabel('Depth (m)', fontsize = 12, fontweight = 'bold', color = 'maroon')
                            ax.invert_yaxis()
                            ax.grid()
                            st.pyplot(fig)
                        else:
                            ax.plot(dfs[f'df{i}'][ROP], dfs[f'df{i}'][depth_col], 'r', lw=2)
                            ax.plot(dfs[f'df{i+1}'][ROP], dfs[f'df{i+1}'][depth_col], 'r', lw=2)
                            ax.plot(df_ROP['Predicted ROP'], df_ROP['Measured Depth m'], 'g',ls='--' ,lw=2)
                            ax.scatter(df_ROP['Predicted ROP'][0], df_ROP['Measured Depth m'][0],
                                        color='blue', marker=drill_bit_marker, s=300)
                            ax.set_title('Predicted ROP',fontsize = 16, fontweight = 'bold', color = 'olive')
                            ax.set_xlabel('ROP (m/h)', fontsize = 12, fontweight = 'bold', color = 'maroon')
                            ax.set_ylabel('Depth (m)', fontsize = 12, fontweight = 'bold', color = 'maroon')
                            ax.invert_yaxis()
                            ax.grid()
                            st.pyplot(fig)                       

                    st.write('---')
                        
                st.dataframe(df_result)
                # Add download button
                csv = df_result.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="results.csv">Download</a>'
                st.markdown(href, unsafe_allow_html=True)


# Add Help Section
if selected_option == "Help!":
    st.header('Welcome to the Rate of Penetration (ROP) prediction web-app.')
    st.write('')
    st.write('Kindly choose the relevant section for which you require assistance from the list provided below.')
    section = st.selectbox('**Select the section**',menu_options[1:-1])
    if section == 'Clean the data':
        st.write('')
        st.write('This section of the web-app helps you clean the **Real-time drilling data**. Here are step by step guidelines to use this app:')
        st.write('1. First, the user needs to select the **"Clean the data"** option from the sidebar menu in the app.')
        st.write('2. Next, the user needs to upload their data file by clicking on the **"Upload your data file"** button and selecting the appropriate file type (**csv** or **xlsx**).')
        st.write('3. Once the file is uploaded, the user can choose to skip the header row by checking the **"Skip header row"** checkbox.')
        st.write('4. The app displays the uploaded data in a table, along with its shape. The user can check which columns they want to keep by selecting the appropriate checkboxes.')
        st.write('5. If the user wants to rename any selected columns, they can do so by checking the **"Rename columns"** checkbox and providing new names for the columns in the provided text input fields.')
        st.write('6. The user can check the number of missing values in each selected column by checking the **"Show number of missing data in each selected columns"** checkbox.')
        st.write('7. If the user wants to fill or drop NaN values, they can do so by checking the **"Process for missing value Values"** checkbox and selecting a **data filling method** from the dropdown menu.')
        st.write('8. Once the data cleaning process is completed, the cleaned data will be displayed in a table, along with its new shape. The user can download the cleaned data by clicking on the **"Download Cleaned Data"** button.')
        st.write('9. Finally, the user can use the cleaned data for further analysis or modeling purposes.')

    if section == "Data Visualization":
        st.write('')
        st.write("Here's a step-by-step guide on how to use this section of the web-app")
        st.write('1. Select "**Data Visualization**" option from the sidebar.')
        st.write('2. Choose a appropriate data source by selecting either "**Upload new data**" or "**Use existing cleaned data**".')
        st.write('3. If you selected "Upload new data", upload your **CSV file**.')
        st.write('4. Once the data is loaded, you can choose to display the data by selecting the "**Display data**" checkbox.')
        st.write('5. Select the type of graph you want to create from the "**Select the type of graph**" dropdown menu.')
        st.write('6. Based on the type of graph selected, you may need to choose **X** and **Y axes** or a column to plot.')
        st.write('7. Choose the **width** and **height** of the plot using the sliders.')
        st.write('8. Click the "**Display Graph**" checkbox to generate the plot.')
        st.write('9. If you chose the **heatmap** (correlation matrix) graph type, you can also select the column that contains values for the rate of penetration. The app will suggest other columns with good correlation based on your selection.')
        st.write('10. you can download the plot by clicking the "**Download Plot**".')
        st.write('11. Enjoy exploring your data!')

    if section == 'Craft an Machine-Learning model':
        st.write('')
        st.write('**Make a Machine Learning model to predict the Rate of Penetration (ROP).**')
        st.write('This section of the web-app helps you predict the **Rate of Penetration (ROP)** using **Real-time drilling data**. Here are step by step guidelines to use this app:')
        st.write('1. Upload your **drilling data** in **CSV** format using the file uploader.')
        st.write('2. Select the **prediction model** from the dropdown menu.')
        st.write('3. Select the target column (in this case **Rate of Penetration (ROP)**) that you want to predict.')
        st.write('4. Select the **input features** that you want to use for the prediction.')
        st.write('5. Split your data into **training** and **testing** sets by selecting the test size.')
        st.write('6. Adjust the **model parameters** as per your requirements.')
        st.write('7. Click on the **"Predict"** button to see the **predicted ROP values**.')
        st.write('Note: This app uses **Random Forest Regression, Gradient Boosting Regression, XGBoost Regression, and Decision Tree Regression,** to predict ROP value.')
        st.write('**Calculate ROP using created machine learning model:**')
        st.write('1. The first task is to select the machine learning model source by selecting either "**Upload new ML model**" or "**Continue with above ML model**".')
        st.write('2. Once the model source is selected, the code will display the chosen model along with its respective hyperparameters.')
        st.write('3. You are now able to provide the parameter values, which will serve as input features for the machine learning model.')
        st.write('4. Upon inputting the values for the various parameters, simply press the "**Calculate**" button to obtain the corresponding **Rate of Penetration value**.')

    if section == "Real-Time Prediction using Continuous approach":
        st.write('')
        st.write('1. Open the web-app and select the "**Real-Time Prediction using Continuous approach**" option from the sidebar.')
        st.write('2. Upload your data file in **CSV** format using the "**Upload a file**" button.')
        st.write('3. Select the "***depth**" column that you want to iterate over from the dropdown list.')
        st.write('4. Specify the "**step size**" (in meters) for splitting the data into smaller chunks. (We have made the assumption that a fresh set of data will arrive after 15 seconds, which will be utilized to update the model and make predictions for the next segment.)')
        st.write('5. Click on the "**Make a machine learning model**" checkbox to start building the machine learning model.')
        st.write('6. Select the **prediction model** from the dropdown list. You can choose from **Random Forest Regression**, **Gradient Boosting Regression**, **XGBoost Regression**, and **Decision Tree Regression**.')
        st.write('7. Select the **target column** for prediction from the dropdown list.')
        st.write('8. Select the **features** that you want to use for prediction from the dropdown list.')
        st.write('9. Adjust the **model parameters** based on the selected model.')
        st.write('10. Click on the "**Predict**" button to start the prediction process.')
        st.write('11. The **predicted ROP values** will be displayed in a table along with the Depth values.')
        st.write('12. You can download the predicted ROP values in CSV format by clicking on the "**Download Predictions**" button.')
        st.write('')

    st.subheader('Write us!')
    with st.form(key='contact_form'):
        name = st.text_input(label='Name')
        email = st.text_input(label='Email')
        message = st.text_area(label='Message')
        submit_button = st.form_submit_button(label='Submit')

    if submit_button:
        data = {'Name': [name], 'Email': [email], 'Message': [message]}
        df = pd.DataFrame(data)

        if not os.path.isfile('contact_data.csv'):
            df.to_csv('contact_data.csv', index=False)
        else:
            existing_data = pd.read_csv('contact_data.csv')
            df.to_csv('contact_data.csv', mode='a', header=False, index=False)

        st.success('Thank you for your message. We will get back to you shortly.')


# Add a contact us button
st.write("")
st.write("")
st.write("")
st.write("")
st.write("")
st.write("-------------------")

email_icon = Image.open('email.png')
if st.button('**Contact Us**',use_container_width=True):
    st.image(email_icon,width=150)
    st.write('Please email us at <span style="font-size:20px">sahilvoraa@gmail.com</span>', unsafe_allow_html=True)

# Define footer and text color
st.write("")
st.write("")
st.write("")
st.write("")
st.write("")
st.write("")
st.write("")
# st.markdown("<h5 style='text-align: center; color: gray'>Created by Sahil Vora</h5>", unsafe_allow_html=True)
