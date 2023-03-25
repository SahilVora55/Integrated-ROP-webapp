import streamlit as st
import base64

st.set_page_config(page_title='ROP Prediction App', page_icon=':chart_with_upwards_trend:', layout='wide')

st.header('Rate of Penetration (ROP) prediction web-app')
st.write('The rate of penetration (ROP), expressed in feet per hour (ft/h), is a critical metric used in the oil and gas industry to evaluate drilling efficiency. ROP refers to the speed at which the drill bit advances into the formation being drilled, and it is influenced by various parameters such as the formation characteristics, drilling parameters, and drilling fluid properties.')
st.write('Predicting ROP accurately is essential for optimizing drilling operations, reducing drilling time, and minimizing drilling costs. It can also help in identifying potential drilling hazards, such as bit wear, formation changes, or drilling fluid-related issues.')


col1, col2, col3 = st.columns(3)

with col1:
    link = "https://sahilvora55-data-cleaning-webapp-clean-dataset1-cdh99t.streamlit.app/"
    img_path = "cleandata.png"

    image = open(img_path, 'rb').read()
    image_base64 = base64.b64encode(image).decode('utf-8')
    image_html = f'<img src="data:image/png;base64,{image_base64}" style="height:300px; padding:70px;" alt="Click to visit Google">'

    # Replace any invalid characters in the link
    link = link.replace('\n', '')

    st.markdown(f'<a href="{link}">{image_html}</a>', unsafe_allow_html=True)

    st.markdown("""
        <div style='background-color: #F0F8FF; padding: 10px;'>
            <span style='color: black;'>
                The first step in any ML project is to clean and preprocess the data. This data typically includes drilling parameters such as weight on bit, rotary speed, mud flow rate, and bit type, as well as geological information such as lithology, porosity, and permeability.  The data may also include information on drilling performance, such as ROP and drilling time. Once the data is collected, it must be cleaned and processed to remove any outliers, missing values, or other errors.
            </span>
        </div>
    """, unsafe_allow_html=True)

with col2:
    link = "https://sahilvora55-visualization-webapp-visualization-tah1k5.streamlit.app/"
    img_path = "viz.png"

    image = open(img_path, 'rb').read()
    image_base64 = base64.b64encode(image).decode('utf-8')
    image_html = f'<img src="data:image/png;base64,{image_base64}" style="height:255px; padding:50px;" alt="Click to visit Google">'

    # Replace any invalid characters in the link
    link = link.replace('\n', '')

    st.markdown(f'<a href="{link}">{image_html}</a>', unsafe_allow_html=True)
    st.write('')
    st.write('')
    st.write('')
    st.markdown("""
        <div style='background-color: #F0F8FF; padding: 10px;'>
            <span style='color: black;'>
                After cleaning the data, the next step is to visualize it to gain insights and identify patterns. This can be done using various visualization techniques, such as scatter plots, histograms, and box plots. Visualization can help identify correlations between drilling parameters and ROP, as well as identify any outliers or anomalies in the data. It can also help in feature selection, which involves identifying the most important drilling parameters that affect ROP.
            </span>
        </div>
    """, unsafe_allow_html=True)
with col3:
    link = "https://sahilvora55-rop-prediction-app-ml-app-6zwzcf.streamlit.app/"
    img_path = "ml.png"

    image = open(img_path, 'rb').read()
    image_base64 = base64.b64encode(image).decode('utf-8')
    image_html = f'<img src="data:image/png;base64,{image_base64}" style="height:255px; padding:50px;" alt="Click to visit Google">'

    # Replace any invalid characters in the link
    link = link.replace('\n', '')

    st.markdown(f'<a href="{link}">{image_html}</a>', unsafe_allow_html=True)
    st.write('')
    st.write('')
    st.write('')

    st.markdown("""
        <div style='background-color: #F0F8FF; padding: 10px;'>
            <span style='color: black;'>
                The final step is to craft an ML model that can predict ROP based on the selected drilling parameters. This involves choosing an appropriate ML algorithm, such as regression, decision trees, or neural networks, and training it on the cleaned and preprocessed data. The model is then tested and evaluated using validation techniques, such as cross-validation, to ensure that it can accurately predict ROP. The model can then be used to predict ROP in real-time drilling operations, helping to optimize drilling performance and reduce costs.
            </span>
        </div>
    """, unsafe_allow_html=True)