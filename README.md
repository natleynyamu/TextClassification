# Sentiment and Text Classification for Product Rating Optimization Model



## Description

The Sentiment and Text Classification for Product Rating Optimization App is a web application that revolutionizes e-commerce product ratings, leveraging deep insights from customer reviews to ensure a more accurate and trustworthy representation of product quality. The model is trained on comments data users have given and it has been preprocessed to ensure better predictions. The app has been deployed on Streamlite. [Here](https://drive.google.com/file/d/1vJLKe8ahDObe6aATvkZo_Xja7d_FVIIG/view?usp=sharing) is how the app works.

Note: Deployed version on the web page [Here](https://product-rating-prediction.streamlit.app/)

## Notebooks and dataset

- [Dataset](https://www.kaggle.com/code/vsridhar7/customer-review-analysis-text-mining/input)

- [Regression model](https://colab.research.google.com/drive/1F6MXo_IgsNROXjzhAfiD5zamWsYUdLWK#scrollTo=0YkZ89K2pbqY)


## Features
- Data input: Users can input the text comment for prediction.

- Prediction: After inputing a desire comment about the product, users can click the "Predict" button to get the model's prediction regarding the rating the product should have based on the comment.

## Packages Used

This project has used the some packages such as numpy, tensorflow, which have to be installed to run this web app locally present in `requirements.txt` file. 

## Installation

To run the project locally, there is a need to have Visual Studio Code (vs code) installed on your PC:

- **[VS Code](https://code.visualstudio.com/download)**: It is a source-code editor made by Microsoft with the Electron Framework, for Windows, Linux, and macOS.

## Usage

1. Clone the project 

``` bash
git clone https://github.com/UmuhireJessie/product-rating-prediction.git

```

2. Open the project with vs code

``` bash
cd product-rating-prediction
code .
```

3. Install the required dependencies

``` bash
pip install -r requirements.txt
```


4. Run the project

``` bash
streamlit run app.py
```

5. Use the link printed in the terminal to visualise the app. (Usually `http://127.0.0.1:8501/`)

## Model Files

- lstm_model.h5: The main sentiment and text classification for product rating optimization model trained on customer' comments data.
- tokenizer.pkl: The tokenizer used for preprocessing the text comment.

## Authors and Acknowledgment

- Jessie Umuhire Umutesi
- Natley Nyamukondiwa
- Emile Kamana

## License
[MIT](https://choosealicense.com/licenses/mit/)
