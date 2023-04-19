from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# 載入模型
lr_model = joblib.load('lr_model.pkl')


@app.route('/predict', methods=['POST'])
def predict():
    # 獲取用戶提供的特徵
    budget = float(request.form['budget'])
    rating = float(request.form['rating'])
    feature = [[budget, rating]]

    # 進行預測
    prediction = lr_model.predict(feature)

    # 返回預測結果
    return render_template('result.html', prediction=prediction[0])


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
