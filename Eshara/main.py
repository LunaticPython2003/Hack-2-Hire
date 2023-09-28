from flask import Flask, Response, render_template

app = Flask(__name__)
@app.route('/convert')
def test():
    return render_template('converter.html')

@app.route('/')
def home():
    return render_template('home.html')

if __name__=="__main__":
    app.run()